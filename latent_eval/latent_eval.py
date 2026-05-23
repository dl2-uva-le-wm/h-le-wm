#!/usr/bin/env python3
"""
Latent trajectory evaluation for hi_lewm_p2_train_hope2 checkpoint.

For each of 50 evaluation episodes:
  1. Encodes expert frames (from dataset) at every timestep  →  "expert latents"
  2. Runs the hierarchical policy in PushT and encodes the frames the agent
     actually observes                                        →  "model latents"
  3. Saves all rows to latent_eval/results/latent_trajectories.csv
  4. Saves one PNG per episode in latent_eval/results/plots/

Dimensionality reduction: UMAP (falls back to PCA if umap-learn is absent).

Architecture constants are derived from:
  config/train/hi_lewm.yaml + data=hi_pusht + hope2 training script overrides.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as T

# ── repo root ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import stable_pretraining as spt
import stable_worldmodel as swm

# Trigger dynamic module registration required for checkpoint unpickling.
import baseline_adapter as _ba
_ = _ba.ARPredictor

from baseline_adapter import ARPredictor, Embedder, MLP
from hi_jepa import HiJEPA
from hi_vq import VQActionEncoder
from hi_policy import HierarchicalWorldModelPolicy, calibrate_latent_prior
from hi_eval import sample_eval_row_indices, get_episodes_length

# ── architecture constants (hope2 training config) ─────────────────────────────
_ENCODER_SCALE  = "tiny"
_PATCH_SIZE     = 14
_IMG_SIZE       = 224
_EMBED_DIM      = 192        # wm.embed_dim  (== ViT-tiny hidden_size)
_HIDDEN_DIM     = 192        # encoder.config.hidden_size
_ACTION_DIM     = 2          # PushT action (x, y)
_FRAMESKIP      = 5          # data.dataset.frameskip
_EFF_ACT_DIM    = _FRAMESKIP * _ACTION_DIM    # 10
_LATENT_ACT_DIM = 32         # wm.high_level.latent_action_dim (hope2 default)
_NUM_WP         = 5          # wm.high_level.waypoints.num
_HIGH_NF        = _NUM_WP - 1                 # 4 — high predictor context
_HISTORY_SIZE   = 3          # wm.history_size
_MAX_SPAN       = 15         # waypoints.max_span  == latent_action_encoder.max_seq_len
_VQ_CODES       = 128        # latent_action_encoder.vq.num_codes
_PRED_KWARGS    = dict(depth=6, heads=16, mlp_dim=2048, dim_head=64,
                       dropout=0.1, emb_dropout=0.0)


# ── device ─────────────────────────────────────────────────────────────────────
def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


# ── image transform (matches hi_eval.py / hi_diagnostics.py) ──────────────────
def make_transform() -> T.Compose:
    stats = spt.data.dataset_stats.ImageNet
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(**stats),
        T.Resize(size=_IMG_SIZE),
    ])


# ── model building ─────────────────────────────────────────────────────────────
def _build_hi_jepa() -> HiJEPA:
    """Instantiate HiJEPA from hardcoded hope2 architecture."""
    encoder = spt.backbone.utils.vit_hf(
        _ENCODER_SCALE,
        patch_size=_PATCH_SIZE,
        image_size=_IMG_SIZE,
        pretrained=False,
        use_mask_token=False,
    )
    low_pred  = ARPredictor(num_frames=_HISTORY_SIZE,
                            input_dim=_EMBED_DIM, hidden_dim=_HIDDEN_DIM,
                            output_dim=_HIDDEN_DIM, **_PRED_KWARGS)
    high_pred = ARPredictor(num_frames=_HIGH_NF,
                            input_dim=_EMBED_DIM, hidden_dim=_HIDDEN_DIM,
                            output_dim=_HIDDEN_DIM, **_PRED_KWARGS)
    act_enc   = Embedder(input_dim=_EFF_ACT_DIM, emb_dim=_EMBED_DIM)
    proj      = MLP(input_dim=_HIDDEN_DIM, output_dim=_EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    lo_proj   = MLP(input_dim=_HIDDEN_DIM, output_dim=_EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    hi_proj   = MLP(input_dim=_HIDDEN_DIM, output_dim=_EMBED_DIM,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    lat_enc   = VQActionEncoder(
        input_dim=_EFF_ACT_DIM, latent_dim=_LATENT_ACT_DIM,
        num_codes=_VQ_CODES, model_dim=_EMBED_DIM,
        num_layers=2, num_heads=4, mlp_dim=768, dropout=0.1,
        max_seq_len=_MAX_SPAN, decoder_hidden_dim=768,
    )
    # latent_action_dim (32) != embed_dim (192)  →  linear projection
    m2c = torch.nn.Linear(_LATENT_ACT_DIM, _EMBED_DIM)
    return HiJEPA(
        encoder=encoder, low_predictor=low_pred, action_encoder=act_enc,
        high_predictor=high_pred, latent_action_encoder=lat_enc,
        macro_to_condition=m2c, projector=proj,
        low_pred_proj=lo_proj, high_pred_proj=hi_proj,
    )


def load_model(ckpt_path: Path, device: torch.device) -> HiJEPA:
    """Load HiJEPA from a Lightning _weights.ckpt or pickled object checkpoint."""
    print(f"[load_model] {ckpt_path.name}  →  {device}")
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Pickled model object (_object.ckpt style)
    if isinstance(raw, torch.nn.Module):
        model = getattr(raw, "model", raw)
        return model.to(device).eval()

    # Lightning checkpoint dict
    if isinstance(raw, dict):
        sd = raw.get("state_dict", raw)

        # spt.Module stores model as self.model  →  prefix "model."
        model_sd = {k[len("model."):]: v for k, v in sd.items()
                    if k.startswith("model.")}
        if not model_sd:
            model_sd = {k: v for k, v in sd.items()
                        if not any(k.startswith(p) for p in ("sigreg.", "optimizer",))}

        model = _build_hi_jepa()
        miss, unexp = model.load_state_dict(model_sd, strict=False)
        if miss:
            warnings.warn(f"[load_model] {len(miss)} missing keys, e.g. {miss[:3]}")
        if unexp:
            warnings.warn(f"[load_model] {len(unexp)} unexpected keys, e.g. {unexp[:3]}")
        return model.to(device).eval()

    raise TypeError(f"Unknown checkpoint type: {type(raw)}")


# ── dataset helpers ────────────────────────────────────────────────────────────
def ep_col_name(dataset) -> str:
    for c in ("episode_idx", "ep_idx"):
        if c in dataset.column_names:
            return c
    raise ValueError("Dataset missing episode column.")


def build_ep_row_map(dataset, ep_col: str) -> dict[int, dict[int, int]]:
    """ep_id → step_idx → dataset_row."""
    eps   = np.asarray(dataset.get_col_data(ep_col))
    steps = np.asarray(dataset.get_col_data("step_idx"))
    m: dict[int, dict[int, int]] = {}
    for row, (ep, st) in enumerate(zip(eps, steps)):
        m.setdefault(int(ep), {})[int(st)] = row
    return m


def build_process_map(dataset, keys: list[str]) -> dict:
    process = {}
    for col in keys:
        if col == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        scaler.fit(data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


# ── frame encoding ─────────────────────────────────────────────────────────────
@torch.inference_mode()
def encode_frames(
    model: HiJEPA,
    frames: np.ndarray,        # (N, H, W, C)  uint8
    tfm: T.Compose,
    device: torch.device,
) -> np.ndarray:               # (N, D_z)
    px = torch.stack([tfm(f) for f in frames], dim=0).to(device)  # (N, C, H, W)
    out = model.encode({"pixels": px.unsqueeze(1)}, encode_actions=False)
    return out["emb"][:, -1].cpu().float().numpy()


# ── environment helpers ────────────────────────────────────────────────────────
def _call_env_method(vec_env, method: str, arg) -> None:
    """Try several ways to call a method on the underlying environment."""
    try:
        vec_env.call(method, [arg])
        return
    except Exception:
        pass
    try:
        vec_env.envs[0].__getattribute__(method)(arg)
        return
    except Exception:
        pass
    warnings.warn(f"[env] could not call {method}; initial state may be wrong.")


# ── policy wrapper ─────────────────────────────────────────────────────────────
class _FakeEnv:
    """Minimal env stub required by HierarchicalWorldModelPolicy.set_env()."""
    num_envs = 1
    import gymnasium as _gym
    action_space = _gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1, _ACTION_DIM), dtype=np.float32
    )


class LatentPolicy:
    """
    Wraps HierarchicalWorldModelPolicy; records the encoded latent of
    every observation *before* dispatching to the planner.

    With record_subgoals=True (requires --record-subgoals flag), also records
    z_subgoal after each high-level replan for temporal-alignment analysis.
    """

    def __init__(
        self,
        model: HiJEPA,
        process: dict,
        transform: dict,
        device: torch.device,
        high_bounds: dict | None = None,
        *,
        high_n_samples: int = 300,
        low_n_samples:  int = 200,
        seed: int = 42,
        record_subgoals: bool = False,
    ):
        self.model  = model
        self.device = device
        self.step_latents: list[np.ndarray] = []
        self._record_subgoals = record_subgoals
        self.subgoal_records: list[dict] = []

        import gymnasium as gym
        high_cfg = swm.policy.PlanConfig(horizon=2, receding_horizon=1, action_block=1)
        low_cfg  = swm.policy.PlanConfig(horizon=5, receding_horizon=1, action_block=5)

        high_solver = swm.solver.CEMSolver(
            model=model, batch_size=1,
            num_samples=high_n_samples, var_scale=1.0,
            n_steps=10, topk=max(1, high_n_samples // 30),
            device=str(device), seed=seed,
        )
        low_solver = swm.solver.CEMSolver(
            model=model, batch_size=1,
            num_samples=low_n_samples, var_scale=1.0,
            n_steps=15, topk=max(1, low_n_samples // 10),
            device=str(device), seed=seed,
        )

        self._policy = HierarchicalWorldModelPolicy(
            model=model,
            high_solver=high_solver,
            low_solver=low_solver,
            high_config=high_cfg,
            low_config=low_cfg,
            macro_replan_interval=5,
            process=process,
            transform=transform,
            high_latent_bounds=high_bounds,
        )
        self._policy.set_env(_FakeEnv())

    def reset_episode(self) -> None:
        self.step_latents    = []
        self.subgoal_records = []
        p = self._policy
        p._action_buffer = deque(maxlen=p.flatten_receding_horizon_low)
        p._next_low_init  = None
        p._next_high_init = None
        p._z_subgoal      = None
        p._steps_since_high = 10 ** 9

    @torch.inference_mode()
    def _record(self, pixels_nhwc: np.ndarray) -> None:
        """Encode current obs frame and record latent (no planning side-effects)."""
        tfm = self._policy.transform.get("pixels")
        frames_chw = []
        for frame in pixels_nhwc:
            if tfm is not None:
                from torchvision import tv_tensors
                t = tfm(tv_tensors.Image(
                    torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
                ))
                frames_chw.append(t.squeeze(0))
            else:
                frames_chw.append(
                    torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                )
        px = torch.stack(frames_chw, 0).to(self.device)   # (N, C, H, W)
        out = self.model.encode(
            {"pixels": px.unsqueeze(1)}, encode_actions=False
        )
        z = out["emb"][:, -1]                               # (N, D_z)
        self.step_latents.append(z[0].cpu().float().numpy())

    def step(self, info: dict) -> np.ndarray:
        """Record latent, then plan and return action."""
        t      = len(self.step_latents)
        pixels = info["pixels"]                             # (1, T, H, W, C)
        self._record(pixels[:, -1])                         # last frame in history
        action = self._policy.get_action(info)

        if self._record_subgoals:
            # _steps_since_high is set to 0 inside _plan_high, then incremented once
            # by get_action. So == 1 means a high-level replan just fired.
            replanned = (self._policy._steps_since_high == 1)
            z_sub = self._policy._z_subgoal
            if z_sub is not None:
                self.subgoal_records.append(dict(
                    timestep=t,
                    replanned=replanned,
                    z_subgoal=z_sub[0].detach().cpu().float().numpy(),
                ))

        return action


# ── rollout ────────────────────────────────────────────────────────────────────
def rollout_episode(
    vec_env,
    policy: LatentPolicy,
    dataset,
    ep_row_map: dict[int, dict[int, int]],
    ep_id: int,
    start_step: int,
    goal_offset: int,
    eval_budget: int,
) -> list[np.ndarray]:
    """Run one episode, return per-step model latents."""
    ep_rows   = ep_row_map.get(ep_id, {})
    start_row = ep_rows.get(start_step)
    goal_row  = ep_rows.get(start_step + goal_offset)

    if start_row is None or goal_row is None:
        warnings.warn(f"ep={ep_id} start={start_step}: row not found, skipping.")
        return []

    start_data = dataset.get_row_data(np.array([start_row]))
    goal_data  = dataset.get_row_data(np.array([goal_row]))
    state      = np.asarray(start_data["state"])[0]
    goal_state = np.asarray(goal_data["state"])[0]
    goal_px    = np.asarray(goal_data["pixels"])[0]   # (H, W, C)

    # reset + inject initial state
    obs, _ = vec_env.reset()
    _call_env_method(vec_env, "_set_state",      state)
    _call_env_method(vec_env, "_set_goal_state", goal_state)

    # try to re-fetch obs after state injection
    try:
        obs, _ = vec_env.reset(options={"skip_init": True})
    except Exception:
        pass

    policy.reset_episode()

    goal_batch = goal_px[None, None]   # (1, 1, H, W, C)

    for _t in range(eval_budget):
        raw = vec_env.render()
        px = raw[0] if isinstance(raw, (list, tuple)) else raw  # (H, W, C)
        px = px[None, None]   # (1, 1, H, W, C)

        action = policy.step({"pixels": px, "goal": goal_batch})
        obs, _rew, done, trunc, _ = vec_env.step(action)
        if np.any(done) or np.any(trunc):
            break

    return list(policy.step_latents)


# ── expert latents ─────────────────────────────────────────────────────────────
def expert_latents(
    model: HiJEPA,
    dataset,
    ep_row_map: dict[int, dict[int, int]],
    ep_id: int,
    start_step: int,
    eval_budget: int,
    tfm: T.Compose,
    device: torch.device,
) -> list[np.ndarray]:
    ep_rows = ep_row_map.get(ep_id, {})
    lats = []
    for t in range(eval_budget):
        row = ep_rows.get(start_step + t)
        if row is None:
            break
        px = np.asarray(dataset.get_row_data(np.array([row]))["pixels"])[0]
        lats.append(encode_frames(model, px[None], tfm, device)[0])
    return lats


# ── CSV ─────────────────────────────────────────────────────────────────────────
def save_csv(records: list[dict], path: Path, D: int) -> None:
    lat_cols   = [f"lat_{i}" for i in range(D)]
    fieldnames = ["trajectory_id", "episode_id", "timestep", "source"] + lat_cols
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row = {
                "trajectory_id": r["trajectory_id"],
                "episode_id":    r["episode_id"],
                "timestep":      r["timestep"],
                "source":        r["source"],
            }
            for i, v in enumerate(r["latent"]):
                row[f"lat_{i}"] = f"{float(v):.6f}"
            w.writerow(row)
    print(f"[CSV] {len(records)} rows  →  {path}")


def save_subgoal_csv(
    all_subgoal_records: list[dict],   # [{trajectory_id, episode_id, timestep, replanned, z_subgoal}, …]
    path: Path,
) -> None:
    """Save subgoal records produced by LatentPolicy.subgoal_records."""
    if not all_subgoal_records:
        return
    D = len(all_subgoal_records[0]["z_subgoal"])
    sg_cols    = [f"sg_{i}" for i in range(D)]
    fieldnames = ["trajectory_id", "episode_id", "timestep", "replanned"] + sg_cols
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_subgoal_records:
            row = {
                "trajectory_id": r["trajectory_id"],
                "episode_id":    r["episode_id"],
                "timestep":      r["timestep"],
                "replanned":     r["replanned"],
            }
            for i, v in enumerate(r["z_subgoal"]):
                row[f"sg_{i}"] = f"{float(v):.6f}"
            w.writerow(row)
    print(f"[CSV] {len(all_subgoal_records)} subgoal rows  →  {path}")


# ── dimensionality reduction ───────────────────────────────────────────────────
def fit_reducer(X: np.ndarray):
    try:
        import umap
        r = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        r.fit(X)
        print("[reducer] UMAP fitted")
        return r
    except ImportError:
        from sklearn.decomposition import PCA
        r = PCA(n_components=2, random_state=42)
        r.fit(X)
        print("[reducer] PCA fitted (install umap-learn for UMAP)")
        return r


# ── per-episode plot ───────────────────────────────────────────────────────────
def plot_episode(
    reducer,
    exp_lats: list[np.ndarray],
    mdl_lats: list[np.ndarray],
    traj_id: int,
    ep_id: int,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    out_dir.mkdir(parents=True, exist_ok=True)

    def proj(lats):
        if not lats:
            return np.empty((0, 2))
        return reducer.transform(np.stack(lats, axis=0))

    exp_2d = proj(exp_lats)
    mdl_2d = proj(mdl_lats)

    fig, ax = plt.subplots(figsize=(7, 6))

    def draw(pts, label, color):
        if len(pts) == 0:
            return
        # color gradient along trajectory
        n = len(pts)
        colors = cm.Blues(np.linspace(0.4, 1.0, n)) if "exp" in label.lower() \
                 else cm.Reds(np.linspace(0.4, 1.0, n))
        for i in range(n - 1):
            ax.plot(pts[i:i+2, 0], pts[i:i+2, 1],
                    "-", color=colors[i], linewidth=1.5, alpha=0.85)
        ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(n),
                   cmap="Blues" if "exp" in label.lower() else "Reds",
                   s=18, zorder=4, alpha=0.9)
        # arrows every ~6 steps
        step = max(1, n // 6)
        for i in range(0, n - 1, step):
            ax.annotate("", xy=pts[i+1], xytext=pts[i],
                        arrowprops=dict(arrowstyle="->",
                                        color=colors[i], lw=1.2))
        # start / end markers
        ax.scatter(*pts[0],  marker="s", s=70, color=colors[0],
                   zorder=5, label=f"{label} start")
        ax.scatter(*pts[-1], marker="*", s=100, color=colors[-1],
                   zorder=5, label=f"{label} end")

    draw(exp_2d, "Expert", "blue")
    draw(mdl_2d, "Model",  "red")

    rname = type(reducer).__name__
    ax.set_title(f"Latent trajectory — traj {traj_id}  ep {ep_id}\n"
                 f"{rname}  ■=start  ★=end  (darker = later)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()

    out_path = out_dir / f"traj_{traj_id:03d}_ep{ep_id}.png"
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


# ── argument parsing ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default=str(REPO_ROOT / "hi_lewm_p2_train_hope2_22253175_weights.ckpt"))
    p.add_argument("--dataset-name",  default="pusht_expert_train")
    p.add_argument("--cache-dir",     default=None)
    p.add_argument("--num-eval",      type=int, default=50)
    p.add_argument("--goal-offset",   type=int, default=25)
    p.add_argument("--eval-budget",   type=int, default=50)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--device",        default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--output-dir",
                   default=str(REPO_ROOT / "latent_eval" / "results"))
    p.add_argument("--high-num-samples", type=int, default=300)
    p.add_argument("--low-num-samples",  type=int, default=200)
    p.add_argument("--record-subgoals",  action="store_true",
                   help="Record z_subgoal at each high-level replan for temporal-alignment analysis")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args       = parse_args()
    device     = resolve_device(args.device)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device  : {device}")
    print(f"output  : {out_dir}")

    # ── dataset ───────────────────────────────────────────────────────────────
    cache_dir = Path(
        args.cache_dir
        or os.environ.get("STABLEWM_HOME", "")
        or swm.data.utils.get_cache_dir()
    )
    keys_to_cache = ["action", "proprio", "state"]
    dataset = swm.data.HDF5Dataset(
        args.dataset_name, keys_to_cache=keys_to_cache, cache_dir=cache_dir
    )
    ep_col    = ep_col_name(dataset)
    ep_row_m  = build_ep_row_map(dataset, ep_col)
    ep_ids, _ = np.unique(np.asarray(dataset.get_col_data(ep_col)), return_index=True)

    # ── sample episodes ───────────────────────────────────────────────────────
    ep_lens       = get_episodes_length(dataset, ep_ids)
    max_start     = ep_lens - args.goal_offset - 1
    ms_dict       = {int(e): int(m) for e, m in zip(ep_ids, max_start)}
    step_arr      = np.asarray(dataset.get_col_data("step_idx"))
    ep_arr        = np.asarray(dataset.get_col_data(ep_col))
    ms_per_row    = np.array([ms_dict.get(int(e), -1) for e in ep_arr])
    valid_mask    = step_arr <= ms_per_row
    valid_idx     = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points")

    sampled       = sample_eval_row_indices(valid_idx, args.num_eval, args.seed)
    row_data      = dataset.get_row_data(sampled)
    eval_eps      = np.asarray(row_data[ep_col]).tolist()
    eval_starts   = np.asarray(row_data["step_idx"]).tolist()

    # ── model ─────────────────────────────────────────────────────────────────
    model = load_model(Path(args.checkpoint), device)
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)

    tfm     = make_transform()
    process = build_process_map(dataset, keys_to_cache)

    # calibrate latent prior
    class _LPCfg:
        _d = dict(enabled=True, num_chunks=1024, min_chunks_for_stats=64,
                  chunk_len=5, lower_q=5.0, upper_q=95.0,
                  margin_ratio=0.05, clamp_abs=3.0, fallback_abs=1.0)
        def get(self, k, default=None): return self._d.get(k, default)

    print("Calibrating latent bounds …")
    high_bounds = calibrate_latent_prior(
        model=model, dataset=dataset, cfg=_LPCfg(),
        process=process, seed=args.seed,
    )

    # ── policy ────────────────────────────────────────────────────────────────
    transform_dict = {"pixels": tfm, "goal": tfm}
    policy = LatentPolicy(
        model, process, transform_dict, device,
        high_bounds=high_bounds,
        high_n_samples=args.high_num_samples,
        low_n_samples=args.low_num_samples,
        seed=args.seed,
        record_subgoals=args.record_subgoals,
    )

    # ── vectorised env (num_envs=1, reused across episodes) ───────────────────
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=2 * args.eval_budget,
        history_size=1,
        frame_skip=1,
        image_shape=(_IMG_SIZE, _IMG_SIZE),
    )
    vec_env = world.envs

    # ── collect latents ───────────────────────────────────────────────────────
    records: list[dict] = []
    ep_data: list[tuple] = []           # (traj_id, ep_id, exp_lats, mdl_lats)
    all_subgoal_records: list[dict] = []

    for traj_id, (ep_id, start_step) in enumerate(
            zip(eval_eps, eval_starts)):
        ep_id      = int(ep_id)
        start_step = int(start_step)
        print(f"\n[{traj_id+1}/{args.num_eval}] ep={ep_id}  start={start_step}")

        exp_lats = expert_latents(
            model, dataset, ep_row_m,
            ep_id, start_step, args.eval_budget, tfm, device,
        )
        print(f"  expert steps : {len(exp_lats)}")

        mdl_lats = rollout_episode(
            vec_env, policy, dataset, ep_row_m,
            ep_id, start_step, args.goal_offset, args.eval_budget,
        )
        print(f"  model  steps : {len(mdl_lats)}")

        for t, z in enumerate(exp_lats):
            records.append(dict(trajectory_id=traj_id, episode_id=ep_id,
                                timestep=t, source="expert", latent=z))
        for t, z in enumerate(mdl_lats):
            records.append(dict(trajectory_id=traj_id, episode_id=ep_id,
                                timestep=t, source="model", latent=z))
        ep_data.append((traj_id, ep_id, exp_lats, mdl_lats))

        if args.record_subgoals:
            for sg in policy.subgoal_records:
                all_subgoal_records.append(dict(
                    trajectory_id=traj_id, episode_id=ep_id,
                    **sg,
                ))
            print(f"  subgoals     : {len(policy.subgoal_records)}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if records:
        D = len(records[0]["latent"])
        save_csv(records, out_dir / "latent_trajectories.csv", D)

    if args.record_subgoals and all_subgoal_records:
        save_subgoal_csv(all_subgoal_records, out_dir / "subgoal_records.csv")

    # ── dimensionality reduction + plots ─────────────────────────────────────
    all_vecs = [r["latent"] for r in records]
    if len(all_vecs) < 4:
        print("Not enough latents to fit reducer; skipping plots.")
        return

    print("\nFitting reducer on all latents …")
    reducer = fit_reducer(np.stack(all_vecs, axis=0))

    print("Generating per-episode plots …")
    for traj_id, ep_id, exp_lats, mdl_lats in ep_data:
        plot_episode(reducer, exp_lats, mdl_lats,
                     traj_id, ep_id, out_dir / "plots")

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
