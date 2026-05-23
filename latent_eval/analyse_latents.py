#!/usr/bin/env python3
"""
Offline analysis of latent_trajectories.csv produced by latent_eval.py.

Extensions:
  1. Per-step drift     — cosine similarity + L2 between expert[t] and model[t]
  2. Latent velocity    — ||z[t+1]-z[t]|| per step; spikes reveal replan churn
  3. Subgoal analysis   — reads subgoal_records.csv if present; checks temporal alignment
  4. Fréchet distance   — distribution-level shift between expert and model clouds
  5. DTW distance       — temporally-flexible alignment (requires dtaidistance)

Usage:
    python latent_eval/analyse_latents.py \
        --csv latent_eval/results/latent_trajectories.csv \
        --output-dir latent_eval/results/analysis \
        --replan-interval 5
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_latents(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    lat_cols = [c for c in df.columns if c.startswith("lat_")]
    if not lat_cols:
        raise ValueError(f"No lat_* columns found in {csv_path}")
    print(f"[load] {len(df)} rows, {len(lat_cols)}-dim latents, "
          f"{df['trajectory_id'].nunique()} trajectories")
    return df, lat_cols


def save_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"[save] {len(records)} rows → {path}")


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# ── Extension 1: Per-step drift ────────────────────────────────────────────────

def analyse_drift(
    df: pd.DataFrame,
    lat_cols: list[str],
    replan_interval: int,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Cosine similarity and L2 distance between expert[t] and model[t] per timestep.
    Reveals when and how fast the model trajectory diverges.
    """
    records = []
    for traj_id, g in df.groupby("trajectory_id"):
        exp = (g[g.source == "expert"]
               .set_index("timestep")[lat_cols])
        mdl = (g[g.source == "model"]
               .set_index("timestep")[lat_cols])
        common = exp.index.intersection(mdl.index)
        if len(common) == 0:
            continue
        E = exp.loc[common].values
        M = mdl.loc[common].values
        cos = cosine_similarity(E, M).diagonal()
        l2  = np.linalg.norm(E - M, axis=1)
        for t, c, d in zip(common, cos, l2):
            records.append(dict(traj_id=int(traj_id), t=int(t),
                                cosine=float(c), l2=float(d)))

    drift_df = pd.DataFrame(records)
    save_csv(records, out_dir / "drift.csv")

    # Aggregate: mean ± std over trajectories per timestep
    agg = drift_df.groupby("t").agg(
        cos_mean=("cosine", "mean"), cos_std=("cosine", "std"),
        l2_mean=("l2", "mean"),     l2_std=("l2", "std"),
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ax, (col_mean, col_std, label, color) in zip(axes, [
        ("cos_mean", "cos_std", "Cosine similarity (expert vs model)", "steelblue"),
        ("l2_mean",  "l2_std",  "L2 distance (expert vs model)",       "firebrick"),
    ]):
        ax.plot(agg.t, agg[col_mean], color=color, lw=2, label=label)
        ax.fill_between(agg.t,
                        agg[col_mean] - agg[col_std].fillna(0),
                        agg[col_mean] + agg[col_std].fillna(0),
                        alpha=0.25, color=color)
        for k in range(0, int(agg.t.max()) + 1, replan_interval):
            ax.axvline(k, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel(label)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Timestep")
    axes[0].set_title(f"Per-step drift  (dashed = replan every {replan_interval} steps)")
    fig.tight_layout()
    savefig(fig, out_dir / "drift.png")
    return drift_df


# ── Extension 2: Latent velocity ──────────────────────────────────────────────

def analyse_velocity(
    df: pd.DataFrame,
    lat_cols: list[str],
    replan_interval: int,
    out_dir: Path,
) -> pd.DataFrame:
    """
    ||z[t+1] - z[t]|| per step for expert and model.
    CEM churn produces periodic spikes in model velocity at replan events.
    """
    records = []
    for traj_id, g in df.groupby("trajectory_id"):
        for source, sg in g.groupby("source"):
            sg = sg.sort_values("timestep")
            Z  = sg[lat_cols].values
            if len(Z) < 2:
                continue
            vel = np.linalg.norm(np.diff(Z, axis=0), axis=1)
            ts  = sg["timestep"].values[1:]
            for t, v in zip(ts, vel):
                records.append(dict(traj_id=int(traj_id), source=source,
                                    t=int(t), velocity=float(v)))

    vel_df = pd.DataFrame(records)
    save_csv(records, out_dir / "velocity.csv")

    agg = vel_df.groupby(["source", "t"]).agg(
        v_mean=("velocity", "mean"), v_std=("velocity", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    for source, color in [("expert", "steelblue"), ("model", "firebrick")]:
        sub = agg[agg.source == source]
        ax.plot(sub.t, sub.v_mean, color=color, lw=2, label=source)
        ax.fill_between(sub.t,
                        sub.v_mean - sub.v_std.fillna(0),
                        sub.v_mean + sub.v_std.fillna(0),
                        alpha=0.25, color=color)
    for k in range(0, int(agg.t.max()) + 1, replan_interval):
        ax.axvline(k, color="gray", lw=0.8, ls="--", alpha=0.5)

    # Ratio: model/expert velocity at each timestep
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    exp_v = agg[agg.source == "expert"].set_index("t")["v_mean"]
    mdl_v = agg[agg.source == "model"].set_index("t")["v_mean"]
    common_t = exp_v.index.intersection(mdl_v.index)
    ratio = mdl_v.loc[common_t] / exp_v.loc[common_t].replace(0, np.nan)
    ax2.plot(common_t, ratio, color="purple", lw=2)
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    for k in range(0, int(common_t.max()) + 1, replan_interval):
        ax2.axvline(k, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax2.set_ylabel("Model / Expert velocity ratio")
    ax2.set_xlabel("Timestep")
    ax2.set_title("Velocity ratio (>1 = model moves faster than expert)")
    fig2.tight_layout()
    savefig(fig2, out_dir / "velocity_ratio.png")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("||z[t+1] − z[t]||")
    ax.set_title(f"Latent velocity  (dashed = replan every {replan_interval} steps)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir / "velocity.png")
    return vel_df


# ── Extension 3: Subgoal analysis ─────────────────────────────────────────────

def analyse_subgoals(
    df: pd.DataFrame,
    lat_cols: list[str],
    subgoal_csv: Path,
    replan_interval: int,
    out_dir: Path,
) -> None:
    """
    For each recorded subgoal z_sub at timestep t, measure:
      - dist(z_sub, expert[t + offset])  for offset in [1, 2, …, replan_interval+2]
    The offset with minimal distance reveals the model's effective temporal lookahead.
    Expected: minimum at offset == replan_interval confirms correct temporal framing;
              minimum at offset < replan_interval confirms the temporal-offset bug.
    """
    if not subgoal_csv.exists():
        print(f"[subgoal] {subgoal_csv} not found — skipping (re-run latent_eval.py "
              "with --record-subgoals to generate it)")
        return

    sg_df = pd.read_csv(subgoal_csv)
    sg_lat_cols = [c for c in sg_df.columns if c.startswith("sg_")]
    if not sg_lat_cols:
        print("[subgoal] no sg_* columns in subgoal CSV — skipping")
        return

    max_offset = replan_interval + 4
    records = []

    for traj_id, g in df.groupby("trajectory_id"):
        ep_id = g["episode_id"].iloc[0]
        exp = (g[g.source == "expert"]
               .set_index("timestep")[lat_cols])
        sg_g = sg_df[sg_df.trajectory_id == traj_id]
        for _, row in sg_g.iterrows():
            t      = int(row["timestep"])
            z_sub  = row[sg_lat_cols].values.astype(float)
            dists  = {}
            for off in range(1, max_offset + 1):
                if (t + off) in exp.index:
                    d = np.linalg.norm(z_sub - exp.loc[t + off].values)
                    dists[off] = float(d)
            if dists:
                best_off = min(dists, key=dists.get)
                records.append(dict(
                    traj_id=int(traj_id), episode_id=int(ep_id),
                    t=t, replanned=bool(row.get("replanned", True)),
                    best_offset=best_off,
                    dist_at_best=dists[best_off],
                    dist_at_replan=dists.get(replan_interval, np.nan),
                    **{f"dist_off{k}": v for k, v in dists.items()},
                ))

    if not records:
        print("[subgoal] no subgoal records to analyse")
        return

    save_csv(records, out_dir / "subgoal_offsets.csv")
    rec_df = pd.DataFrame(records)

    # Per-offset mean distance
    dist_cols = [c for c in rec_df.columns if c.startswith("dist_off")]
    offsets   = [int(c.replace("dist_off", "")) for c in dist_cols]
    means     = [rec_df[c].mean() for c in dist_cols]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(offsets, means, color="steelblue", edgecolor="white")
    ax.axvline(replan_interval, color="firebrick", lw=2, ls="--",
               label=f"replan_interval={replan_interval}")
    ax.set_xlabel("Temporal offset from subgoal timestep")
    ax.set_ylabel("Mean L2 distance to expert latent")
    ax.set_title("Subgoal temporal alignment\n"
                 "(minimum bar = model's effective lookahead)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir / "subgoal_offsets.png")

    best_off_counts = rec_df["best_offset"].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.bar(best_off_counts.index, best_off_counts.values, color="purple", edgecolor="white")
    ax2.axvline(replan_interval, color="firebrick", lw=2, ls="--",
                label=f"replan_interval={replan_interval}")
    ax2.set_xlabel("Best-matching temporal offset")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of best-matching offset across all subgoals")
    ax2.legend()
    fig2.tight_layout()
    savefig(fig2, out_dir / "subgoal_best_offset_hist.png")

    print(f"[subgoal] mean best offset = {rec_df['best_offset'].mean():.2f}  "
          f"(expected {replan_interval})")


# ── Extension 4: Fréchet distance ─────────────────────────────────────────────

def _frechet(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two multivariate Gaussians fit on X and Y."""
    mu_x, mu_y = X.mean(0), Y.mean(0)
    # Regularise to avoid singular covariance with few samples
    reg = eps * np.eye(X.shape[1])
    C_x = np.cov(X, rowvar=False) + reg
    C_y = np.cov(Y, rowvar=False) + reg
    diff     = mu_x - mu_y
    covmean  = sqrtm(C_x @ C_y).real
    return float(diff @ diff + np.trace(C_x + C_y - 2 * covmean))


def analyse_frechet(
    df: pd.DataFrame,
    lat_cols: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Per-episode and global Fréchet distance between expert and model latent clouds.
    Large FD = model occupies a different region of latent space = distribution shift.
    """
    records = []
    for traj_id, g in df.groupby("trajectory_id"):
        E = g[g.source == "expert"][lat_cols].values
        M = g[g.source == "model"][lat_cols].values
        if len(E) < 4 or len(M) < 4:
            continue
        try:
            fd = _frechet(E, M)
        except Exception as exc:
            warnings.warn(f"traj {traj_id}: Fréchet failed: {exc}")
            fd = np.nan
        records.append(dict(traj_id=int(traj_id), frechet=float(fd),
                            n_expert=len(E), n_model=len(M)))

    if not records:
        print("[frechet] not enough data")
        return pd.DataFrame()

    fd_df = pd.DataFrame(records)
    save_csv(records, out_dir / "frechet.csv")

    # Global FD
    all_E = df[df.source == "expert"][lat_cols].values
    all_M = df[df.source == "model"][lat_cols].values
    try:
        global_fd = _frechet(all_E, all_M)
        print(f"[frechet] global FD = {global_fd:.4f}")
    except Exception as exc:
        global_fd = np.nan
        warnings.warn(f"Global Fréchet failed: {exc}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(fd_df.traj_id, fd_df.frechet, color="darkorange", edgecolor="white", width=0.8)
    ax.axhline(global_fd, color="firebrick", lw=2, ls="--",
               label=f"Global FD = {global_fd:.2f}")
    ax.set_xlabel("Trajectory ID")
    ax.set_ylabel("Fréchet distance")
    ax.set_title("Per-episode Fréchet distance (expert vs model latent cloud)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_dir / "frechet.png")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(fd_df.frechet.dropna(), bins=20, color="darkorange", edgecolor="white")
    ax2.axvline(global_fd, color="firebrick", lw=2, ls="--", label=f"Global FD={global_fd:.2f}")
    ax2.set_xlabel("Fréchet distance")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of per-episode Fréchet distances")
    ax2.legend()
    fig2.tight_layout()
    savefig(fig2, out_dir / "frechet_hist.png")

    return fd_df


# ── Extension 5: DTW distance ─────────────────────────────────────────────────

def _dtw_distance(E: np.ndarray, M: np.ndarray) -> float:
    """DTW distance between two multidimensional sequences, with fallback."""
    try:
        from dtaidistance import dtw_ndim
        return float(dtw_ndim.distance(E.astype(np.double), M.astype(np.double)))
    except ImportError:
        pass
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        dist, _ = fastdtw(E, M, dist=euclidean)
        return float(dist)
    except ImportError:
        raise ImportError(
            "DTW requires dtaidistance or fastdtw. "
            "Install with: pip install dtaidistance"
        )


def analyse_dtw(
    df: pd.DataFrame,
    lat_cols: list[str],
    out_dir: Path,
    pca_dim: int = 20,
) -> pd.DataFrame:
    """
    DTW distance between expert and model latent trajectories.
    Compared to aligned L2 (Extension 1): if DTW ≪ L2, the model is temporally offset
    but on the correct manifold — actionable via goal re-indexing.

    pca_dim: dimensionality reduction before DTW (192-dim DTW is very slow).
    """
    from sklearn.decomposition import PCA

    # Fit PCA on all latents
    all_Z = df[lat_cols].values
    pca   = PCA(n_components=pca_dim, random_state=42)
    pca.fit(all_Z)
    var   = pca.explained_variance_ratio_.sum()
    print(f"[dtw] PCA {pca_dim}-dim explains {var*100:.1f}% variance")

    records = []
    for traj_id, g in df.groupby("trajectory_id"):
        E_full = g[g.source == "expert"].sort_values("timestep")[lat_cols].values
        M_full = g[g.source == "model"].sort_values("timestep")[lat_cols].values
        if len(E_full) < 2 or len(M_full) < 2:
            continue
        E = pca.transform(E_full)
        M = pca.transform(M_full)
        try:
            dtw_d = _dtw_distance(E, M)
        except ImportError as exc:
            print(f"[dtw] {exc}")
            return pd.DataFrame()
        except Exception as exc:
            warnings.warn(f"traj {traj_id} DTW failed: {exc}")
            dtw_d = np.nan
        # Aligned L2: truncate to shorter trajectory
        n = min(len(E), len(M))
        aligned_l2 = float(np.linalg.norm(E[:n] - M[:n]))
        records.append(dict(traj_id=int(traj_id), dtw=dtw_d,
                            aligned_l2=aligned_l2, n_expert=len(E), n_model=len(M)))

    if not records:
        return pd.DataFrame()

    dtw_df = pd.DataFrame(records)
    save_csv(records, out_dir / "dtw.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(dtw_df.dtw, dtw_df.aligned_l2, alpha=0.7, color="purple")
    lim = max(dtw_df.dtw.max(), dtw_df.aligned_l2.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="DTW = aligned L2")
    axes[0].set_xlabel("DTW distance")
    axes[0].set_ylabel("Aligned L2 distance")
    axes[0].set_title("DTW vs aligned L2\n(below diagonal = temporal offset, not manifold shift)")
    axes[0].legend(fontsize=8)

    ratio = dtw_df.dtw / dtw_df.aligned_l2.replace(0, np.nan)
    axes[1].hist(ratio.dropna(), bins=20, color="purple", edgecolor="white")
    axes[1].axvline(1.0, color="firebrick", lw=2, ls="--", label="ratio=1")
    axes[1].set_xlabel("DTW / aligned L2")
    axes[1].set_ylabel("Count")
    axes[1].set_title("DTW/L2 ratio per episode\n(<1 = temporal offset dominates)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, out_dir / "dtw.png")

    print(f"[dtw] mean DTW={dtw_df.dtw.mean():.3f}  "
          f"mean aligned_L2={dtw_df.aligned_l2.mean():.3f}  "
          f"ratio={ratio.mean():.3f}")
    return dtw_df


# ── Summary plot ───────────────────────────────────────────────────────────────

def summary_plot(
    drift_df: pd.DataFrame,
    vel_df: pd.DataFrame,
    fd_df: pd.DataFrame,
    dtw_df: pd.DataFrame,
    replan_interval: int,
    out_dir: Path,
) -> None:
    """Four-panel summary: one metric per extension."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Drift (cosine)
    if not drift_df.empty:
        agg = drift_df.groupby("t")["cosine"].mean()
        axes[0, 0].plot(agg.index, agg.values, color="steelblue", lw=2)
        for k in range(0, int(agg.index.max()) + 1, replan_interval):
            axes[0, 0].axvline(k, color="gray", lw=0.7, ls="--", alpha=0.5)
        axes[0, 0].set_title("Ext 1: Expert–Model cosine similarity per step")
        axes[0, 0].set_xlabel("Timestep"); axes[0, 0].set_ylabel("Cosine sim")

    # Velocity
    if not vel_df.empty:
        vagg = vel_df.groupby(["source", "t"])["velocity"].mean().reset_index()
        for src, color in [("expert", "steelblue"), ("model", "firebrick")]:
            sub = vagg[vagg.source == src]
            axes[0, 1].plot(sub.t, sub.velocity, color=color, lw=2, label=src)
        for k in range(0, int(vagg.t.max()) + 1, replan_interval):
            axes[0, 1].axvline(k, color="gray", lw=0.7, ls="--", alpha=0.5)
        axes[0, 1].set_title("Ext 2: Latent velocity per step")
        axes[0, 1].set_xlabel("Timestep"); axes[0, 1].set_ylabel("||Δz||")
        axes[0, 1].legend(fontsize=8)

    # Fréchet
    if not fd_df.empty:
        axes[1, 0].hist(fd_df.frechet.dropna(), bins=15, color="darkorange", edgecolor="white")
        axes[1, 0].set_title("Ext 4: Fréchet distance distribution")
        axes[1, 0].set_xlabel("FD"); axes[1, 0].set_ylabel("Count")

    # DTW vs L2
    if not dtw_df.empty:
        axes[1, 1].scatter(dtw_df.dtw, dtw_df.aligned_l2, alpha=0.7, color="purple", s=30)
        lim = max(dtw_df.dtw.max(), dtw_df.aligned_l2.max()) * 1.05
        axes[1, 1].plot([0, lim], [0, lim], "k--", lw=1)
        axes[1, 1].set_title("Ext 5: DTW vs aligned L2")
        axes[1, 1].set_xlabel("DTW"); axes[1, 1].set_ylabel("Aligned L2")

    fig.suptitle("Latent evaluation summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir / "summary.png")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline latent-space analysis of latent_trajectories.csv"
    )
    p.add_argument("--csv",
                   default="latent_eval/results/latent_trajectories.csv",
                   help="Path to latent_trajectories.csv")
    p.add_argument("--subgoal-csv",
                   default=None,
                   help="Path to subgoal_records.csv (generated by latent_eval.py "
                        "--record-subgoals). Auto-detected if omitted.")
    p.add_argument("--output-dir",
                   default="latent_eval/results/analysis",
                   help="Directory for output CSVs and plots")
    p.add_argument("--replan-interval", type=int, default=5,
                   help="High-level replan interval in env steps (default 5)")
    p.add_argument("--dtw-pca-dim", type=int, default=20,
                   help="PCA dim before DTW (0 = skip DTW)")
    p.add_argument("--skip-dtw",    action="store_true",
                   help="Skip DTW (saves time if dtaidistance not installed)")
    p.add_argument("--skip-frechet", action="store_true")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    csv_path = Path(args.csv)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, lat_cols = load_latents(csv_path)

    # Subgoal CSV auto-detection
    subgoal_csv = Path(args.subgoal_csv) if args.subgoal_csv else \
                  csv_path.parent / "subgoal_records.csv"

    print("\n=== Extension 1+2: Drift + Velocity ===")
    drift_df = analyse_drift(df, lat_cols, args.replan_interval, out_dir)
    vel_df   = analyse_velocity(df, lat_cols, args.replan_interval, out_dir)

    print("\n=== Extension 3: Subgoal alignment ===")
    analyse_subgoals(df, lat_cols, subgoal_csv, args.replan_interval, out_dir)

    fd_df  = pd.DataFrame()
    dtw_df = pd.DataFrame()

    if not args.skip_frechet:
        print("\n=== Extension 4: Fréchet distance ===")
        fd_df = analyse_frechet(df, lat_cols, out_dir)

    if not args.skip_dtw and args.dtw_pca_dim > 0:
        print("\n=== Extension 5: DTW distance ===")
        dtw_df = analyse_dtw(df, lat_cols, out_dir, pca_dim=args.dtw_pca_dim)

    print("\n=== Summary plot ===")
    summary_plot(drift_df, vel_df, fd_df, dtw_df, args.replan_interval, out_dir)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
