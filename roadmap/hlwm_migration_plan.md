# H-LeWM Migration Plan (Modules / Training / Inference)

This migration is intentionally breaking-first:

- no compatibility layer
- no backward checkpoint support
- remove old hierarchy logic entirely

Goal architecture:

- 2 levels only (`P1` low-level, `P2` high-level)
- shared latent space
- no inverse dynamics
- variable-waypoint training
- top-down planning philosophy from HLWM

Core implementation rule: reuse original LEWM modules from `third_party/lewm` wherever possible.

## Modules

- Remove old hierarchical stack from local code:
  - delete ID-based components and usage (`id2`, `id3`, `k1`, `k2`, `num_levels`, midpoint-anchor logic)
  - delete old losses tied to ID branches
- Rebuild hierarchy with original LEWM primitives:
  - use the same predictor module class used by original LEWM for both `P1` and `P2` (imported from third-party LEWM module path)
  - use the original encoder and action embedding interfaces from third-party LEWM code
- Predictor contract:
  - `P1` stays the standard low-level predictor
  - `P2` is another instance of the same predictor class (not a custom predictor class)
  - `P2` is conditioned on latent actions mapped into the action input space expected by the LEWM action path
- Latent action path:
  - keep latent action encoder `A_psi` (transformer + `[CLS]`) to encode action chunks between waypoints
  - set default `latent_action_dim = wm.embed_dim`
  - because `P2` expects action inputs in real-action dimensionality, add projection layer:
    - if `latent_action_dim == real_action_dim`: identity
    - else: learned linear projection `latent_action_dim -> real_action_dim`
- Breaking policy:
  - no legacy branches, no migration shims, no old checkpoint loading support
  - new architecture becomes the only supported hierarchical format

## Training

- Full-target training design (long-term):
  - two losses plus SIGReg:
    - low-level loss on `P1`
    - high-level waypoint loss on `P2`
    - shared latent regularization
  - variable-waypoint sampling (non-fixed stride), default `N=3`
  - joint end-to-end optimization across shared encoder + both predictors + latent action encoder
- Data/waypoint pipeline:
  - sample ordered waypoint indices `t1 < ... < tN` per sequence
  - build action chunks `a[t_k:t_{k+1}]`
  - encode chunks with `A_psi`
  - project latent action to real-action input dim for `P2` action path
  - predict `z_hat[t_{k+1}]` from `z[t_k]` and projected action input
- Loss structure:
  - `L_total = alpha * L_low + beta * L_high + lambda * L_SIGReg`
  - defaults: `alpha=1.0`, `beta=1.0`
- Pretrained low-level bootstrapping requirement (current priority):
  - load pretrained LEWM encoder + low-level predictor from LEWM checkpoint
  - train only high-level components initially (`P2`, `A_psi`, latent-to-action projection if used)
  - freeze encoder and `P1` in the initial phase
- Training outputs/logging for current phase:
  - `l2_pred_loss`, optional `sigreg_loss` depending on freeze strategy, total `loss`
  - waypoint gap stats and latent-action norm stats

## Inference

- Final-target inference design (planned, not immediate):
  - high-level CEM in latent action space
  - low-level CEM in primitive action space
  - top-down subgoal extraction (`z_sub`) from best high-level rollout
- Current migration phase:
  - inference refactor is explicitly deferred
  - no production inference/planner rewrite in this phase
  - focus is only training correctness of `P2` over pretrained low-level stack

## Hydra Configuration Contract (Add / Remove / Configurable)

Add new config groups/keys:

- `pretrained_low_level`
  - `enabled` (bool)
  - `source_policy` (string run/policy name)
  - `checkpoint.path` (explicit path override)
  - `checkpoint.selection_mode` (`latest`, `best`, `epoch`, `explicit_path`)
  - `checkpoint.epoch` (int, used when mode=`epoch`)
  - `freeze.encoder` (bool, default true for current phase)
  - `freeze.low_level_predictor` (bool, default true for current phase)
  - `freeze.low_level_action_encoder` (bool, default true for current phase)
- `wm.high_level`
  - `enabled` (bool)
  - `latent_action_dim` (int, default `${wm.embed_dim}`)
  - `real_action_dim` (int, default derived from dataset action dim * frameskip)
  - `latent_to_action_proj` (`auto`, `identity`, `linear`)
  - `waypoints.num` (int, default 3)
  - `waypoints.min_stride` (int)
  - `waypoints.max_span` (int)
- `loss`
  - `alpha` (float)
  - `beta` (float)
  - `sigreg.weight` (float; can be 0 for strictly frozen-encoder phase)
- `checkpointing`
  - `object_dump.epoch_interval` (int)
  - `weights_dump.enabled` (bool)
  - `weights_dump.every_n_epochs` (int)

Remove old config keys/groups:

- `wm.num_levels`
- `wm.k1`
- `wm.k2`
- old inverse-dynamics loss group (`loss.inverse_dynamics.*`)
- old predictor groups that exist only for custom ID-based branches

Configurable checkpoint sampling behavior (required):

- make low-level checkpoint selection fully configurable in Hydra via `pretrained_low_level.checkpoint.selection_mode`
- allow deterministic selection by epoch
- allow latest/best auto-selection
- allow direct explicit checkpoint path override

## Current Implementation Scope (Do This First)

This first implementation pass is intentionally narrow and clean:

- use original LEWM code path/modules from third-party as the base
- instantiate `P2` as the same predictor module class as LEWM
- add latent-action encoder + latent-to-real-action projection for `P2` input
- load pretrained LEWM encoder + `P1`
- freeze low-level stack
- train only second-level predictor path (`P2` + action-latent path)
- do not implement joint training in this phase
- do not implement inference/planning refactor in this phase
- do not add compatibility support for old architecture/checkpoints
