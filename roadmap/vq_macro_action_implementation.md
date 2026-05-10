# VQ Macro-Action Implementation

This note records exactly what was added for the VQ macro-action path, how it works at train and eval time, and how to launch the new PushT run.

## Goal

The original hierarchy used a continuous transformer-based macro-action encoder:

- action chunk `(a_t, ..., a_{t+h-1})`
- transformer encoder with `[CLS]`
- linear head to one continuous latent action
- high-level CEM searched directly in that continuous latent space

The main change is:

- keep the action-chunk encoder
- insert a VQ bottleneck over the macro-action latent
- keep the high-level planner structure
- quantize planner-generated macro actions back onto the learned codebook before `P2` rollout

This keeps the low-level planner unchanged and only changes the high-level macro-action path.

## Files Added / Changed

### New file

- `hi_vq.py`

This contains:

- `VectorQuantizer`
- `VQActionEncoder`

### Updated files

- `hi_module.py`
- `hi_jepa.py`
- `hi_train.py`
- `config/train/hi_lewm.yaml`
- `tests/test_hi_planning.py`
- `tests/test_hi_train_speedups.py`
- `jobs/train/pusht/train_vq_hope1.sh`
- `jobs/train/pusht/README.md`

## New Modules

## `hi_vq.py`

### `VectorQuantizer`

Purpose:

- stores a learnable codebook
- maps a continuous latent to its nearest code
- returns a straight-through quantized latent for backprop

Outputs:

- `quantized`: nearest codebook vector
- `quantized_st`: straight-through version used by the forward path
- `indices`: chosen code ids
- `codebook_loss`
- `commitment_loss`
- `perplexity`
- `active_codes`

### `VQActionEncoder`

Purpose:

- encode a variable-length action chunk into one macro-action latent
- quantize that latent through the codebook
- reconstruct the original action chunk from the quantized latent

Structure:

1. `input_proj` maps action tokens to model dimension
2. `[CLS]` token + positional embeddings are added
3. transformer encoder processes the chunk
4. `output_proj` maps the `[CLS]` state to `latent_action_dim`
5. `VectorQuantizer` snaps that latent to a codebook vector
6. decoder reconstructs the action chunk from the quantized latent

So the current VQ path is:

`action chunk -> transformer encoder -> latent -> nearest code -> decoder reconstruction`

## Compatibility Layer

## `hi_module.py`

The existing `LatentActionEncoder` was not removed. Instead, it was given two small compatibility methods:

- `encode_with_info(...)`
- `quantize_latents(...)`

For the old continuous encoder:

- `encode_with_info` just returns `{"macro_actions": ...}`
- `quantize_latents` is identity

This lets the rest of the stack call one interface regardless of whether the backend is continuous or VQ.

## Model Integration

## `hi_jepa.py`

Three things were added.

### 1. `encode_macro_actions_with_info(...)`

This is the new macro-action API used by training.

Behavior:

- if the backend exposes `encode_with_info`, use it
- otherwise fall back to the old `encode_macro_actions(...)`

This means training can now receive:

- `macro_actions`
- optional VQ losses
- optional VQ usage statistics

without changing the external hierarchy structure.

### 2. `quantize_macro_actions_for_planning(...)`

This is the planner-facing helper.

Behavior:

- if the backend exposes `quantize_latents`, use it
- otherwise return the input unchanged

### 3. High-level rollout quantization

Inside `rollout_high(...)`, the autoregressive macro-action history is now quantized before being fed into `predict_high(...)`.

That is the key inference-time change.

Before:

- CEM samples arbitrary continuous macro-action vectors
- `P2` consumes them directly

Now:

- CEM still samples continuous vectors
- those vectors are snapped to the nearest codebook vectors
- `P2` rolls out only on quantized macro actions

This is not a fully discrete planner. It is still continuous CEM outside, but manifold-constrained inside the model rollout.

## Training Changes

## `hi_train.py`

Three helper pieces were added.

### 1. `build_macro_action_encoder(...)`

This instantiates the selected backend from config:

- `continuous`
- `vq`

So model construction no longer hardcodes `LatentActionEncoder`.

### 2. `encode_macro_actions_with_aux(...)`

This standardizes the output of the macro-action backend during training.

Returned dict always contains:

- `macro_actions`

For VQ it may also contain:

- `recon_loss`
- `commitment_loss`
- `codebook_loss`
- `perplexity`
- `active_codes`

### 3. `add_macro_action_aux_losses(...)`

This pulls VQ aux losses out of the encoder output and adds them to the training `output` dict.

Logged fields:

- `vq_recon_loss`
- `vq_commitment_loss`
- `vq_codebook_loss`
- `vq_perplexity`
- `vq_active_codes`
- `vq_loss`

### Forward pass changes

Both:

- `hi_lejepa_forward(...)`
- `hi_lejepa_forward_p2_frozen(...)`

now:

1. build action chunks between waypoints
2. call `encode_macro_actions_with_aux(...)`
3. use `macro_output["macro_actions"]` for the high-level predictor
4. add `vq_loss` into the final objective

The high-level prediction loss itself was not replaced.

Current total loss is:

`alpha * l1_pred_loss + beta * l2_pred_loss + lambda * sigreg_loss + vq_loss`

with:

`vq_loss = recon_weight * recon_loss + commitment_weight * commitment_loss + codebook_weight * codebook_loss`

## Config Changes

## `config/train/hi_lewm.yaml`

The train config now includes explicit VQ settings:

- `latent_action_encoder.type: vq`
- `latent_action_encoder.vq.num_codes`
- `latent_action_encoder.vq.decoder_hidden_dim`
- `loss.vq.recon_weight`
- `loss.vq.commitment_weight`
- `loss.vq.codebook_weight`

Important: `wm.high_level.latent_action_dim` is still controlled separately. The VQ codebook vectors live in that latent-action dimension.

## PushT Training Script

## `jobs/train/pusht/train_vq_hope1.sh`

This is the new scratch-node PushT launcher for the VQ setup.

What it does:

1. resolves repo root
2. requires a scratch-node allocation with `TMPDIR`
3. activates `lewm-gpu`
4. loads W&B credentials
5. copies:
   - `pusht_expert_train.h5`
   - `pusht/lewm_object.ckpt`
   into node-local scratch
6. launches `hi_train.py`
7. keeps the low-level pretrained stack frozen
8. explicitly forces VQ overrides in the Hydra command

Important explicit overrides in the script:

- `training.train_low_level=False`
- `pretrained_low_level.enabled=True`
- all low-level freeze flags set to `True`
- `latent_action_encoder.type=vq`
- `wm.high_level.latent_action_dim=16`
- `wm.high_level.macro_to_condition_proj=linear`
- `latent_action_encoder.vq.num_codes=128`
- `loss.vq.*` explicitly set

The reason those are explicit in the script is to avoid silent drift if the default config changes later.

## Current Inference Semantics

The current inference change is intentionally narrow.

### What changed

- high-level planner candidates are quantized before `P2` rollout

### What did not change

- low-level planner
- CEM outer loop
- action-space definition for the high-level solver
- latent prior calibration logic in `hi_policy.py`

So this is:

- not yet a discrete planner over code indices
- not yet a codebook-aware prior or sampling scheme

It is the minimal inference-time change needed so the VQ bottleneck affects actual high-level planning.

## Tests Added

## `tests/test_hi_planning.py`

Added a test that verifies:

- if the latent-action backend exposes `quantize_latents`
- `rollout_high(...)` uses the quantized values rather than the raw planner inputs

## `tests/test_hi_train_speedups.py`

Added a test that verifies:

- training forward can consume macro-action aux outputs
- `vq_loss` is accumulated correctly from the configured weights

## Verification Status

What was verified directly in the current environment:

- shell syntax for `jobs/train/pusht/train_vq_hope1.sh`
- Python syntax compilation for the edited Python files

What was not verified here:

- runtime execution of the updated training/eval path
- pytest execution

Reason:

- the default interpreter in this environment does not provide `torch` or `pytest`

So the next real check should be:

1. run the new job script in the intended conda environment
2. confirm VQ metrics appear in logs
3. confirm `macro_action_norm`, `vq_perplexity`, and `vq_active_codes` look sane
4. then run eval to see whether quantized high-level rollout improves stability
