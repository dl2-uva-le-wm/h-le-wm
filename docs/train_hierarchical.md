# Train Hierarchical Models

This guide explains how to train the canonical default hierarchical model from a pretrained low-level LeWM checkpoint.

The canonical training entrypoint is:

```bash
python -m h_le_wm.train.hierarchical
```

The reader-facing canonical train specs are:

```bash
python -m h_le_wm.experiments.run --spec train/pusht/hierarchical_default
python -m h_le_wm.experiments.run --spec train/cube/hierarchical_default
```

For the exact 15-epoch paper-shaped runs, there are also thin wrappers:

```bash
bash scripts/train_pusht_hierarchical_default.sh
bash scripts/train_cube_hierarchical_default.sh
```

Run the commands in this guide from the repository root.

This is a developer and research workflow. The default paper reproduction path consumes staged hierarchical checkpoints instead of retraining them, while `paper/from_scratch` uses the canonical train specs documented here.

## Prerequisites

1. Install the environment from [Install](install.md).
2. Export `STABLEWM_HOME` and stage the dataset you want to train on.
3. Stage a pretrained low-level checkpoint for that dataset.

PushT setup:

```bash
conda activate lewm-gpu
source scripts/setup_datasets.sh --datasets pusht
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/pusht/lewm=/absolute/path/to/pusht_lewm_object.ckpt
```

Cube setup:

```bash
conda activate lewm-gpu
source scripts/setup_datasets.sh --datasets cube
bash scripts/setup_baseline_checkpoints.sh \
  --checkpoint baseline/cube/lewm=/absolute/path/to/cube_lewm_object.ckpt
```

Optional validation before training:

```bash
python -m h_le_wm.validate env
python -m h_le_wm.validate datasets
python -m h_le_wm.validate checkpoints
```

## Base Config

Inspect the available Hydra surface with:

```bash
python -m h_le_wm.train.hierarchical --help
```

The default hierarchical training config lives in [h_le_wm/config/train/hi_lewm.yaml](../h_le_wm/config/train/hi_lewm.yaml).

Important defaults:

- training starts from a pretrained low-level checkpoint
- low-level modules are frozen by default
- training saves object checkpoints and full Lightning weights checkpoints every epoch
- the default dataset preset is PushT

Dataset presets:

- PushT: [h_le_wm/config/train/data/hi_pusht.yaml](../h_le_wm/config/train/data/hi_pusht.yaml)
- Cube: [h_le_wm/config/train/data/hi_ogb.yaml](../h_le_wm/config/train/data/hi_ogb.yaml)

## Canonical PushT Training

This command mirrors the repo's canonical default PushT run shape and writes outputs to the stable `runs/` location expected elsewhere in the repo:

```bash
bash scripts/train_pusht_hierarchical_default.sh
```

Equivalent package-first command:

```bash
python -m h_le_wm.experiments.run --spec train/pusht/hierarchical_default
```

After a successful run, the canonical epoch-15 object checkpoint is:

```text
${STABLEWM_HOME}/runs/pusht_hierarchical_default/pusht_hierarchical_default_epoch_15_object.ckpt
```

That path matches the supported-first-class checkpoint name `hierarchical/pusht/default_epoch15`.

## Canonical Cube Training

Cube training uses the `hi_ogb` dataset preset and the Cube baseline checkpoint:

```bash
bash scripts/train_cube_hierarchical_default.sh
```

After a successful run, the canonical epoch-15 object checkpoint is:

```text
${STABLEWM_HOME}/runs/cube_hierarchical_default/cube_hierarchical_default_epoch_15_object.ckpt
```

That path matches the supported-first-class checkpoint name `hierarchical/cube/default_epoch15`.

## Outputs

Training writes into the directory named by `subdir`.

Typical artifacts include:

- `config.yaml`
- `<output_model_name>_object.ckpt`
- `<output_model_name>_epoch_<N>_object.ckpt`
- `<output_model_name>_weights.ckpt`
- `<output_model_name>_epoch_<N>_weights.ckpt`

If you keep `subdir` relative, it is resolved under `STABLEWM_HOME`.

## Common Overrides

- Change training length: `trainer.max_epochs=<N>`
- Change latent action size: `wm.high_level.latent_action_dim=<D>`
- Disable Weights & Biases locally: `wandb.enabled=False`
- Point at a different low-level checkpoint explicitly:
  `pretrained_low_level.checkpoint.path=/absolute/path/to/object.ckpt`
- Change the run root:
  `subdir=runs/<custom_name>`
- Change the saved model stem:
  `output_model_name=<custom_name>`

The training entrypoint also supports unfrozen or joint variants through config overrides, but the commands above document the canonical frozen-low-level default setup used by the public checkpoint surface.
