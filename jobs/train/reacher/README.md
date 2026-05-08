# Reacher Train Jobs

This directory holds Reacher training and evaluation jobs following the same scratch-node pattern as the PushT training flow.

## Main script

- `train_hope1.sh`: hierarchical Reacher P2 training with a frozen pretrained low-level LEWM loaded from shared scratch and copied to node-local `TMPDIR` before launch.
- `eval_hope1.sh`: hierarchical Reacher evaluation for a trained hope1 run via `hi_eval.py --config-name=hi_reacher`, using the `rome` CPU partition.
- Hydra data override used by the job: `data=hi_reacher`

## Defaults

- Dataset: `reacher.h5`
- Pretrained checkpoint: `reacher/lewm_object.ckpt`
- Shared data root auto-detection order:
  - `SCRATCH_STABLEWM_HOME` if set
  - `/scratch-shared/$USER/stablewm_data`
  - `/scratch_shared/$USER/stablewm_data`

## Submit

```bash
cd jobs/train/reacher
sbatch train_hope1.sh
```

Useful overrides:

```bash
MAX_EPOCHS=10 sbatch train_hope1.sh
TRAIN_RUN_NAME=hi_lewm_reacher_custom sbatch train_hope1.sh
SCRATCH_STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch train_hope1.sh
PRETRAINED_LEWM_CKPT=/scratch-shared/$USER/stablewm_data/reacher/lewm_object.ckpt sbatch train_hope1.sh
```

## Evaluate

```bash
cd jobs/train/reacher
sbatch eval_hope1.sh
```

Useful overrides:

```bash
RUN_NAME=hi_lewm_reacher_train_hope1_<jobid> sbatch eval_hope1.sh
CHECKPOINT_EPOCH=10 sbatch eval_hope1.sh
GOAL_OFFSET_STEPS=25 EVAL_BUDGET=50 sbatch eval_hope1.sh
STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch eval_hope1.sh
```
