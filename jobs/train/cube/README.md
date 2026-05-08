# OGBench-Cube Train Jobs

This directory holds OGBench-Cube training and evaluation jobs following the same scratch-node pattern as the Reacher training flow.

## Main script

- `train_hope1.sh`: hierarchical OGBench-Cube P2 training with a frozen pretrained low-level LEWM loaded from shared scratch and copied to node-local `TMPDIR` before launch.
- `eval_hope1.sh`: hierarchical OGBench-Cube evaluation for a trained hope1 run via `hi_eval.py --config-name=hi_cube`, using the `rome` CPU partition.
- Hydra data override used by the job: `data=hi_cube`

## Defaults

- Dataset: `ogbench/cube_single_expert.h5`
- Pretrained checkpoint: `cube/lewm_object.ckpt`
- Shared data root auto-detection order:
  - `SCRATCH_STABLEWM_HOME` if set
  - `/scratch-shared/$USER/stablewm_data`
  - `/scratch_shared/$USER/stablewm_data`

## Submit

```bash
cd jobs/train/cube
sbatch train_hope1.sh
```

Useful overrides:

```bash
MAX_EPOCHS=10 sbatch train_hope1.sh
TRAIN_RUN_NAME=hi_lewm_cube_custom sbatch train_hope1.sh
SCRATCH_STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch train_hope1.sh
PRETRAINED_LEWM_CKPT=/scratch-shared/$USER/stablewm_data/cube/lewm_object.ckpt sbatch train_hope1.sh
```

## Evaluate

```bash
cd jobs/train/cube
sbatch eval_hope1.sh
```

Useful overrides:

```bash
RUN_NAME=hi_lewm_cube_train_hope1_<jobid> sbatch eval_hope1.sh
CHECKPOINT_EPOCH=10 sbatch eval_hope1.sh
GOAL_OFFSET_STEPS=25 EVAL_BUDGET=50 sbatch eval_hope1.sh
STABLEWM_HOME=/scratch-shared/$USER/stablewm_data sbatch eval_hope1.sh
```
