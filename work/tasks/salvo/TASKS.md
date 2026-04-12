# `salvo` Tasks (Environment + Benchmarking)

## Scope Ownership

- `work/env_setup.md`
- `work/tasks/salvo/*`

## Task 1: Environment/Data Readiness Matrix

Create:

- `work/tasks/salvo/environment_matrix.md`

Use this exact template:

```markdown
# Environment Readiness Matrix

| env | dataset_name | source | frameskip | train_ready | eval_ready | blockers | owner |
|-----|--------------|--------|-----------|-------------|------------|----------|-------|
| PushT | pusht_expert_train | local/HF | 5 | Y/N | Y/N | ... | ... |
| TwoRoom | ... | ... | ... | ... | ... | ... | ... |
| Manipulator | ... | ... | ... | ... | ... | ... | ... |
| Humanoid | ... | ... | ... | ... | ... | ... | ... |
```

Done criteria:

- all four environments listed with no empty blocker/owner fields.

## Task 2: Benchmark Protocol File

Create:

- `work/tasks/salvo/benchmark_protocol.md`

Include this fixed protocol snippet:

```markdown
## Fixed Evaluation Protocol

- Seeds: [42, 3072, 7777]
- num_eval: 50
- goal_offset_steps: 25
- eval_budget: 50
- CEM samples/iters: keep identical across flat and hi runs
- Hardware fields to log:
  - GPU model
  - CUDA version
  - wall clock eval time
  - average time per episode
```

Done criteria:

- protocol used by all benchmark runs in PR notes.

## Task 3: Result Schema + Aggregator Template

Create:

- `work/tasks/salvo/results_schema.json`
- `work/tasks/salvo/results_template.csv`

JSON schema snippet:

```json
{
  "run_id": "string",
  "model": "lewm|hi_lewm",
  "config_path": "string",
  "seed": 0,
  "env": "string",
  "num_eval": 0,
  "success_rate": 0.0,
  "eval_time_sec": 0.0,
  "goal_offset_steps": 0,
  "eval_budget": 0,
  "notes": "string"
}
```

CSV header snippet:

```text
run_id,model,config_path,seed,env,num_eval,success_rate,eval_time_sec,goal_offset_steps,eval_budget,notes
```

## Task 4: Summary Table for Report

Create:

- `work/tasks/salvo/final_report_table.md`

Template snippet:

```markdown
| env | model | mean_success_rate | std_success_rate | mean_eval_time_sec | comments |
|-----|-------|-------------------|------------------|--------------------|----------|
| PushT | LeWM | ... | ... | ... | ... |
| PushT | Hi-LeWM | ... | ... | ... | ... |
```

## Handoff Artifact

- One final package in this folder with:
  - matrix
  - protocol
  - schema
  - summary table template.
