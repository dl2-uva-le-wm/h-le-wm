# `salvo` Tasks (Environment + Benchmarking)

## Scope Ownership

- `roadmap/env_setup.md`
- `roadmap/tasks/salvo/*`

## Coexistence & Non-Regression Policy

1. Benchmark protocol must always include both base and hierarchical runs.
2. Keep output artifacts separated by model (`lewm` vs `hi_lewm`).
3. Any report table must show side-by-side comparability fields.

## Pre-Flight Checks (Validate What Is Already Done)

Run:

```bash
test -f roadmap/env_setup.md && echo "env_setup.md exists"
mkdir -p roadmap/tasks/salvo
ls -la roadmap/tasks/salvo
```

Mark when verified:

- [ ] env setup doc exists
- [ ] salvo task folder ready

## Task 1: Environment/Data Readiness Matrix

Create:

- `roadmap/tasks/salvo/environment_matrix.md`

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

- `roadmap/tasks/salvo/benchmark_protocol.md`

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

- `roadmap/tasks/salvo/results_schema.json`
- `roadmap/tasks/salvo/results_template.csv`

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

- `roadmap/tasks/salvo/final_report_table.md`

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

Post-implementation checks:

```bash
test -f roadmap/tasks/salvo/environment_matrix.md && echo "matrix ok"
test -f roadmap/tasks/salvo/benchmark_protocol.md && echo "protocol ok"
test -f roadmap/tasks/salvo/results_schema.json && echo "schema ok"
test -f roadmap/tasks/salvo/results_template.csv && echo "csv template ok"
test -f roadmap/tasks/salvo/final_report_table.md && echo "report table ok"
```

- [ ] all benchmark docs/artifacts generated
- [ ] protocol fields align with eval runs
