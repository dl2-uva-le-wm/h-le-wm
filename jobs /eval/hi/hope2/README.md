# Hope2 Eval Jobs

CPU PushT evaluation jobs for the `hope2` hierarchical run.

## Layout

- `hope2_pusht_eval_base.sh`: shared CPU launcher targeting `hi_lewm_p2_train_hope2_22253175`.
- `hope2_d25_best_prev_eval.sh`: `d=25`, best prior planning setup (`H=1`, low horizon `2`).
- `hope2_d25_h2_eval.sh`: `d=25`, same CEM settings as the paired `d=25` job, but `H=2`.
- `hope2_d25_searchboost_eval.sh`: `d=25`, same short-horizon setup but with stronger low-level CEM search.
- `hope2_d25_replan3_eval.sh`: `d=25`, same short-horizon setup but with more frequent high-level replanning (`k=3`).
- `hope2_d25_lowh1_eval.sh`: `d=25`, same short-horizon setup but with low-level horizon `1` and a heavier success-oriented CEM budget.
- `hope2_d25_h2_searchboost_eval.sh`: `d=25`, `H=2` plus a heavier success-oriented CEM search.
- `hope2_d50_h3_eval.sh`: `d=50`, sensible medium-horizon setup with `H=3`.
- `hope2_d50_h2_eval.sh`: `d=50`, same CEM settings as the paired `d=50` job, but `H=2`.
- `hope2_d50_paperscaled_h1_eval.sh`: `d=50`, higher-compute paper-scaled CEM budget with `H=1`.
- `hope2_d50_paperscaled_h2_eval.sh`: `d=50`, same paper-scaled CEM budget, but `H=2`.
- `hope2_d50_searchboost_h2_eval.sh`: `d=50`, `H=2` plus a heavier success-oriented CEM search.
- `hope2_d50_replan3_h2_eval.sh`: `d=50`, `H=2` plus more frequent high-level replanning (`k=3`) and a heavier success-oriented CEM budget.

## Default checkpoint behavior

These jobs default to:

- `RUN_NAME=hi_lewm_p2_train_hope2_22253175`
- `CHECKPOINT_EPOCH=latest`
- `EVAL_DEVICE=cpu`

Override `CHECKPOINT_EPOCH` if you want to lock evaluation to a specific saved object checkpoint.
