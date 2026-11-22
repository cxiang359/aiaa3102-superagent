# SuperAgent experiment archive

This snapshot captures the AIAA3102 classification SuperAgent and its existing experiments as of 2026-11-22.

| School API task | Validation accuracy | Test accuracy | API calls |
|---|---:|---:|---:|
| Vehicle coupon acceptance | 76.70% | 76.70% | 4 |
| Headline CTR comparison | 70.53% | 69.74% | 4 |

See the [framework and results notes](SUPERAGENT_FRAMEWORK_AND_RESULTS.md) for the initial runner, offline comparisons, and image smoke test. The historical API logs use an illustrative billing rate.

The [experiment release](../../releases/tag/experiments-2026-11-22) includes the complete project ZIP with the original directory structure, both released datasets, candidate and final model files, predictions, and logs. Extract it into an empty directory. Cache files, process locks, and environment or credential files are excluded. Historical logs may preserve original machine paths.

## Run index

The table shows the recorded accuracy where the structured runner has a result. The legacy logs use their original reporting format.

| Run | Status | Validation | Test | API calls |
|---|---|---:|---:|---:|
| [runs/campus_real/coupon_campus_real/run_20261122_155944_b2a952](runs/campus_real/coupon_campus_real/run_20261122_155944_b2a952/logs/run_summary.json) | success | 76.70% | 76.70% | 4 |
| [runs/campus_real/headlines_campus_real/run_20261122_160511_cea16b](runs/campus_real/headlines_campus_real/run_20261122_160511_cea16b/logs/run_summary.json) | success | 70.53% | 69.74% | 4 |
| [runs/hybrid_upgrade/coupon_campus/run_20261122_150056_3ba81d](runs/hybrid_upgrade/coupon_campus/run_20261122_150056_3ba81d/logs/run_summary.json) | partial | 75.96% | — | 0 |
| [runs/hybrid_upgrade/coupon_oof/run_20261122_150056_898e52](runs/hybrid_upgrade/coupon_oof/run_20261122_150056_898e52/logs/run_summary.json) | success | 76.70% | — | 0 |
| [runs/hybrid_upgrade/headlines_oof/run_20261122_150056_033f94](runs/hybrid_upgrade/headlines_oof/run_20261122_150056_033f94/logs/run_summary.json) | success | 70.53% | — | 0 |
| [runs/structured_comparison/coupon_campus/run_20261122_140821_0c1b1c](runs/structured_comparison/coupon_campus/run_20261122_140821_0c1b1c/logs/run_summary.json) | failed | — | — | 0 |
| [runs/structured_comparison/coupon_multi/run_20261122_140821_6f439d](runs/structured_comparison/coupon_multi/run_20261122_140821_6f439d/logs/run_summary.json) | success | — | — | 0 |
| [runs/structured_comparison/coupon_single/run_20261122_140821_e79a8d](runs/structured_comparison/coupon_single/run_20261122_140821_e79a8d/logs/run_summary.json) | success | — | — | 0 |
| [runs/structured_comparison/headlines_multi/run_20261122_140821_8e78e8](runs/structured_comparison/headlines_multi/run_20261122_140821_8e78e8/logs/run_summary.json) | success | — | — | 0 |
| [runs/structured_comparison/headlines_single/run_20261122_140821_da4f33](runs/structured_comparison/headlines_single/run_20261122_140821_da4f33/logs/run_summary.json) | success | — | — | 0 |
| [runs/task1_campus_trial/run_20261122_121755](runs/task1_campus_trial/run_20261122_121755/logs/run_summary.json) | failed_no_working_solution | — | — | 0 |
| [runs/task1_campus_trial/run_20261122_122119](runs/task1_campus_trial/run_20261122_122119/logs/run_summary.json) | failed_no_working_solution | — | — | 0 |
| [runs/task1_campus_trial/run_20261122_123258](runs/task1_campus_trial/run_20261122_123258/logs/run_summary.json) | failed_no_working_solution | — | — | 2 |
| [runs/task1_campus_trial/run_20261122_132426](runs/task1_campus_trial/run_20261122_132426/logs/run_summary.json) | success | — | — | 5 |
