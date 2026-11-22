# AIAA3102 Final Project — General Algorithm SuperAgent

An autonomous agent that, given a previously unseen classification task (data +
objective), carries out the full algorithm development process: data profiling,
solution planning, code generation in a sandboxed loop, validation-driven
iteration, and final model packaging — with honest token/cost accounting
(DeepSeek API, official standard prices, RMB 20 per-task budget).

## Architecture

```
superagent/
  config.py      # allowed models + official prices, budget & timeout limits
  ledger.py      # per-call token/cost ledger -> logs/llm_calls.jsonl, budget fuse
  llm.py         # DeepSeek client: retries, robust JSON parsing, call logging
  profiler.py    # automatic task-dir profiling (npy/csv/parquet/json/md)
  knowledge.py   # knowledge base: static recipes + cross-task experience store
  sandbox.py     # subprocess sandbox: python -I solution.py, timeout, tail-capture
  prompts.py     # stage prompts + the solution.py contract (VAL_METRIC / FINAL)
  agent.py       # orchestrator: plan -> draft -> improve (greedy) -> finalize
  main.py        # CLI
  report.py      # aggregate run summaries into a markdown table
kb/
  static_recipes.md   # task-agnostic best practices injected into prompts
  experience.json     # experience records written after each run
run_agent.py     # unified entry point
```

## Environment setup

Requires Python 3.10+ (developed on 3.14). Install dependencies:

```bash
pip install -r requirements.txt
```

Configure API access in `.env` (project root):

```
DEEPSEEK_API_KEY=sk-...
```

## Running a task

```bash
python run_agent.py \
  --task-dir "Released_Task1-In_Vehicle_Coupon_Recommendation/in_vehicle_coupon_recommendation" \
  --task-name task1_coupon \
  --metric accuracy \
  --max-iters 5 \
  --budget-rmb 20
```

Each run writes to `runs/<task_name>/run_<timestamp>/`:

- `workspace/` — agent-generated code (solution.py, best_solution.py) and artifacts
- `deliverables/` — **final_solution.py, model.joblib, metrics.json,
  val_predictions.npy, test_predictions.npy, run_summary.json**
- `logs/` — data_profile.md, plan.json, events.jsonl, **llm_calls.jsonl**
  (every model call with input/output tokens and cost in RMB), run_summary.json

Aggregate all runs:

```bash
python -m superagent.report
```

## Design notes (why this is general)

- The agent never receives task identities or hand-made per-task solutions;
  it works only from the auto-generated data profile and task-folder docs.
- The `solution.py` contract (print `VAL_METRIC`, save pipeline + predictions,
  `FINAL=1` mode retrains and predicts test once) lets the harness evaluate any
  generated solution mechanically, on any task.
- Solution search is greedy (AIDE-style): keep the best validation score;
  test data is touched exactly once, at finalize, only for reporting.
- The knowledge base carries static best practices plus distilled experience
  from every completed run, retrieved by modality/scale for future tasks.

## Constraint compliance

- Models: DeepSeek (`deepseek-chat`), officially released before 2026-10-21,
  priced at official standard rates (¥2/M input, ¥3/M output — see
  `superagent/config.py::MODEL_PRICES_RMB_PER_MTOK`).
- Budget: hard fuse at 90% of the per-task budget (default RMB 20); all calls
  logged to `llm_calls.jsonl` for audit.
- Test-set discipline: enforced by contract wording + FINAL-only mode.

## AI tool usage statement

Development was assisted by AI coding tools (system design suggestions, code
authoring and debugging). All agent runs reported in the experiment logs were
produced by the submitted SuperAgent autonomously; no manual code changes or
solution selection during runs.
## Structured search and final evaluation

The structured runner accepts the released tabular NPY layout and CSV
classification tasks. It benchmarks registered models, scores validation
predictions independently, and stores each candidate in its own directory.
Search reads train and validation only. Final evaluation refits the frozen
configuration on train + validation, then scores test predictions once.

- `--backend offline` uses a deterministic search policy with zero API calls.
- `--backend campus --key-file <path>` asks the school model to choose among the
  same registered candidates using the dataset profile and observed results.
- `--strategy single` starts from one candidate. `--strategy multi` starts
  from up to three families and considers the best two families for refinement.
  Both obey the same `--max-trials` cap.
- `--compare-and-evaluate` runs both strategies with equal budget ceilings,
  selects the winner using validation accuracy, and evaluates that winner on
  test. It uses the single-start result to break a tie. It does not finalize
  after an incomplete comparison.
- `--evaluate-run <run-directory>` performs only the final evaluation for a
  previously frozen run. Candidate predictions, model, and development hashes
  are checked before refitting.
- `--backend legacy` retains the earlier generated-code runner. Use an explicit
  structured backend to use the development/test separation described here.

Offline, the core dependencies are NumPy, pandas, scikit-learn, SciPy, joblib,
threadpoolctl and requests. LightGBM is optional for structured search. To
run the released headline task, for example:

```powershell
python -X utf8 -B run_agent.py --backend offline --compare-and-evaluate --max-trials 5 --budget-rmb 1 --task-name headlines --task-dir '.\Released_Task2-Headlines_CTR\Released_Task2-Headlines_CTR'
```

In an environment that can reach the school gateway, use `--backend campus`
and add `--key-file <path>`. The recorded price basis is the same illustrative
rate as the prior campus trial; confirm the course-date official rate before
using cost figures for submission. The structured catalog covers tabular, single-text, A/B text and a CPU image
baseline. Pretrained vision training and cross-task case retrieval are planned in [SUPERAGENT_IMPROVEMENTS.md](SUPERAGENT_IMPROVEMENTS.md).

### Metrics, feature search, OOF, and recovery

- Structured runs support `accuracy`, `macro_f1`, `weighted_f1`, `roc_auc`
  (macro one-vs-rest for multiclass), and `log_loss` (lower is better).
  Each scored candidate writes validation probabilities in task class order.
- `--feature-search` lets the campus policy propose up to four tabular
  feature operations: frequency, missingness, interaction, and numeric
  difference. These are validated and fitted on the appropriate training
  split. They do not execute model-supplied Python. Arbitrary Python proposal
  execution in a filesystem-isolated worker is separate work.
- `--oof-ensemble` evaluates one additional candidate after the regular trial
  cap. Its two members are from different model families. Weights are chosen
  using stratified training OOF probabilities, and the resulting blend is
  scored against the individual candidates using validation.
- `--resume-run <run-directory> --key-file <path>` resumes a search created by
  the current runner. Completed candidates and charged API calls are reused.
  The runner checks development-file hashes and dependency versions. It
  rejects resuming after final evaluation has started.
- `--finalize` evaluates the frozen result after a structured run, including
  the verified fallback after a partial API run. The status still reports the
  API interruption. When test labels are absent, finalization saves
  predictions and probabilities and leaves the test score empty.
- `--campus-direct` ignores the proxy environment for the campus API on
  networks where direct connections are authorized and supported. The
  normal campus route retains the system proxy.
- The CPU image baseline accepts NPY image tensors and class-subfolder image
  splits. It resizes images and fits logistic regression. It needs Pillow.
  The paired-text `pair_style/swap` variant adds style differences, reversed
  training pairs and bidirectional probability averaging.

## Experiment archive

See the [experiment archive](EXPERIMENT_ARCHIVE.md) for the run index and downloadable datasets, models, predictions, and logs.
