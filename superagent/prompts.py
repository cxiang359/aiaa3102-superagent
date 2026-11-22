"""Prompt templates for the SuperAgent's stages.

The solution CONTRACT is the keystone: it defines exactly what the generated
``solution.py`` must do so the harness can evaluate it mechanically
(VAL_METRIC line, artifacts, FINAL mode for test prediction).
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are the decision-making core of a General Algorithm SuperAgent for \
classification tasks. Given a previously unseen dataset and objective, you autonomously \
analyze data, design solutions, write complete runnable Python code, run experiments, \
and iteratively improve based on observed validation results.

Rules you must follow:
1. Generality: use only what you can learn from the provided data profile and documentation. \
You must not assume any task-specific identity or invoke prepared task-specific solutions.
2. The designated test set must NEVER be used for training, hyperparameter tuning, or \
solution selection. Model selection happens on validation data only. Test evaluation \
happens exclusively in FINAL mode at the very end.
3. Every answer that includes code must be a JSON object with keys "thought", \
"change_summary" and "code" (the complete, self-contained Python source). No placeholders, \
no omitted sections, no "..." inside code.
4. Be budget-conscious: prefer efficient, well-established approaches; make one meaningful \
change per iteration so its effect is measurable.
5. Keep each script's runtime under {timeout_s} seconds and make it deterministic (fixed seeds).
"""

CONTRACT_TEMPLATE = """
=== CONTRACT for solution.py (must be followed exactly) ===
1. Self-contained single Python file. Available packages: numpy, pandas, scikit-learn, \
scipy, lightgbm, xgboost, catboost, matplotlib, joblib. Standard library is allowed.
2. It will be executed with its working directory being the workspace folder. Use these \
constants (already defined for you, hard-code them into the script):
   TASK_DIR = r"{task_dir}"   # read-only task data
   WORK_DIR  = r"{work_dir}"   # save artifacts here
3. The script must, in development mode (default):
   a. Load the training and validation data from TASK_DIR itself.
   b. Train one or more candidate models, compare them on validation data, and select \
by "{metric}" (higher is better).
   c. Print exactly one line: VAL_METRIC: <float>  (the best validation "{metric}").
   d. Save into WORK_DIR: model.joblib (the winning fitted pipeline INCLUDING preprocessing), \
metrics.json (dict with at least {{"val_metric": float}}), val_predictions.npy \
(predicted labels for the validation set, same order).
4. FINAL mode: when os.environ.get("FINAL") == "1", the script must retrain the winning \
configuration on the union of train (+ validation if it does not harm comparison) data, \
load the test inputs from TASK_DIR, write test_predictions.npy into WORK_DIR (predicted \
labels, same row order as the test inputs), and, only if the test ground-truth labels \
exist in TASK_DIR, additionally print: TEST_METRIC: <float>.
5. Never write any file into TASK_DIR. Never load test labels outside FINAL mode evaluation \
printing. Keep runtime under {timeout_s}s.
=== END CONTRACT ===
"""


def _contract(task_dir: str, work_dir: str, metric: str, timeout_s: int) -> str:
    return CONTRACT_TEMPLATE.format(
        task_dir=task_dir, work_dir=work_dir, metric=metric, timeout_s=timeout_s
    )


def _system(timeout_s: int) -> str:
    return SYSTEM_PROMPT.format(timeout_s=timeout_s)


def plan_messages(profile_md: str, task_name: str, metric: str, knowledge_md: str = "") -> list[dict]:
    return [
        {"role": "system", "content": _system(600)},
        {
            "role": "user",
            "content": f"""A new classification task "{task_name}" must be solved. The target \
metric is "{metric}" (higher is better). Below is an automatically generated profile of the \
task folder (data files + documentation).

{profile_md}

=== Knowledge base: static recipes and prior experience ===
{knowledge_md}

Analyze this task. Return ONLY a JSON object:
{{
  "analysis": "what kind of data/features/labels this is, key challenges, pitfalls",
  "plan": "your modeling strategy: preprocessing, model candidates, validation procedure",
  "first_idea": "concrete description of the first solution.py you will write"
}}""",
        },
    ]


def draft_messages(
    profile_md: str,
    task_name: str,
    metric: str,
    plan: dict,
    task_dir: str,
    work_dir: str,
    timeout_s: int,
    knowledge_md: str = "",
) -> list[dict]:
    return [
        {"role": "system", "content": _system(timeout_s)},
        {
            "role": "user",
            "content": f"""Task "{task_name}" data profile:

{profile_md}

Your plan: {plan.get('plan', '')}
First idea: {plan.get('first_idea', '')}

=== Knowledge base: static recipes and prior experience ===
{knowledge_md}

{_contract(task_dir, work_dir, metric, timeout_s)}

Write the first complete solution.py now. Keep it simple, correct and fast; make sure the \
data loading matches the profile above exactly. Return ONLY a JSON object with keys \
"thought", "change_summary", "code".""",
        },
    ]


def improve_messages(
    profile_md: str,
    task_name: str,
    metric: str,
    history: list[dict],
    best_code: str,
    best_val: float,
    task_dir: str,
    work_dir: str,
    timeout_s: int,
    knowledge_md: str = "",
    remaining_iters: int = 0,
    remaining_budget_rmb: float = 0.0,
) -> list[dict]:
    hist_lines = []
    for h in history:
        outcome = h.get("outcome", "")
        hist_lines.append(
            f"- iter {h['iter']}: val_{metric}={h.get('val', 'n/a')} {outcome} | {h.get('change_summary', '')[:300]}"
        )
    history_md = "\n".join(hist_lines) or "(no history yet)"
    knowledge_md = knowledge_md or "(no prior experience available)"
    return [
        {"role": "system", "content": _system(timeout_s)},
        {
            "role": "user",
            "content": f"""Task "{task_name}" data profile:

{profile_md}

=== Experiment history (chronological) ===
{history_md}

=== Current best code (val_{metric}={best_val}) ===
```python
{best_code}
```

=== Knowledge base: relevant prior experience ===
{knowledge_md}

You have {remaining_iters} iteration(s) left and about ¥{remaining_budget_rmb:.2f} of the \
LLM budget remaining; plan your remaining moves accordingly (save expensive ideas for \
when they have the best expected payoff; the LAST iteration may be spent on a safe, \
high-value consolidation such as ensembling).

{_contract(task_dir, work_dir, metric, timeout_s)}

Based on the history and results, propose the next single meaningful improvement (e.g. \
better preprocessing/encoding, stronger model, hyperparameters, ensembling, feature \
engineering, handling of text/missing values). Do not repeat failed directions. Return \
ONLY a JSON object with keys "thought", "change_summary", "code" (complete new solution.py).""",
        },
    ]


def debug_messages(
    profile_md: str,
    task_name: str,
    metric: str,
    failing_code: str,
    stderr: str,
    stdout: str,
    task_dir: str,
    work_dir: str,
    timeout_s: int,
    timed_out: bool = False,
) -> list[dict]:
    return [
        {"role": "system", "content": _system(timeout_s)},
        {
            "role": "user",
            "content": f"""Task "{task_name}". The generated solution.py FAILED to run.

{"The run TIMED OUT (over " + str(timeout_s) + "s)." if timed_out else "Error output:"}
{stderr}

Last stdout:
{stdout}

=== Failing code ===
```python
{failing_code}
```

{_contract(task_dir, work_dir, metric, timeout_s)}

Fix the code so it runs to completion and satisfies the contract. Minimize changes: keep \
the modeling approach, fix only what is broken. Return ONLY a JSON object with keys \
"thought", "change_summary", "code".""",
        },
    ]
