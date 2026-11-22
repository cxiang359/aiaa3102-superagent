"""The SuperAgent orchestrator: plan -> draft -> iterate -> finalize.

Implements a greedy solution search (AIDE-style): every iteration produces a
complete solution.py, is executed in a sandbox, evaluated on validation data,
and kept only if it beats the current best. All decisions, code, results and
costs are recorded in the run's log directory.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import traceback
from pathlib import Path

from .config import (
    BUDGET_RMB_PER_TASK,
    DEFAULT_MAX_ITERS,
    DEFAULT_MODEL,
    BUDGET_SAFETY_MARGIN,
    SCRIPT_TIMEOUT_S,
)
from .knowledge import (
    build_experience_record,
    infer_modality,
    infer_scale,
    load_static_recipes,
    retrieve_experience,
    save_experience,
)
from .ledger import BudgetExceeded, CostLedger
from .llm import LLMClient
from .profiler import format_profile, scan_task_dir
from .prompts import debug_messages, draft_messages, improve_messages, plan_messages
from .sandbox import run_script, syntax_check

VAL_METRIC_RE = re.compile(r"VAL_METRIC:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
TEST_METRIC_RE = re.compile(r"TEST_METRIC:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


class SuperAgent:
    def __init__(
        self,
        task_dir: str,
        task_name: str,
        metric: str = "accuracy",
        model: str = DEFAULT_MODEL,
        budget_rmb: float = BUDGET_RMB_PER_TASK,
        run_root: str | Path = "runs",
        max_iters: int = DEFAULT_MAX_ITERS,
        timeout_s: int = SCRIPT_TIMEOUT_S,
    ) -> None:
        self.task_dir = Path(task_dir).resolve()
        self.task_name = task_name
        self.metric = metric
        self.model = model
        self.budget_rmb = budget_rmb
        self.max_iters = max_iters
        self.timeout_s = timeout_s

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(run_root) / task_name / f"run_{stamp}"
        self.workspace = self.run_dir / "workspace"
        self.deliverables = self.run_dir / "deliverables"
        self.logs_dir = self.run_dir / "logs"
        for d in (self.workspace, self.deliverables, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.ledger = CostLedger(
            budget_rmb=budget_rmb,
            log_path=self.logs_dir / "llm_calls.jsonl",
            safety_margin=BUDGET_SAFETY_MARGIN,
        )
        self.llm = LLMClient(model=model, ledger=self.ledger)

        # experiment state
        self.history: list[dict] = []
        self.best_val: float | None = None
        self.best_code: str | None = None
        self.best_iter: int | None = None
        self.out_of_budget = False

    # -- logging ------------------------------------------------------------
    def _event(self, kind: str, **data) -> None:
        rec = {"ts": time.strftime("%H:%M:%S"), "kind": kind, **data}
        with (self.logs_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        short = {k: (str(v)[:120] if isinstance(v, str) else v) for k, v in data.items()}
        print(f"  [{kind}] {json.dumps(short, ensure_ascii=False, default=str)}", flush=True)

    # -- helpers ------------------------------------------------------------
    def _write_solution(self, code: str) -> Path:
        path = self.workspace / "solution.py"
        path.write_text(code, encoding="utf-8")
        return path

    def _run_and_evaluate(self, path: Path) -> dict:
        # fast syntax gate (~1s) to avoid wasting a full sandbox run on SyntaxError
        syn = syntax_check(path)
        if not syn["ok"]:
            return {
                "run": {
                    "ok": False,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "[syntax error]\n" + syn["stderr"],
                    "timed_out": False,
                },
                "val": None,
            }
        result = run_script(path, timeout_s=self.timeout_s)
        val = None
        if result["ok"]:
            match = VAL_METRIC_RE.search(result["stdout"])
            if match:
                val = float(match.group(1))
            else:
                result["ok"] = False
                result["stderr"] = (
                    f"[contract violation] VAL_METRIC line not found.\n" + result["stderr"]
                )
        return {"run": result, "val": val}

    def _attempt_with_debug_loop(self, builder, first_code: str, stage_label: str) -> dict | None:
        """Run a piece of generated code; on failure, debug up to N times."""
        code = first_code
        for debug_round in range(1 + self.max_debug_attempts()):
            path = self._write_solution(code)
            outcome = self._run_and_evaluate(path)
            if outcome["val"] is not None:
                return {"code": code, "path": path, **outcome}
            self._event(
                "script_failed",
                stage=stage_label,
                debug_round=debug_round,
                stderr=outcome["run"]["stderr"][-600:],
            )
            if debug_round >= self.max_debug_attempts():
                return None
            try:
                dbg = self.llm.chat_json(
                    debug_messages(
                        self.profile_md,
                        self.task_name,
                        self.metric,
                        code,
                        outcome["run"]["stderr"],
                        outcome["run"]["stdout"],
                        str(self.task_dir),
                        str(self.workspace),
                        self.timeout_s,
                        timed_out=outcome["run"]["timed_out"],
                    ),
                    stage=f"debug_{stage_label}",
                )
                code = dbg.get("code") or ""
                if not code:
                    return None
            except BudgetExceeded:
                self.out_of_budget = True
                return None
        return None

    def max_debug_attempts(self) -> int:
        from .config import MAX_DEBUG_ATTEMPTS

        return MAX_DEBUG_ATTEMPTS

    def _keep_best(self, code: str, val: float, iteration: int) -> bool:
        if self.best_val is None or val > self.best_val:
            self.best_val = val
            self.best_code = code
            self.best_iter = iteration
            (self.workspace / "best_solution.py").write_text(code, encoding="utf-8")
            return True
        return False

    # -- main loop ----------------------------------------------------------
    def run(self) -> dict:
        print(f"=== SuperAgent run on '{self.task_name}' ===", flush=True)
        print(f"    task_dir: {self.task_dir}", flush=True)
        print(f"    run_dir:  {self.run_dir}", flush=True)
        print(f"    model:    {self.model}  budget: ¥{self.budget_rmb}", flush=True)
        try:
            self._phase_profile_and_plan()
            self._phase_draft()
            self._phase_improve()
        except BudgetExceeded:
            self.out_of_budget = True
            self._event("budget_exceeded", note="stopping iteration early")
        except Exception as exc:  # noqa: BLE001
            self._event("fatal_error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])
        finally:
            try:
                summary = self._phase_finalize()
            except Exception as exc:  # noqa: BLE001
                summary = {"status": "finalize_failed", "error": str(exc)}
                self._event("finalize_failed", error=str(exc), trace=traceback.format_exc()[-2000:])
        print(f"=== Run finished: spent ¥{self.ledger.spent_rmb:.4f} ===", flush=True)
        return summary

    def _phase_profile_and_plan(self) -> None:
        self._event("stage", name="profile_and_plan")
        profile = scan_task_dir(self.task_dir)
        self.profile_md = format_profile(profile)
        (self.logs_dir / "data_profile.md").write_text(self.profile_md, encoding="utf-8")
        self._event("profile_done", chars=len(self.profile_md))

        # knowledge base: static recipes + retrieved prior experience
        self.modality = infer_modality(self.task_dir)
        self.scale = infer_scale(self.task_dir)
        self.knowledge_md = (
            f"--- Static recipes (general best practices) ---\n{load_static_recipes()}\n\n"
            f"--- Prior experience retrieved for modality='{self.modality}' scale='{self.scale}' ---\n"
            f"{retrieve_experience(self.modality, self.scale)}"
        )
        (self.logs_dir / "knowledge_used.md").write_text(self.knowledge_md, encoding="utf-8")
        self._event("knowledge", modality=self.modality, scale=self.scale)

        plan = self.llm.chat_json(
            plan_messages(self.profile_md, self.task_name, self.metric, self.knowledge_md),
            stage="plan",
        )
        self.plan = plan
        (self.logs_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._event("plan_done", analysis=plan.get("analysis", "")[:400], plan=plan.get("plan", "")[:400])

    def _phase_draft(self) -> None:
        self._event("stage", name="draft")
        draft = self.llm.chat_json(
            draft_messages(
                self.profile_md,
                self.task_name,
                self.metric,
                self.plan,
                str(self.task_dir),
                str(self.workspace),
                self.timeout_s,
                knowledge_md=self.knowledge_md,
            ),
            stage="draft",
        )
        code = draft.get("code") or ""
        if not code:
            raise RuntimeError("Draft returned no code")
        result = self._attempt_with_debug_loop(lambda: None, code, "draft")
        if result is None:
            self._event("draft_failed")
            self.history.append({"iter": 0, "change_summary": draft.get("change_summary", "initial draft"), "val": None, "outcome": "FAILED"})
            return
        improved = self._keep_best(result["code"], result["val"], 0)
        self.history.append(
            {
                "iter": 0,
                "change_summary": draft.get("change_summary", "initial draft"),
                "val": result["val"],
                "outcome": "new best" if improved else "kept (first success)",
                "code_chars": len(result["code"]),
            }
        )
        self._event("draft_ok", val=result["val"], best=self.best_val)

    def _phase_improve(self) -> None:
        if self.best_code is None:
            self._event("stage", name="improve_skipped_no_draft")
            return
        self._event("stage", name="improve", max_iters=self.max_iters)
        no_gain_streak = 0
        EARLY_STOP_PATIENCE = 2  # stop after N consecutive non-improving iterations
        for iteration in range(1, self.max_iters + 1):
            if self.out_of_budget:
                break
            if no_gain_streak >= EARLY_STOP_PATIENCE:
                self._event(
                    "early_stop",
                    note=f"no improvement for {no_gain_streak} consecutive iterations",
                )
                break
            self._event("iteration", i=iteration)
            remaining_iters = self.max_iters - iteration + 1
            remaining_budget = max(
                0.0,
                self.budget_rmb * (1.0 - BUDGET_SAFETY_MARGIN) - self.ledger.spent_rmb,
            )
            imp = self.llm.chat_json(
                improve_messages(
                    self.profile_md,
                    self.task_name,
                    self.metric,
                    self.history,
                    self.best_code,
                    self.best_val,
                    str(self.task_dir),
                    str(self.workspace),
                    self.timeout_s,
                    knowledge_md=self.knowledge_md,
                    remaining_iters=remaining_iters,
                    remaining_budget_rmb=remaining_budget,
                ),
                stage=f"improve_{iteration}",
            )
            code = imp.get("code") or ""
            if not code:
                self.history.append({"iter": iteration, "change_summary": imp.get("change_summary", ""), "val": None, "outcome": "no code returned"})
                continue
            result = self._attempt_with_debug_loop(lambda: None, code, f"improve_{iteration}")
            if result is None:
                self.history.append({"iter": iteration, "change_summary": imp.get("change_summary", ""), "val": None, "outcome": "FAILED after debug attempts"})
                no_gain_streak += 1
                self._event("iteration_failed", i=iteration)
                continue
            improved = self._keep_best(result["code"], result["val"], iteration)
            self.history.append(
                {
                    "iter": iteration,
                    "change_summary": imp.get("change_summary", ""),
                    "val": result["val"],
                    "outcome": "new best" if improved else ("worse - rejected" if result["val"] <= self.best_val else "kept"),
                    "code_chars": len(result["code"]),
                }
            )
            no_gain_streak = 0 if improved else no_gain_streak + 1
            self._event("iteration_done", i=iteration, val=result["val"], best=self.best_val, improved=improved)

    def _phase_finalize(self) -> dict:
        self._event("stage", name="finalize")
        summary: dict = {
            "task_name": self.task_name,
            "metric": self.metric,
            "model": self.model,
            "best_val_metric": self.best_val,
            "best_iter": self.best_iter,
            "out_of_budget": self.out_of_budget,
            "history": self.history,
            "cost": self.ledger.summary(),
            "task_dir": str(self.task_dir),
            "run_dir": str(self.run_dir),
        }

        if self.best_code is None:
            summary["status"] = "failed_no_working_solution"
            self._event("finalize_no_solution")
            (self.logs_dir / "run_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            return summary

        # copy best solution to deliverables and run it in FINAL mode
        final_path = self.deliverables / "final_solution.py"
        final_path.write_text(self.best_code, encoding="utf-8")
        result = run_script(final_path, timeout_s=self.timeout_s, env_extra={"FINAL": "1"})
        test_metric = None
        if not result["ok"]:
            # one debug retry for FINAL mode
            self._event("final_run_failed", stderr=result["stderr"][-600:])
            try:
                dbg = self.llm.chat_json(
                    debug_messages(
                        self.profile_md,
                        self.task_name,
                        self.metric,
                        self.best_code,
                        result["stderr"],
                        result["stdout"],
                        str(self.task_dir),
                        str(self.workspace),
                        self.timeout_s,
                        timed_out=result["timed_out"],
                    ),
                    stage="debug_final",
                )
                if dbg.get("code"):
                    self.best_code = dbg["code"]
                    final_path.write_text(self.best_code, encoding="utf-8")
                    (self.workspace / "best_solution.py").write_text(self.best_code, encoding="utf-8")
                    result = run_script(final_path, timeout_s=self.timeout_s, env_extra={"FINAL": "1"})
            except BudgetExceeded:
                self.out_of_budget = True

        if result["ok"]:
            match = TEST_METRIC_RE.search(result["stdout"])
            test_metric = float(match.group(1)) if match else None

        # collect artifacts written into workspace during the FINAL run
        artifact_names = ["model.joblib", "metrics.json", "val_predictions.npy", "test_predictions.npy"]
        copied = []
        for name in artifact_names:
            src = self.workspace / name
            if src.exists():
                shutil.copy2(src, self.deliverables / name)
                copied.append(name)
        if result["ok"] and "test_predictions.npy" not in copied:
            self._event("warning", note="FINAL run ok but test_predictions.npy missing")

        summary.update(
            {
                "status": "success" if result["ok"] else "final_run_failed",
                "test_metric": test_metric,
                "deliverables": copied,
                "final_stdout_tail": result["stdout"][-800:],
            }
        )
        summary["cost"] = self.ledger.summary()
        (self.logs_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (self.deliverables / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        self._event("finalize_done", status=summary["status"], test_metric=test_metric)
        self._write_experience(summary)
        return summary

    def _write_experience(self, summary: dict) -> None:
        """Distill this run into the knowledge base for future tasks (v0.2)."""
        approach = ""
        lessons: list[str] = []
        try:
            lessons_call = self.llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": "You are the experience-distillation module of a general "
                        "ML agent. Summarize what worked on this task in reusable, task-agnostic "
                        "lessons. Be concise.",
                    },
                    {
                        "role": "user",
                        "content": f"""Task type: {self.modality}/{self.scale}, metric={self.metric}.
Plan: {json.dumps(self.plan, ensure_ascii=False)[:1200]}
Experiment history: {json.dumps(self.history, ensure_ascii=False)[:2000]}
Outcome: status={summary.get('status')}, best_val={summary.get('best_val_metric')}, test={summary.get('test_metric')}

Return ONLY JSON: {{"approach_summary": "1-2 sentences on the winning approach",
"lessons": ["reusable lesson 1", "lesson 2", "lesson 3"]}}""",
                    },
                ],
                stage="kb_distill",
            )
            approach = lessons_call.get("approach_summary", "")
            lessons = [str(x) for x in lessons_call.get("lessons", [])][:5]
        except (BudgetExceeded, Exception) as exc:  # noqa: BLE001 - KB write must not crash the run
            self._event("kb_distill_failed", error=str(exc)[:200])
            best_rec = next(
                (h for h in reversed(self.history) if h.get("outcome") == "new best"),
                self.history[-1] if self.history else {},
            )
            approach = best_rec.get("change_summary", "")

        record = build_experience_record(
            task_name=self.task_name,
            task_dir=self.task_dir,
            status=summary.get("status", "unknown"),
            best_val_metric=summary.get("best_val_metric"),
            approach_summary=approach,
            lessons=lessons,
        )
        try:
            save_experience(record)
            self._event("kb_written", modality=record["modality"], scale=record["scale"])
        except Exception as exc:  # noqa: BLE001
            self._event("kb_write_failed", error=str(exc)[:200])
