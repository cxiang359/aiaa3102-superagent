"""Run one validation-only coupon trial through the HKUST(GZ) API."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import requests

from superagent import agent, config, llm


MODEL = "DeepSeek-V4.1-Flash"
API_URL = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
TRAIN_VAL_FILES = ("C_train.npy", "y_train.npy", "C_val.npy", "y_val.npy", "Readme.md", "info.json")


class CampusClient:
    def __init__(self, model, ledger):
        self.model = model
        self.ledger = ledger

    def _complete(self, messages, stage, temperature, attempt):
        self.ledger.check_budget()
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": config.MAX_OUTPUT_TOKENS,
                "thinking": {"type": "disabled"},
            },
            timeout=(10, config.API_TIMEOUT_S),
        )
        response.raise_for_status()
        result = response.json()
        usage = result["usage"]
        choice = result["choices"][0]
        finish_reason = choice.get("finish_reason")
        cost = self.ledger.record(
            stage,
            self.model,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            extra={"attempt": attempt, "finish_reason": finish_reason},
        )
        print(
            f"    [llm:{stage}] attempt={attempt} finish={finish_reason} cost={cost:.4f} RMB",
            flush=True,
        )
        return choice["message"].get("content") or "", finish_reason

    def chat_json(self, messages, stage="misc", temperature=1.0):
        expects_code = stage == "draft" or stage.startswith(("improve_", "debug_"))
        compact_instruction = (
            "Keep the JSON compact. Keep thought and change_summary to one sentence each. "
            "Use a compact, self-contained Python script. Keep the proposed experiment, "
            "required outputs and validation protocol. Omit long comments and diagnostics."
            if expects_code
            else "Keep the JSON compact and include only the requested fields."
        )
        retry_instruction = (
            "The previous reply was incomplete or unusable. Return the full JSON object from "
            "the beginning and finish with the closing brace. Do not continue a partial reply."
        )
        problem = ""
        for attempt in (1, 2):
            instruction = compact_instruction if attempt == 1 else f"{compact_instruction} {retry_instruction}"
            last_message = {**messages[-1], "content": f"{messages[-1]['content']}\n\n{instruction}"}
            request = [*messages[:-1], last_message]
            content, finish_reason = self._complete(request, stage, temperature, attempt)
            if finish_reason == "length":
                problem = "output reached the token limit"
                continue
            answer = llm.parse_json_object(content)
            if answer and (not expects_code or answer.get("code")):
                return answer
            problem = "response did not contain the required JSON"
        raise ValueError(f"{stage} failed after 2 attempts: {problem}")


class ValidationTrial(agent.SuperAgent):
    def _phase_finalize(self):
        summary = {
            "status": "success" if self.best_code is not None else "failed_no_working_solution",
            "mode": "validation_only",
            "model": self.model,
            "best_val_metric": self.best_val,
            "best_iter": self.best_iter,
            "history": self.history,
            "cost": self.ledger.summary(),
            "out_of_budget": self.out_of_budget,
            "run_dir": str(self.run_dir),
        }
        (self.logs_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--max-iters", type=int, default=1)
    args = parser.parse_args()

    key = args.key_file.read_text(encoding="utf-8-sig").strip()
    if not key:
        parser.error("The API key file is empty.")

    root = config.PROJECT_ROOT
    source = root / "Released_Task1-In_Vehicle_Coupon_Recommendation" / "in_vehicle_coupon_recommendation"
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    task_dir = Path(tempfile.mkdtemp(prefix="coupon_train_val_", dir=runs))
    for name in TRAIN_VAL_FILES:
        shutil.copy2(source / name, task_dir / name)

    # Account at DeepSeek's published peak uncached rate for this trial.
    config.MODEL_PRICES_RMB_PER_MTOK[MODEL] = {"input": 2.0, "output": 8.0}
    agent.LLMClient = CampusClient
    trial = ValidationTrial(
        task_dir=str(task_dir),
        task_name="task1_campus_trial",
        model=MODEL,
        budget_rmb=1.0,
        run_root=runs,
        max_iters=args.max_iters,
    )
    trial.llm.api_key = key
    summary = trial.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
