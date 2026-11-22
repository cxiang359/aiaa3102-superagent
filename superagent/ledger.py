"""Cost ledger: tracks every LLM call's token usage and cost in RMB.

The course requires accurate records of all agents' model calls, input/output
token usage and computed cost. Every call is appended to ``llm_calls.jsonl``
and a cumulative check enforces the per-task budget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import MODEL_PRICES_RMB_PER_MTOK


class BudgetExceeded(RuntimeError):
    """Raised before issuing a new LLM call when the budget is exhausted."""


class CostLedger:
    def __init__(
        self,
        budget_rmb: float,
        log_path: Path,
        safety_margin: float = 0.10,
        prices: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.budget_rmb = float(budget_rmb)
        self.safety_margin = float(safety_margin)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.prices = prices or MODEL_PRICES_RMB_PER_MTOK
        self.spent_rmb = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_calls = 0
        self._records: list[dict] = []
        if self.log_path.exists():
            with self.log_path.open(encoding="utf-8") as stored:
                for line in stored:
                    record = json.loads(line)
                    if record["call_index"] != self.n_calls + 1:
                        raise ValueError("The API ledger contains a missing or repeated call index.")
                    cost = self.compute_cost(record["model"], record["input_tokens"], record["output_tokens"])
                    if abs(cost - record["cost_rmb"]) > 0.00000051:
                        raise ValueError("The API ledger was recorded with a different price basis.")
                    self.n_calls += 1
                    self.input_tokens += record["input_tokens"]
                    self.output_tokens += record["output_tokens"]
                    self.spent_rmb += cost
                    self._records.append(record)

    # -- accounting ---------------------------------------------------------
    def _price_for(self, model: str) -> tuple[float, float]:
        if model not in self.prices:
            raise KeyError(
                f"Model '{model}' has no official price entry; add it to "
                f"MODEL_PRICES_RMB_PER_MTOK after verifying it complies with "
                f"the course model constraint."
            )
        p = self.prices[model]
        return p["input"], p["output"]

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pin, pout = self._price_for(model)
        return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout

    def check_budget(self) -> None:
        """Raise if no further call can be afforded (with a safety margin)."""
        cap = self.budget_rmb * (1.0 - self.safety_margin)
        if self.spent_rmb >= cap:
            raise BudgetExceeded(
                f"Budget reached: {self.spent_rmb:.4f} RMB spent "
                f"(cap {self.budget_rmb:.2f} RMB with margin)."
            )

    def record(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        extra: dict | None = None,
    ) -> float:
        cost = self.compute_cost(model, input_tokens, output_tokens)
        self.spent_rmb += cost
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.n_calls += 1
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "call_index": self.n_calls,
            "stage": stage,
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_rmb": round(cost, 6),
            "cumulative_cost_rmb": round(self.spent_rmb, 6),
        }
        if extra:
            rec.update(extra)
        self._records.append(rec)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return cost

    def summary(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "spent_rmb": round(self.spent_rmb, 4),
            "budget_rmb": self.budget_rmb,
        }
