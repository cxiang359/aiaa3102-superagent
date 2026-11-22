"""Campus chat policy for the structured SuperAgent search."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from .feature_program import parse_features
from .ledger import BudgetExceeded
from .llm import parse_json_object


CAMPUS_MODEL = "DeepSeek-V4.1-Flash"
CAMPUS_URL = "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions"
# Same basis as the historical trial. Confirm the course-date rate before submission.
TRIAL_PRICES = {"input": 2.0, "output": 8.0}
MAX_PLAN_TOKENS = 1200


class CampusPolicy:
    def __init__(self, ledger, key_file=None, *, direct=False):
        self.ledger = ledger
        self.session = requests.Session()
        self.session.trust_env = not direct
        self.key = (
            Path(key_file).read_text(encoding="utf-8-sig").strip()
            if key_file else os.environ.get("CAMPUS_API_KEY", "").strip()
        )
        if not self.key:
            raise ValueError("Supply --key-file or configure CAMPUS_API_KEY.")

    def choose(self, profile, options, history, count, stage):
        data = {
            "profile": profile,
            "options": options,
            "observations": history,
            "select_count": count,
        }
        content = (
            "Select promising classification experiments using training profile and validation feedback. "
            "Prefer diverse model families during initial exploration. During refinement, make one "
            "meaningful change at a time. Use only the candidate IDs in the options. "
            'Return a compact JSON object {"choices":[{"id":0,"reason":"one short sentence"}]}. '
            f"Return exactly {count} distinct choices. Input:\n{json.dumps(data, ensure_ascii=False)}"
        )
        def parse(answer):
            choices = answer.get("choices") if answer else None
            if not isinstance(choices, list) or len(choices) != count:
                raise ValueError("Wrong number of candidates.")
            ids = [item.get("id") if isinstance(item, dict) else None for item in choices]
            if not all(type(i) is int and 0 <= i < len(options) for i in ids) or len(set(ids)) != count:
                raise ValueError("Invalid candidate IDs.")
            return [(item["id"], str(item.get("reason", ""))[:250]) for item in choices]

        return self._request(content, stage, parse)

    def propose_features(self, profile, observations):
        data = json.dumps({"profile": profile, "observations": observations}, ensure_ascii=False)
        content = (
            'Propose a single trainable feature experiment for tabular classification. '
            'Use one to four operations on columns from the profile, with no code or filesystem access. '
            'Use frequency or missing with one column, interaction or difference with two columns. '
            'Return JSON {"features":[{"op":"interaction","columns":["col1","col2"]}],"reason":"short hypothesis"}. '
            f"Input:\n{data}"
        )

        def parse(answer):
            features = parse_features(answer.get("features"), profile["features"])
            if not features:
                raise ValueError("No features proposed.")
            return features, str(answer.get("reason", ""))[:250]

        return self._request(content, "feature_program", parse)

    def _request(self, content, stage, parse):
        for attempt in range(1, 4):
            if attempt > 1:
                time.sleep(2 ** (attempt - 2))
            prompt = content if attempt == 1 else content + "\nReturn only valid complete JSON."
            self.ledger.check_budget()
            # UTF-8 size plus message framing is a conservative input estimate for this request.
            reserve = self.ledger.compute_cost(CAMPUS_MODEL, len(prompt.encode("utf-8")) + 128, MAX_PLAN_TOKENS)
            if self.ledger.spent_rmb + reserve > self.ledger.budget_rmb:
                raise BudgetExceeded("Remaining allowance cannot cover the next planning request.")
            try:
                response = self.session.post(
                    CAMPUS_URL,
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={
                        "model": CAMPUS_MODEL, "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": MAX_PLAN_TOKENS, "temperature": 0.2, "thinking": {"type": "disabled"},
                    },
                    timeout=(10, 120),
                )
                response.raise_for_status()
            except requests.HTTPError as error:
                if attempt == 3 or error.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
                continue
            except requests.exceptions.ProxyError:
                raise
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 3:
                    raise
                continue
            result = response.json()
            usage, choice = result["usage"], result["choices"][0]
            finish = choice.get("finish_reason")
            self.ledger.record(
                stage, CAMPUS_MODEL, usage["prompt_tokens"], usage["completion_tokens"],
                extra={"attempt": attempt, "finish_reason": finish, "served_model": result.get("model")},
            )
            answer = parse_json_object(choice["message"].get("content") or "")
            if finish == "length" or not isinstance(answer, dict):
                continue
            try:
                return parse(answer)
            except ValueError:
                continue
        raise ValueError(f"{stage}: model did not return a valid response")
