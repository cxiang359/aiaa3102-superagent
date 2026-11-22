"""DeepSeek LLM client with retries, JSON parsing and full call logging."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .config import (
    API_TIMEOUT_S,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MAX_OUTPUT_TOKENS,
)
from .ledger import CostLedger


class LLMClient:
    def __init__(self, model: str, ledger: CostLedger) -> None:
        from openai import OpenAI

        if not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is missing. Put it into the project .env file."
            )
        self.model = model
        self.ledger = ledger
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=API_TIMEOUT_S,
        )

    # -- raw chat -----------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        stage: str = "misc",
        temperature: float = 1.0,
        max_retries: int = 3,
    ) -> str:
        """Send a chat completion; retry on transient errors; log tokens/cost."""
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            self.ledger.check_budget()  # may raise BudgetExceeded
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                usage = resp.usage
                in_tok = getattr(usage, "prompt_tokens", 0) or 0
                out_tok = getattr(usage, "completion_tokens", 0) or 0
                cost = self.ledger.record(stage, self.model, in_tok, out_tok)
                content = resp.choices[0].message.content or ""
                print(
                    f"    [llm:{stage}] in={in_tok} out={out_tok} "
                    f"cost=¥{cost:.4f} cum=¥{self.ledger.spent_rmb:.4f}",
                    flush=True,
                )
                return content
            except Exception as exc:  # noqa: BLE001 - retry any transient error
                last_err = exc
                wait = min(2 ** attempt * 5, 60)
                print(f"    [llm:{stage}] attempt {attempt} failed: {exc}; retry in {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")

    # -- structured output --------------------------------------------------
    def chat_json(
        self,
        messages: list[dict[str, str]],
        stage: str = "misc",
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Ask for a JSON object and parse it robustly (fences / trailing text)."""
        text = self.chat(messages, stage=stage, temperature=temperature)
        parsed = parse_json_object(text)
        if parsed is not None:
            return parsed
        # Fallback: model returned fenced code instead of JSON.
        code = extract_python_code(text)
        if code:
            return {"thought": "(unparsed answer, fenced code extracted)", "code": code}
        raise ValueError(f"Could not parse JSON from model answer:\n{text[:1500]}")


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object found in ``text`` (tolerates code fences)."""
    candidates: list[str] = []
    for match in _JSON_FENCE_RE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text.strip())
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Last resort: scan for the outermost {...} via a brace matcher.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : idx + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
    return None


def extract_python_code(text: str) -> str | None:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
