"""Global configuration: API access, model pricing, budget and runtime limits.

All monetary figures follow the course rule: only Qwen / DeepSeek models
officially released on or before 2026-10-21 with official standard API
pricing are allowed, budget RMB 20 per evaluation task.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser (no external dependency)."""
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


_ENV = _load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or _ENV.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ---------------------------------------------------------------------------
# Allowed models (course rule: officially released <= 2026-10-21) and their
# OFFICIAL STANDARD prices in RMB per 1M tokens, used for honest cost
# accounting without cache/promotion discounts. Prices are configurable here
# so they can be re-verified against the provider's price page before the
# final submission.
# ---------------------------------------------------------------------------
MODEL_PRICES_RMB_PER_MTOK: dict[str, dict[str, float]] = {
    "deepseek-chat": {"input": 2.0, "output": 3.0},
    "deepseek-reasoner": {"input": 4.0, "output": 16.0},
}

DEFAULT_MODEL = "deepseek-chat"

# Budget accounting (per evaluation task).
BUDGET_RMB_PER_TASK = 20.0
BUDGET_SAFETY_MARGIN = 0.10  # stop issuing new LLM calls at 90% of budget

# Sandbox / loop limits.
SCRIPT_TIMEOUT_S = 600
MAX_DEBUG_ATTEMPTS = 3       # consecutive debug retries per failed script run
DEFAULT_MAX_ITERS = 5        # improvement iterations after the first draft

# LLM call limits.
MAX_OUTPUT_TOKENS = 8192
API_TIMEOUT_S = 300
