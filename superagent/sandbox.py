"""Sandboxed execution of agent-generated scripts in a subprocess.

Scripts run with the same interpreter (so the environment's packages are
available), in isolated mode (-I), with a hard timeout and captured output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .config import SCRIPT_TIMEOUT_S

STDOUT_KEEP = 6000  # characters kept from the tail of each stream


def syntax_check(script_path: Path) -> dict:
    """Fast py_compile check (~1s) to catch syntax errors before a full run."""
    script_path = Path(script_path).resolve()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-I", "-m", "py_compile", str(script_path)],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        if proc.returncode == 0:
            return {"ok": True, "stderr": ""}
        return {"ok": False, "stderr": (proc.stderr or "").strip()[-2000:]}
    except Exception as exc:
        return {"ok": True, "stderr": f"[syntax_check skipped: {exc}]"}


def run_script(
    script_path: Path,
    timeout_s: int = SCRIPT_TIMEOUT_S,
    env_extra: dict[str, str] | None = None,
) -> dict:
    """Run a python script; return {ok, returncode, stdout, stderr, timed_out}."""
    script_path = Path(script_path).resolve()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)

    cmd = [sys.executable, "-I", str(script_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            env=env,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": _tail(proc.stdout or ""),
            "stderr": _tail(proc.stderr or ""),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "ok": False,
            "returncode": None,
            "stdout": _tail(stdout),
            "stderr": _tail(stderr + f"\n[TIMEOUT after {timeout_s}s]"),
            "timed_out": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"[sandbox error] {type(exc).__name__}: {exc}",
            "timed_out": False,
        }


def _tail(text: str, keep: int = STDOUT_KEEP) -> str:
    text = text.strip()
    if len(text) <= keep:
        return text
    return "…[truncated]…\n" + text[-keep:]
