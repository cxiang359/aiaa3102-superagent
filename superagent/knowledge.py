"""Knowledge base module (v0.2).

Two layers, following the Agent-Workflow-Memory idea of reusing accumulated
agent experience across tasks:

1. Static recipes (``kb/static_recipes.md``): hand-written, task-agnostic
   best practices for the common data modalities (tabular / text / image),
   injected into planning prompts.
2. Experience store (``kb/experience.json``): one record per completed run,
   written automatically at finalize time. Records are retrieved for future
   tasks by modality + scale similarity, so the agent starts each new task
   from what worked (and what failed) before.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .profiler import scan_task_dir

KB_DIR = Path(__file__).resolve().parent.parent / "kb"
STATIC_RECIPES_PATH = KB_DIR / "static_recipes.md"
EXPERIENCE_PATH = KB_DIR / "experience.json"


# ---------------------------------------------------------------------------
# modality inference (used for retrieval signatures)
# ---------------------------------------------------------------------------
def infer_modality(task_dir: str | Path) -> str:
    """Rough modality tag: 'text' | 'tabular' | 'image' | 'unknown'."""
    profile = scan_task_dir(task_dir)
    has_image_files = any(
        p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".npy"}
        and _looks_image_like(p)
        for p in Path(task_dir).rglob("*")
        if p.is_file() and not any(part.startswith(".") or part == "__MACOSX" for part in p.parts)
    )
    if has_image_files:
        return "image"

    text_cols = 0
    tabular_cols = 0
    for f in profile["files"]:
        for col in f.get("columns", []):
            if col.get("type") == "categorical_or_text" and col.get("avg_len", 0) > 40:
                text_cols += 1
            elif col.get("type") in {"numeric"} or "top_values" in col or "value_counts" in col:
                tabular_cols += 1
    if text_cols >= 1 and text_cols >= tabular_cols:
        return "text"
    if tabular_cols >= 1:
        return "tabular"
    return "unknown"


def _looks_image_like(path: Path) -> bool:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        return True
    if path.suffix.lower() == ".npy":
        try:
            import numpy as np

            arr = np.load(path, allow_pickle=True, mmap_mode="r")
            return arr.ndim in {3, 4} and arr.shape[-1] in {1, 3}
        except Exception:  # noqa: BLE001
            return False
    return False


def infer_scale(task_dir: str | Path) -> str:
    """Bucket the training size to make retrieval a bit more specific."""
    sizes = []
    profile = scan_task_dir(task_dir)
    for f in profile["files"]:
        if f.get("kind") in {"npy", "table"} and isinstance(f.get("shape"), list):
            sizes.append(f["shape"][0])
    if not sizes:
        return "unknown"
    n = max(sizes)
    if n <= 2000:
        return "small"
    if n <= 20000:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# static recipes
# ---------------------------------------------------------------------------
def load_static_recipes() -> str:
    if STATIC_RECIPES_PATH.exists():
        return STATIC_RECIPES_PATH.read_text(encoding="utf-8")
    return "(no static recipes found)"


# ---------------------------------------------------------------------------
# experience store
# ---------------------------------------------------------------------------
def _load_experience() -> list[dict]:
    if EXPERIENCE_PATH.exists():
        try:
            return json.loads(EXPERIENCE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_experience(record: dict) -> None:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_experience()
    records.append(record)
    EXPERIENCE_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def retrieve_experience(modality: str, scale: str, k: int = 3) -> str:
    """Return markdown of the most relevant prior experiences."""
    records = _load_experience()
    if not records:
        return "(no prior experience available - this is the first task)"

    def score(rec: dict) -> int:
        s = 0
        if rec.get("modality") == modality:
            s += 2
        if rec.get("scale") == scale:
            s += 1
        if rec.get("status") == "success":
            s += 1
        return s

    ranked = sorted(records, key=score, reverse=True)[:k]
    blocks = []
    for rec in ranked:
        blocks.append(
            f"### Prior run: {rec.get('task_signature', 'unnamed')} "
            f"[{rec.get('modality')}/{rec.get('scale')}, {rec.get('status')}, "
            f"val={rec.get('best_val_metric')}]"
            f"\n- approach: {rec.get('approach_summary', '')}"
            f"\n- lessons: {'; '.join(rec.get('lessons', []))}"
        )
    return "\n\n".join(blocks)


def build_experience_record(
    task_name: str,
    task_dir: str | Path,
    status: str,
    best_val_metric: float | None,
    approach_summary: str,
    lessons: list[str],
) -> dict:
    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_signature": task_name,
        "modality": infer_modality(task_dir),
        "scale": infer_scale(task_dir),
        "status": status,
        "best_val_metric": best_val_metric,
        "approach_summary": approach_summary,
        "lessons": lessons,
    }
