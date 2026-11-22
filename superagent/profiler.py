"""Automatic task/data profiling (no LLM needed).

Scans a task directory and produces a compact structured summary of every
data file: shapes, dtypes, per-column statistics, class balance for label-like
arrays and file previews for documents. The formatted report is injected into
the agent's planning prompt, so the agent can reason about any new task
without task-specific hard-coding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_EXTS = {".npy", ".csv", ".parquet", ".pq", ".pkl", ".pickle", ".json"}
DOC_EXTS = {".md", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {"__MACOSX", "__pycache__", ".git", ".ipynb_checkpoints"}
LABEL_HINTS = ("y_", "label", "target", "class")

MAX_COLS_REPORTED = 60
MAX_TOP_VALUES = 8
MAX_TEXT_SAMPLE_LEN = 100
MAX_DOC_CHARS = 2500
MAX_FILE_BYTES_DETAILED = 300 * 1024 * 1024  # 300MB


def scan_task_dir(task_dir: str | Path) -> dict:
    """Return a structured profile of every relevant file in ``task_dir``."""
    task_dir = Path(task_dir)
    profile: dict = {
        "task_dir": str(task_dir),
        "files": [],
        "docs": {},
        "notes": [],
    }
    if not task_dir.exists():
        profile["notes"].append(f"ERROR: task dir does not exist: {task_dir}")
        return profile

    files = [
        p
        for p in sorted(task_dir.rglob("*"))
        if p.is_file()
        and not any(part in SKIP_DIRS for part in p.parts)
        and not p.name.startswith(".")
        and p.suffix.lower() not in {".zip", ".gz", ".ipynb", ".png", ".jpg", ".jpeg"}
    ]

    for path in files:
        entry: dict = {"name": str(path.relative_to(task_dir)), "size_bytes": path.stat().st_size}
        suffix = path.suffix.lower()
        try:
            if suffix == ".npy":
                entry.update(_profile_npy(path))
            elif suffix in {".csv", ".tsv"}:
                sep = "\t" if suffix == ".tsv" else ","
                entry.update(_profile_table(pd.read_csv(path, sep=sep)))
            elif suffix in {".parquet", ".pq"}:
                entry.update(_profile_table(pd.read_parquet(path)))
            elif suffix == ".json":
                entry.update(_profile_json(path))
            elif suffix in {".md", ".txt", ".yaml", ".yml"}:
                entry["kind"] = "doc"
                profile["docs"][entry["name"]] = _read_doc(path)
            else:
                entry["kind"] = "other"
        except Exception as exc:  # noqa: BLE001 - profiling must never crash
            entry["kind"] = "unreadable"
            entry["error"] = f"{type(exc).__name__}: {exc}"
        profile["files"].append(entry)

    profile["notes"].extend(_infer_split_structure(profile["files"]))
    return profile


# ---------------------------------------------------------------------------
# .npy handling
# ---------------------------------------------------------------------------
def _profile_npy(path: Path) -> dict:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES_DETAILED:
        return {"kind": "npy", "note": f"file too large for detailed profiling ({size} bytes)"}
    arr = np.load(path, allow_pickle=True)
    entry: dict = {"kind": "npy", "shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.ndim == 1:
        entry["summary"] = _summarize_1d(arr)
    elif arr.ndim == 2:
        entry["columns"] = _summarize_columns(arr)
    else:
        entry["note"] = f"ndim={arr.ndim}; mean={_safe_mean(arr)}; std={_safe_std(arr)}"
    return entry


def _summarize_1d(arr: np.ndarray) -> dict:
    flat = np.asarray(arr).ravel()
    info: dict = {"n": int(flat.shape[0])}
    try:
        values = flat.astype(float)
        valid = values[~np.isnan(values)] if values.dtype.kind == "f" else values
        info["type"] = "numeric"
        info["min"] = float(np.min(valid))
        info["max"] = float(np.max(valid))
        info["mean"] = float(np.mean(valid))
        uniq = np.unique(valid)
        if len(uniq) <= 20:
            info["value_counts"] = {str(v): int((valid == v).sum()) for v in uniq}
            info["note"] = "looks like discrete labels/classes"
    except (ValueError, TypeError):
        # object array (strings etc.)
        s = pd.Series(flat.astype(object))
        info["type"] = "object"
        info["n_unique"] = int(s.nunique())
        vc = s.value_counts().head(MAX_TOP_VALUES)
        info["top_values"] = {str(k): int(v) for k, v in vc.items()}
    return info


def _summarize_columns(arr: np.ndarray) -> list[dict]:
    cols: list[dict] = []
    n_cols = arr.shape[1]
    for j in range(min(n_cols, MAX_COLS_REPORTED)):
        col = arr[:, j]
        info = _summarize_1d(col)
        info["index"] = j
        cols.append(info)
    if n_cols > MAX_COLS_REPORTED:
        cols.append({"note": f"…{n_cols - MAX_COLS_REPORTED} more columns not shown"})
    return cols


# ---------------------------------------------------------------------------
# table handling (csv/parquet)
# ---------------------------------------------------------------------------
def _profile_table(df: pd.DataFrame) -> dict:
    entry: dict = {
        "kind": "table",
        "shape": list(df.shape),
        "columns": [],
    }
    for name in df.columns[:MAX_COLS_REPORTED]:
        col = df[name]
        info: dict = {"name": str(name)}
        info["missing_pct"] = round(float(col.isna().mean()) * 100, 2)
        if pd.api.types.is_numeric_dtype(col):
            info["type"] = "numeric"
            desc = col.describe()
            info["min"] = float(desc.get("min", 0.0))
            info["max"] = float(desc.get("max", 0.0))
            info["mean"] = round(float(desc.get("mean", 0.0)), 4)
            nuniq = int(col.nunique())
            info["n_unique"] = nuniq
            if nuniq <= 10:
                vc = col.value_counts().head(MAX_TOP_VALUES)
                info["value_counts"] = {str(k): int(v) for k, v in vc.items()}
        else:
            s = col.dropna().astype(str)
            info["type"] = "categorical_or_text"
            avg_len = float(s.str.len().mean()) if len(s) else 0.0
            info["avg_len"] = round(avg_len, 1)
            info["n_unique"] = int(s.nunique())
            if avg_len <= 40:
                vc = s.value_counts().head(MAX_TOP_VALUES)
                info["top_values"] = {str(k)[:MAX_TEXT_SAMPLE_LEN]: int(v) for k, v in vc.items()}
            else:
                info["samples"] = [t[:MAX_TEXT_SAMPLE_LEN] for t in s.head(3)]
        entry["columns"].append(info)
    if len(df.columns) > MAX_COLS_REPORTED:
        entry["columns"].append({"note": f"…{len(df.columns) - MAX_COLS_REPORTED} more columns"})
    return entry


def _profile_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and len(data) <= 40:
        return {"kind": "json", "content": data}
    return {"kind": "json", "preview": json.dumps(data, ensure_ascii=False)[:MAX_DOC_CHARS]}


def _read_doc(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_DOC_CHARS:
        return text[:MAX_DOC_CHARS] + "\n…[truncated]"
    return text


# ---------------------------------------------------------------------------
# structure inference
# ---------------------------------------------------------------------------
def _infer_split_structure(files: list[dict]) -> list[str]:
    """Notes about train/val/test split structure found by filename patterns."""
    notes: list[str] = []
    stems = [Path(f["name"]).stem.lower() for f in files]
    has_npy = any(f.get("kind") == "npy" for f in files)
    if has_npy:
        features = sorted({s.replace("c_", "").replace("x_", "") for s in stems if s.startswith(("c_", "x_"))})
        labels = sorted({s.replace("y_", "") for s in stems if s.startswith("y_")})
        if features and labels:
            notes.append(
                "Detected .npy split files: feature files "
                f"{features} and label files {labels} (splits: train/val/test). "
                "Feature arrays load with np.load(path, allow_pickle=True)."
            )
    csvs = [s for s in stems if s.endswith(".csv") or s in ("train", "val", "test")]
    csvs = [f["name"] for f in files if f.get("kind") == "table"]
    if csvs:
        notes.append(f"Detected table files: {csvs} (likely train/val/test splits).")
    return notes


# ---------------------------------------------------------------------------
# markdown formatting
# ---------------------------------------------------------------------------
def format_profile(profile: dict) -> str:
    lines: list[str] = [f"TASK DIR: {profile['task_dir']}", ""]

    if profile["docs"]:
        lines.append("## Documentation files in the task folder")
        for name, text in profile["docs"].items():
            lines.append(f"### {name}")
            lines.append(text.strip())
            lines.append("")

    lines.append("## Data files")
    for f in profile["files"]:
        lines.append(f"### {f['name']} ({f.get('size_bytes', 0)} bytes, kind={f.get('kind')})")
        if f.get("kind") == "npy":
            lines.append(f"- shape: {f.get('shape')}  dtype: {f.get('dtype')}")
            if "summary" in f:
                lines.append(f"- 1-D summary: {json.dumps(f['summary'], ensure_ascii=False)}")
            for col in f.get("columns", []):
                if "note" in col and "index" not in col:
                    lines.append(f"- {col['note']}")
                    continue
                lines.append(f"- col[{col.get('index', '?')}]: {json.dumps(col, ensure_ascii=False)}")
            for extra in f.get("note", []) if isinstance(f.get("note"), list) else []:
                lines.append(f"- {extra}")
            if isinstance(f.get("note"), str):
                lines.append(f"- {f['note']}")
        elif f.get("kind") == "table":
            lines.append(f"- shape: {f.get('shape')}")
            for col in f.get("columns", []):
                lines.append(f"- {json.dumps(col, ensure_ascii=False)}")
        elif f.get("kind") == "json":
            lines.append(f"- {json.dumps(f.get('content', f.get('preview', '')), ensure_ascii=False)}")
        elif f.get("kind") == "doc":
            lines.append("- (document, shown above)")
        else:
            lines.append(f"- {json.dumps({k: v for k, v in f.items() if k not in ('name', 'size_bytes')}, ensure_ascii=False)}")
        lines.append("")

    if profile["notes"]:
        lines.append("## Structure notes")
        for note in profile["notes"]:
            lines.append(f"- {note}")
    return "\n".join(lines)


if __name__ == "__main__":  # manual debugging helper
    import sys

    print(format_profile(scan_task_dir(sys.argv[1])))
