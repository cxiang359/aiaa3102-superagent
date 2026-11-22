"""Compile declarative feature operations into fitted transformations."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Feature:
    op: str
    columns: tuple[str, ...]


def parse_features(items, allowed):
    if not isinstance(items, list) or len(items) > 4:
        raise ValueError("Use at most four feature operations.")
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each feature must be an object.")
        op, columns = item.get("op"), item.get("columns")
        if not isinstance(op, str) or op not in {"frequency", "interaction", "missing", "difference"}:
            raise ValueError("Unknown feature operation.")
        if not isinstance(columns, list) or not all(isinstance(c, str) and c in allowed for c in columns):
            raise ValueError("Feature operation uses an unknown input column.")
        count = 2 if op in {"interaction", "difference"} else 1
        if len(columns) != count or len(set(columns)) != count:
            raise ValueError("Invalid feature operation arity.")
        result.append(Feature(op, tuple(columns)))
    if len(result) != len(set(result)):
        raise ValueError("Duplicate features.")
    return tuple(result)


def _name(feature):
    return "__agent_" + feature.op + "_" + repr(feature.columns)


def _strings(series):
    return series.astype("string").fillna("<missing>")


class FeatureProgram:
    def __init__(self, features):
        self.features = features

    def fit(self, frame):
        self.frequencies = {
            item: _strings(frame[item.columns[0]]).value_counts(normalize=True)
            for item in self.features if item.op == "frequency"
        }
        return self

    def transform(self, frame):
        changed = frame.copy()
        for item in self.features:
            name = _name(item)
            if name in changed.columns:
                raise ValueError("Generated feature name collides with an existing column.")
            left = frame[item.columns[0]]
            if item.op == "frequency":
                changed[name] = _strings(left).map(self.frequencies[item]).fillna(0).astype(float)
            elif item.op == "interaction":
                right = frame[item.columns[1]]
                changed[name] = _strings(left).str.cat(_strings(right), sep="\u001f")
            elif item.op == "missing":
                changed[name] = left.isna().astype(float)
            else:
                changed[name] = pd.to_numeric(left, errors="coerce") - pd.to_numeric(
                    frame[item.columns[1]], errors="coerce"
                )
        return changed
