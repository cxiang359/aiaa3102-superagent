"""Released classification splits, without reading the test split during search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .image_data import flat_images, folder_samples, pixels


_TARGET_NAMES = {"class", "label", "target", "y", "label_pairwise"}
_TEXT_PAIR = re.compile(r"^(.+?)[_-]([ab])$", re.IGNORECASE)


@dataclass
class TaskData:
    root: Path
    layout: str
    modality: str
    train: pd.DataFrame
    val: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    label: str | None
    text_columns: tuple[str, ...] = ()

    @property
    def classes(self):
        return np.unique(self.y_train)

    def profile(self):
        classes, counts = np.unique(self.y_train, return_counts=True)
        return {
            "modality": self.modality,
            "features": [str(column) for column in self.train.columns],
            "training_rows": len(self.train),
            "validation_rows": len(self.val),
            "training_class_counts": dict(zip(map(str, classes), map(int, counts))),
            "missing_fraction": round(float(self.train.isna().to_numpy().mean()), 4),
            "text_columns": list(self.text_columns),
        }

    def test_features(self):
        if self.layout == "npy":
            return _npy_features(self.root / "C_test.npy", list(self.train.columns))
        if self.layout == "image_folder":
            directory = self.root / "test"
            return pixels(flat_images(directory) or [path for path, _ in folder_samples(directory)])
        return pd.read_csv(self.root / "test.csv", usecols=list(self.train.columns))[self.train.columns]

    def test_labels(self):
        if self.layout == "npy":
            path = self.root / "y_test.npy"
            return _labels(np.load(path)) if path.is_file() else None
        if self.layout == "image_folder":
            if flat_images(self.root / "test"):
                return None
            return _labels([label for _, label in folder_samples(self.root / "test")])
        path = self.root / "test.csv"
        if self.label not in pd.read_csv(path, nrows=0).columns:
            return None
        return _labels(pd.read_csv(path, usecols=[self.label])[self.label])


def _labels(values):
    labels = np.asarray(values).reshape(-1)
    if not len(labels) or pd.isna(labels).any():
        raise ValueError("Labels must be nonempty and cannot contain missing values.")
    return labels


def _npy_features(path, columns=None):
    # The released coupon task documents that features are NumPy object arrays.
    values = np.load(path, allow_pickle=True)
    if values.ndim in (3, 4) and len(values):
        return pixels(values)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError(f"{path.name}: expected a nonempty feature matrix")
    if columns is None:
        columns = [f"feature_{index}" for index in range(values.shape[1])]
    return pd.DataFrame(values, columns=columns)


def _text_layout(frame):
    text = []
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            values = frame[column].dropna().astype(str)
            if not values.empty and values.str.split().str.len().median() >= 3:
                text.append(column)
    if len(text) == 2 and len(frame.columns) == 2:
        match = [_TEXT_PAIR.match(column) for column in text]
        if all(match) and match[0][1].casefold() == match[1][1].casefold():
            return "paired_text", tuple(sorted(text, key=lambda column: _TEXT_PAIR.match(column)[2].casefold()))
    if len(text) == 1 and len(frame.columns) == 1:
        return "text", tuple(text)
    return "tabular", ()


def load_development(task_dir, label=None):
    root = Path(task_dir).resolve()
    if (root / "C_train.npy").is_file():
        if label is not None:
            raise ValueError("--label-column is only used for CSV tasks")
        train = _npy_features(root / "C_train.npy")
        val = _npy_features(root / "C_val.npy", list(train.columns))
        y_train = _labels(np.load(root / "y_train.npy"))
        y_val = _labels(np.load(root / "y_val.npy"))
        modality = "image" if all(str(c).startswith("px") for c in train.columns) else "tabular"
        task = TaskData(root, "npy", modality, train, val, y_train, y_val, None)
    elif (root / "train.csv").is_file():
        train = pd.read_csv(root / "train.csv")
        val = pd.read_csv(root / "val.csv")
        if label is None:
            possible = [c for c in train if c.casefold() in _TARGET_NAMES or c.casefold().startswith("label_")]
            if len(possible) != 1:
                raise ValueError("Specify --label-column when the CSV target is ambiguous.")
            label = possible[0]
        y_train, y_val = _labels(train.pop(label)), _labels(val.pop(label))
        if set(train.columns) != set(val.columns):
            raise ValueError("Training and validation feature columns differ.")
        val = val[train.columns]
        modality, text_columns = _text_layout(train)
        if modality == "paired_text" and len(np.unique(y_train)) != 2:
            raise ValueError("The A/B preference tool requires a binary target.")
        task = TaskData(root, "csv", modality, train, val, y_train, y_val, label, text_columns)
    elif (root / "train").is_dir() and (root / "val").is_dir():
        train_samples, val_samples = [folder_samples(root / split) for split in ("train", "val")]
        task = TaskData(root, "image_folder", "image", pixels(p for p, _ in train_samples),
                        pixels(p for p, _ in val_samples), _labels([y for _, y in train_samples]),
                        _labels([y for _, y in val_samples]), None)
    else:
        raise ValueError("Expected C_train.npy, train.csv, or train/ and val/ image directories.")

    if not task.train.shape[1] or len(task.train) != len(task.y_train) or len(task.val) != len(task.y_val):
        raise ValueError("Features and labels do not align.")
    if len(task.classes) < 2 or not np.isin(task.y_val, task.classes).all():
        raise ValueError("Invalid training classes or unseen validation class.")
    return task


def score_predictions(labels, predicted, classes):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    if labels.ndim != 1 or labels.shape != predicted.shape or not len(labels):
        raise ValueError("Expected one prediction for every scoreable example.")
    if not np.isin(predicted, classes).all() or not np.isin(labels, classes).all():
        raise ValueError("Predictions or labels contain an unknown class.")
    return float(np.mean(predicted == labels))

