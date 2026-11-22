"""Reusable, allowlisted classification pipelines for the experiment search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .feature_program import Feature, FeatureProgram


@dataclass(frozen=True)
class TrialSpec:
    family: str
    variant: str = "base"
    features: tuple[Feature, ...] = ()


def _categorical_values(frame):
    # Convert mixed object arrays from the released data into stable categories.
    return frame.astype("string").fillna("<missing>").astype(object)


class TabularModel:
    def __init__(self, spec, seed):
        self.spec, self.seed = spec, seed

    def _prepare(self, frame):
        values = self.program.transform(frame)
        values[self.categorical] = _categorical_values(values[self.categorical])
        return values

    def fit(self, frame, labels):
        self.program = FeatureProgram(self.spec.features).fit(frame)
        prepared = self.program.transform(frame)
        numeric = [c for c in prepared if pd.api.types.is_numeric_dtype(prepared[c])]
        self.categorical = [c for c in prepared if c not in numeric]
        family, variant = self.spec.family, self.spec.variant
        native = family == "hgb_native"
        encoding = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if native else OneHotEncoder(handle_unknown="ignore", sparse_output=family != "hgb_onehot")
        )
        numeric_transformer = SimpleImputer(strategy="median", keep_empty_features=True)
        if family in {"tabular_lr", "image_lr"}:
            numeric_transformer = Pipeline([("impute", numeric_transformer), ("scale", StandardScaler())])
        transform = ColumnTransformer(
            [("num", numeric_transformer, numeric), ("cat", encoding, self.categorical)],
            sparse_threshold=0.0 if family == "hgb_onehot" or native else 1.0,
        )
        if family.startswith("hgb_"):
            mask = ([False] * len(numeric) + [True] * len(self.categorical)) if native else None
            if native and any(prepared[c].nunique(dropna=False) > 255 for c in self.categorical):
                raise ValueError("Native histogram boosting supports up to 255 categories per feature.")
            model = HistGradientBoostingClassifier(
                max_iter=250, max_leaf_nodes=63 if variant == "wide" else 31,
                min_samples_leaf=12 if variant == "compact_leaf" else 20,
                categorical_features=mask, random_state=self.seed,
            )
        elif family == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=250, min_samples_leaf=1 if variant == "leaf1" else 2,
                random_state=self.seed, n_jobs=2,
            )
        elif family in {"tabular_lr", "image_lr"}:
            model = LogisticRegression(C=3.0 if variant == "c3" else 1.0, max_iter=1000)
        elif family == "lightgbm":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=250, num_leaves=63 if variant == "wide" else 31,
                random_state=self.seed, n_jobs=2, verbosity=-1,
            )
        else:
            raise ValueError(f"Unknown tabular model: {family}")
        self.pipeline = Pipeline([("features", transform), ("model", model)])
        with threadpool_limits(limits=2):
            self.pipeline.fit(self._prepare(frame), labels)
        self.classes_ = self.pipeline.classes_
        return self

    def predict(self, frame):
        return self.classes_[self.predict_proba(frame).argmax(axis=1)]

    def predict_proba(self, frame):
        with threadpool_limits(limits=2):
            return self.pipeline.predict_proba(self._prepare(frame))


class TextModel:
    def __init__(self, spec, columns, seed):
        self.spec, self.columns, self.seed = spec, columns, seed

    def _strings(self, frame):
        return [frame[c].astype("string").fillna("").astype(str) for c in self.columns]

    def _features(self, frame):
        fields = self._strings(frame)
        parts = []
        for vectorizer in self.vectorizers:
            encoded = vectorizer.transform(fields[0])
            if len(fields) == 2:
                encoded = encoded - vectorizer.transform(fields[1])
            parts.append(encoded)
        if self.spec.family == "pair_style":
            def style(series):
                return np.column_stack([
                    series.str.len() / 70, series.str.split().str.len() / 12,
                    series.str.count(r"[!?]"), series.str.count(r"\d") / 3,
                    series.str.count(r"[A-Z]") / 8, series.str.count(r"[,:;]"),
                ])
            parts.append(sparse.csr_matrix(style(fields[0]) - style(fields[1])))
        return sparse.hstack(parts, format="csr")

    def fit(self, frame, labels):
        family, variant = self.spec.family, self.spec.variant
        fields = self._strings(frame)
        sample = pd.concat(fields, ignore_index=True)
        self.vectorizers = []
        if family not in {"pair_char", "text_char"}:
            self.vectorizers.append(TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60_000,
            ))
        if family not in {"pair_word", "text_word"}:
            self.vectorizers.append(TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=80_000,
            ))
        for vectorizer in self.vectorizers:
            vectorizer.fit(sample)
        pair = len(fields) == 2
        self.model = LogisticRegression(
            C={"c3": 3.0, "c03": 0.3, "swap": 0.5}.get(variant, 1.0), max_iter=1000,
            fit_intercept=not pair, random_state=self.seed,
        )
        features = self._features(frame)
        if variant == "swap":
            classes = np.unique(labels)
            if not pair or len(classes) != 2:
                raise ValueError("Swap augmentation requires a binary preference task.")
            # Reversing a pair negates every feature in this model.
            features = sparse.vstack([features, -features], format="csr")
            labels = np.concatenate([labels, np.where(labels == classes[0], classes[1], classes[0])])
        with threadpool_limits(limits=2):
            self.model.fit(features, labels)
        self.classes_ = self.model.classes_
        return self

    def predict(self, frame):
        return self.classes_[self.predict_proba(frame).argmax(axis=1)]

    def predict_proba(self, frame):
        with threadpool_limits(limits=2):
            features = self._features(frame)
            probabilities = self.model.predict_proba(features)
            if self.spec.variant == "swap":
                probabilities = (probabilities + self.model.predict_proba(-features)[:, ::-1]) / 2
            return probabilities


def _installed(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def dependencies():
    return {name: _installed(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "xgboost", "torch", "torchvision", "Pillow")}


def catalog(task):
    available = {
        "tabular": {
            "hgb_onehot": ("base", "wide", "compact_leaf"),
            "extra_trees": ("base", "leaf1"),
            "tabular_lr": ("base", "c3"),
        },
        "image": {"image_lr": ("base", "c3")},
        "paired_text": {
            "pair_word_char": ("base", "c3", "c03"), "pair_word": ("base", "c3"),
            "pair_char": ("base", "c3"), "pair_style": ("base", "swap", "c3", "c03"),
        },
        "text": {"text_word_char": ("base", "c3", "c03"), "text_word": ("base", "c3"),
                 "text_char": ("base", "c3")},
    }[task.modality].copy()
    if task.modality == "tabular":
        if _installed("lightgbm"):
            available["lightgbm"] = ("base", "wide")
        nominal = [c for c in task.train if not pd.api.types.is_numeric_dtype(task.train[c])]
        if nominal and all(task.train[c].nunique(dropna=False) <= 255 for c in nominal):
            available["hgb_native"] = ("base", "wide", "compact_leaf")
    return available


def build_classifier(task, spec, seed):
    if task.modality in {"tabular", "image"}:
        return TabularModel(spec, seed)
    return TextModel(spec, task.text_columns, seed)


def variants(spec, available):
    return [replace(spec, variant=name) for name in available[spec.family] if name != spec.variant]

