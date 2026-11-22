"""Optional out-of-fold tuning for a two-model probability blend."""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .classifiers import TrialSpec, build_classifier
from .feature_program import Feature
from .metrics import aligned_probabilities, quality, score


def restore_spec(spec):
    return TrialSpec(spec["family"], spec["variant"], tuple(
        Feature(item["op"], tuple(item["columns"])) for item in spec.get("features", ())
    ))


class ProbabilityBlend:
    def __init__(self, models, weights, classes):
        self.models = models
        self.weights = weights
        self.classes_ = classes

    def predict_proba(self, frame):
        return sum(w * aligned_probabilities(model, frame, self.classes_)
                   for model, w in zip(self.models, self.weights))


def tune_blend(task, specs, seed, metric):
    _, counts = np.unique(task.y_train, return_counts=True)
    folds = min(3, int(counts.min()))
    if folds < 2:
        raise ValueError("Not enough training examples per class for stratified OOF.")
    oof = np.empty((len(specs), len(task.train), len(task.classes)), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fit, held in splitter.split(task.train, task.y_train):
        for index, spec in enumerate(specs):
            model = build_classifier(task, restore_spec(spec), seed)
            model.fit(task.train.iloc[fit], task.y_train[fit])
            oof[index, held] = aligned_probabilities(model, task.train.iloc[held], task.classes)
    # Prefer the equal blend when several weights tie.
    weight_grid = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
    weights = max(weight_grid, key=lambda weight: quality(
        metric, score(metric, task.y_train, np.average(oof, axis=0, weights=weight), task.classes)[0]
    ))
    return weights, oof, folds


def fit_frozen(task, spec, seed, features, labels):
    if spec["family"] == "oof_blend":
        models = [fit_frozen(task, item, seed, features, labels) for item in spec["members"]]
        return ProbabilityBlend(models, spec["weights"], task.classes)
    model = build_classifier(task, restore_spec(spec), seed)
    return model.fit(features, labels)
