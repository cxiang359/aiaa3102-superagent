"""Classification metrics with a shared class and probability ordering."""

import numpy as np
from sklearn.metrics import f1_score, log_loss, roc_auc_score

METRICS = ("accuracy", "macro_f1", "weighted_f1", "roc_auc", "log_loss")


def aligned_probabilities(model, frame, classes):
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    source = list(model.classes_)
    if set(source) != set(classes):
        raise ValueError("Fitted class set differs from the task class set.")
    probabilities = probabilities[:, [source.index(label) for label in classes]]
    if probabilities.shape != (len(frame), len(classes)):
        raise ValueError("Invalid probability matrix shape.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (
        probabilities > 1
    ).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid class probabilities.")
    return probabilities


def score(metric, labels, probabilities, classes):
    indices = np.argmax(probabilities, axis=1)
    predictions = np.asarray(classes)[indices]
    if len(predictions) != len(labels) or not np.isin(labels, classes).all():
        raise ValueError("Evaluation labels do not match the prediction contract.")
    if metric == "accuracy":
        value = np.mean(predictions == labels)
    elif metric in {"macro_f1", "weighted_f1"}:
        value = f1_score(labels, predictions, labels=classes, average=metric.removesuffix("_f1"), zero_division=0)
    elif metric == "roc_auc":
        if len(classes) == 2:
            value = roc_auc_score(np.asarray(labels) == classes[1], probabilities[:, 1])
        else:
            value = roc_auc_score(labels, probabilities, labels=classes, multi_class="ovr", average="macro")
    elif metric == "log_loss":
        value = log_loss(labels, probabilities, labels=classes)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    if not np.isfinite(value):
        raise ValueError(f"{metric} is undefined for this split.")
    return float(value), predictions


def quality(metric, value):
    return -value if metric == "log_loss" else value


def error_profile(labels, predictions, classes):
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(labels, predictions, labels=classes)
    return {"labels": list(map(str, classes)), "confusion_matrix": matrix.tolist()}
