"""Run and compare bounded candidate searches, then evaluate a frozen winner."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import requests

from .campus_policy import CAMPUS_MODEL, TRIAL_PRICES, CampusPolicy
from .classifiers import TrialSpec, catalog, dependencies, variants
from .ledger import BudgetExceeded, CostLedger
from .ensemble import fit_frozen, restore_spec, tune_blend
from .metrics import aligned_probabilities, error_profile, quality, score
from .task_data import load_development


def _write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _run_lock(run_dir):
    # OS-managed locks are released when a process exits, including after a crash.
    with (run_dir / ".run.lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Run is already active: {run_dir}") from error
        yield


def _digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _development_files(task):
    if task.layout == "image_folder":
        from .image_data import folder_samples

        names = tuple(p.relative_to(task.root).as_posix() for split in ("train", "val")
                      for p, _ in folder_samples(task.root / split))
    elif task.layout == "csv":
        names = ("train.csv", "val.csv")
    else:
        names = ("C_train.npy", "y_train.npy", "C_val.npy", "y_val.npy")
    return {name: _digest(task.root / name) for name in names}


def _fit_trial(task, run_dir, trial_id, spec, parent, reason, seed, metric, oof=None):
    directory = run_dir / "candidates" / str(trial_id)
    directory.mkdir(parents=True)
    spec = asdict(spec) if isinstance(spec, TrialSpec) else spec
    record = {"id": trial_id, "parent_id": parent, "spec": spec, "reason": reason}
    started = time.monotonic()
    try:
        model = fit_frozen(task, spec, seed, task.train, task.y_train)
        probabilities = aligned_probabilities(model, task.val, task.classes)
        value, prediction = score(metric, task.y_val, probabilities, task.classes)
        np.save(directory / "val_probs.npy", probabilities)
        _write_json(directory / "class_order.json", task.classes.tolist())
        np.save(directory / "val_predictions.npy", prediction)
        if oof is not None:
            np.save(directory / "train_oof_probs.npy", oof)
        joblib.dump(model, directory / "model.joblib")
        record.update(
            status="success", val_score=value, val_accuracy=float(np.mean(task.y_val == prediction)),
            diagnostic=error_profile(task.y_val, prediction, task.classes),
        )
        if task.modality == "paired_text":
            a, b = task.text_columns
            reversed_frame = task.val.rename(columns={a: b, b: a})
            reversed_probs = aligned_probabilities(model, reversed_frame, task.classes)
            reversed_labels = task.classes[reversed_probs.argmax(axis=1)]
            record["swap_consistency"] = float(np.mean(reversed_labels != prediction))
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["seconds"] = round(time.monotonic() - started, 3)
    _write_json(directory / "result.json", record)
    print(f"candidate={trial_id} {spec['family']}/{spec['variant']} {record['status']} "
          f"val_accuracy={record.get('val_accuracy')}", flush=True)
    return record


def _rank(history, metric):
    return sorted(
        (entry for entry in history if entry["status"] == "success"),
        key=lambda entry: (-quality(metric, entry["val_score"]), entry["id"]),
    )


def _refinement_options(history, available, beam_width, metric):
    ranked = _rank(history, metric)
    parents = []
    families = set()
    for parent in ranked:
        family = parent["spec"]["family"]
        if family not in families:
            parents.append(parent)
            families.add(family)
        if len(parents) == beam_width:
            break
    refinements = [entry for entry in history if entry["parent_id"] is not None]
    parents.sort(key=lambda parent: sum(
        entry["spec"]["family"] == parent["spec"]["family"] for entry in refinements
    ))
    groups = [
        [(spec, parent["id"]) for spec in variants(restore_spec(parent["spec"]), available)]
        for parent in parents
    ]
    # Interleave options so a second retained family can actually receive trials.
    options = [option for batch in zip_longest(*groups) for option in batch if option]
    options.extend((TrialSpec(family), None) for family in available)
    seen = {restore_spec(entry["spec"]) for entry in history}
    unique = []
    for spec, parent in options:
        if spec not in seen:
            unique.append((spec, parent))
            seen.add(spec)
    return unique


def _choose(policy, profile, options, history, count, stage):
    if policy is None:
        return [(index, "Deterministic portfolio order") for index in range(count)]
    if len(options) == count:
        return [(index, "All remaining eligible candidates") for index in range(count)]
    offered = [{"id": index, **asdict(spec), "parent_id": parent} for index, (spec, parent) in enumerate(options)]
    observations = [
        {key: entry[key] for key in ("id", "spec", "status", "val_score", "diagnostic", "error") if key in entry}
        for entry in history[-12:]
    ]
    return policy.choose(profile, offered, observations, count, stage)


def run_search(task, run_root, name, *, backend, strategy, max_trials, budget, seed,
               key_file=None, metric="accuracy", direct=False, feature_search=False, oof_ensemble=False):
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Task name must be a single path component.")
    if strategy not in {"single", "multi"} or backend not in {"offline", "campus"}:
        raise ValueError("Invalid structured search mode.")
    if max_trials < 1 or not 0 < budget <= 20:
        raise ValueError("Use at least one trial and a budget in (0, 20] RMB.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_root).resolve() / name / f"run_{stamp}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    config = {
        "task_dir": str(task.root), "label": task.label, "name": name, "backend": backend,
        "strategy": strategy, "max_trials": max_trials, "budget": budget, "seed": seed,
        "metric": metric, "direct": direct, "feature_search": feature_search,
        "oof_ensemble": oof_ensemble, "development_sha256": _development_files(task),
        "dependencies": dependencies(),
    }
    with _run_lock(run_dir):
        _write_json(run_dir / "logs" / "search_config.json", config)
        return _continue_search(task, run_dir, config, key_file)


def resume_search(run_dir, *, key_file=None, direct=False):
    run_dir = Path(run_dir).resolve(strict=True)
    with _run_lock(run_dir):
        if (run_dir / "final").exists():
            raise ValueError("This run has already begun final evaluation.")
        config = _read_json(run_dir / "logs" / "search_config.json")
        summary = run_dir / "logs" / "run_summary.json"
        if summary.is_file() and _read_json(summary)["status"] == "success":
            raise ValueError("Search already completed. Evaluate its frozen result instead.")
        task = load_development(config["task_dir"], config["label"])
        if _development_files(task) != config["development_sha256"] or dependencies() != config["dependencies"]:
            raise ValueError("Development files or runtime dependencies changed since the run began.")
        if direct:
            config["direct"] = True
            _write_json(run_dir / "logs" / "search_config.json", config)
        return _continue_search(task, run_dir, config, key_file)


def _continue_search(task, run_dir, config, key_file):
    started = time.monotonic()
    name, backend, strategy = (config[key] for key in ("name", "backend", "strategy"))
    max_trials, budget, seed, metric = (config[key] for key in ("max_trials", "budget", "seed", "metric"))
    direct, feature_search, oof_ensemble = (config[key] for key in ("direct", "feature_search", "oof_ensemble"))
    logs = run_dir / "logs"
    ledger = CostLedger(budget, logs / "llm_calls.jsonl", prices={CAMPUS_MODEL: TRIAL_PRICES})
    available = catalog(task)
    profile = task.profile()
    profile["dependency_versions"] = config["dependencies"]
    profile["metric"] = metric
    profile["maximize_metric"] = metric != "log_loss"
    hashes = config["development_sha256"]
    _write_json(logs / "profile.json", profile)
    candidates = run_dir / "candidates"
    previous = sorted((p for p in candidates.iterdir() if p.is_dir() and p.name.isdecimal()),
                      key=lambda p: int(p.name)) if candidates.exists() else []
    history = [_read_json(p / "result.json") for p in previous if (p / "result.json").is_file()]
    next_id = max((int(p.name) for p in previous), default=-1) + 1
    stop_reason, failure, notes = "trial_limit", None, []

    def run_trial(spec, parent, reason, oof=None):
        nonlocal next_id
        history.append(_fit_trial(task, run_dir, next_id, spec, parent, reason, seed, metric, oof))
        next_id += 1
        _write_json(logs / "checkpoint.json", {"history": history, "cost": ledger.summary()})

    starts = [(TrialSpec(family), None) for family in available]
    count = min(3 if strategy == "multi" else 1, max_trials, len(starts))
    try:
        # Always preserve a usable independent baseline before the first API call.
        if not history:
            base, _ = starts[0]
            run_trial(base, None, "Fast registered baseline")
        policy = CampusPolicy(ledger, key_file, direct=direct) if backend == "campus" else None
        initial_path = logs / "initial_plan.json"
        if count > 1:
            remaining = starts[1:]
            if not initial_path.exists():
                choices = _choose(policy, profile, remaining, history, count - 1, "initial_design")
                _write_json(initial_path, choices)
            for index, reason in _read_json(initial_path):
                spec, parent = remaining[index]
                if len(history) < max_trials and not any(row["spec"]["family"] == spec.family for row in history):
                    run_trial(spec, parent, reason)
        has_features = any(row["spec"].get("features") for row in history)
        if policy and feature_search and task.modality == "tabular" and len(history) < max_trials and not has_features:
            baseline = next(iter(_rank(history, metric)), None)
            if baseline:
                try:
                    features, reason = policy.propose_features(profile, baseline.get("diagnostic"))
                    spec = replace(restore_spec(baseline["spec"]), features=features)
                    run_trial(spec, baseline["id"], reason)
                except ValueError as error:
                    notes.append(f"Feature proposal skipped: {error}")
        while len(history) < max_trials:
            options = _refinement_options(history, available, 2 if strategy == "multi" else 1, metric)
            if not options:
                stop_reason = "portfolio_exhausted"
                break
            (index, reason), = _choose(policy, profile, options, history, 1, "refine")
            spec, parent = options[index]
            run_trial(spec, parent, reason)
    except BudgetExceeded as error:
        stop_reason, failure = "budget", str(error)
    except (requests.RequestException, ValueError, KeyError, TypeError, IndexError, OSError) as error:
        stop_reason, failure = "policy_error", f"{type(error).__name__}: {error}"
    if oof_ensemble and failure is None and not any(row["spec"]["family"] == "oof_blend" for row in history):
        ranked = _rank(history, metric)
        if ranked:
            first = ranked[0]
            second = next((row for row in ranked if row["spec"]["family"] != first["spec"]["family"]), None)
            if second:
                specs = [first["spec"], second["spec"]]
                try:
                    weights, oof, folds = tune_blend(task, specs, seed, metric)
                    run_trial({"family": "oof_blend", "variant": "weighted", "members": specs,
                               "weights": weights, "folds": folds}, first["id"],
                              "Two-family blend, weights selected on training OOF", oof)
                except (ValueError, RuntimeError, ImportError) as error:
                    notes.append(f"OOF blend skipped: {error}")
    best = next(iter(_rank(history, metric)), None)
    status = "success" if best is not None and failure is None else ("partial" if best else "failed")
    summary = {
        "status": status, "backend": backend, "strategy": strategy, "task": name,
        "metric": metric, "seed": seed, "max_trials": max_trials, "notes": notes,
        "best": best, "history": history, "stop_reason": stop_reason, "failure": failure,
        "seconds": round(time.monotonic() - started, 3), "cost": ledger.summary(),
        "cost_basis": "DeepSeek peak uncached rate used by the trial; course-date rate not certified",
        "run_dir": str(run_dir),
    }
    if best:
        directory = run_dir / "candidates" / str(best["id"])
        _write_json(run_dir / "frozen.json", {
            "spec": best["spec"], "candidate_id": best["id"], "seed": seed, "metric": metric,
            "task_dir": str(task.root), "label": task.label, "development_sha256": hashes,
            "validation_predictions_sha256": _digest(directory / "val_predictions.npy"),
            "validation_probs_sha256": _digest(directory / "val_probs.npy"),
            "validation_model_sha256": _digest(directory / "model.joblib"),
        })
    _write_json(logs / "run_summary.json", summary)
    return summary


def finalize_run(run_dir):
    run_dir = Path(run_dir).resolve()
    with _run_lock(run_dir):
        return _finalize_locked(run_dir)


def _finalize_locked(run_dir):
    frozen = _read_json(run_dir / "frozen.json")
    task = load_development(frozen["task_dir"], frozen["label"])
    if _development_files(task) != frozen["development_sha256"]:
        raise ValueError("Development files changed after freezing the candidate.")
    candidate = run_dir / "candidates" / str(frozen["candidate_id"])
    for filename, field in (("val_predictions.npy", "validation_predictions_sha256"),
                            ("model.joblib", "validation_model_sha256")):
        if _digest(candidate / filename) != frozen[field]:
            raise ValueError(f"Frozen candidate artifact changed: {filename}")
    if "validation_probs_sha256" in frozen and _digest(candidate / "val_probs.npy") != frozen["validation_probs_sha256"]:
        raise ValueError("Frozen validation probabilities changed.")
    final = run_dir / "final"
    final.mkdir()  # A dedicated directory reserves the one final evaluation.
    try:
        spec = frozen["spec"]
        model = fit_frozen(task, spec, frozen["seed"], pd.concat([task.train, task.val], ignore_index=True),
                           np.concatenate([task.y_train, task.y_val]))
        probabilities = aligned_probabilities(model, task.test_features(), task.classes)
        labels = task.test_labels()
        metric = frozen.get("metric", "accuracy")
        if labels is None:
            value, accuracy = None, None
            predictions = task.classes[probabilities.argmax(axis=1)]
        else:
            value, predictions = score(metric, labels, probabilities, task.classes)
            accuracy = float(np.mean(labels == predictions))
        np.save(final / "test_probs.npy", probabilities)
        _write_json(final / "class_order.json", task.classes.tolist())
        np.save(final / "test_predictions.npy", predictions)
        joblib.dump(model, final / "model.joblib")
        report = {
            "status": "success", "metric": metric, "test_score": value,
            "test_accuracy": accuracy, "test_rows": len(predictions),
            "candidate_id": frozen["candidate_id"], "spec": frozen["spec"],
            "fit_split": "train+val", "run_dir": str(run_dir),
        }
        _write_json(final / "result.json", report)
        return report
    except Exception as error:
        _write_json(final / "result.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise



def compare_and_evaluate(task, run_root, name, *, backend, max_trials, budget, seed, key_file=None, **options):
    """Use validation to select a search result, then read test data once."""
    if not 0 < budget <= 20:
        raise ValueError("Combined budget must be in (0, 20] RMB.")
    single = run_search(
        task, run_root, f"{name}_single", backend=backend, strategy="single",
        max_trials=max_trials, budget=budget / 2, seed=seed, key_file=key_file, **options,
    )
    if single["status"] != "success":
        return {"status": "incomplete", "single": single}
    multi = run_search(
        task, run_root, f"{name}_multi", backend=backend, strategy="multi",
        max_trials=max_trials, budget=budget / 2, seed=seed, key_file=key_file, **options,
    )
    if multi["status"] != "success":
        return {"status": "incomplete", "single": single, "multi": multi}
    return evaluate_comparison(single, multi)


def evaluate_comparison(single, multi):
    if single["status"] != "success" or multi["status"] != "success":
        raise ValueError("Two successful development runs are needed for comparison.")
    if single["strategy"] != "single" or multi["strategy"] != "multi":
        raise ValueError("Expected one single-start run and one multi-start run.")
    single_frozen = _read_json(Path(single["run_dir"]) / "frozen.json")
    multi_frozen = _read_json(Path(multi["run_dir"]) / "frozen.json")
    shared = ("task_dir", "development_sha256", "label")
    if any(single_frozen[key] != multi_frozen[key] for key in shared):
        raise ValueError("Comparison runs used different data or labels.")
    shared = ("backend", "seed", "max_trials", "metric")
    if any(single[key] != multi[key] for key in shared):
        raise ValueError("Comparison runs used different settings.")
    if single["cost"]["budget_rmb"] != multi["cost"]["budget_rmb"]:
        raise ValueError("Comparison runs used different budget ceilings.")
    # Tie-break to the simpler single-start policy, without inspecting test data.
    winner = max((single, multi), key=lambda result: quality(
        result["metric"], result["best"].get("val_score", result["best"]["val_accuracy"])
    ))
    final = finalize_run(winner["run_dir"])
    report = {
        "status": "success", "backend": winner["backend"], "selected_strategy": winner["strategy"],
        "single": single, "multi": multi, "final": final,
    }
    _write_json(Path(winner["run_dir"]) / "final" / "comparison.json", report)
    return report
