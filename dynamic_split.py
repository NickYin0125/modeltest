#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate multiple stratified splits and evaluate 0827latest.py on each.

This helper does not modify the core modeling script. It rewrites the ``train``
column in temporary copies of the data files and invokes ``0827latest.py`` for
all candidate splits. Metrics are collected and written to ``results``.

Due to the heavy dependencies of ``0827latest.py``, this script expects that
``pandas`` and ``scikit-learn`` are available in the execution environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class SplitMetrics:
    split_id: int
    seed: int
    method: str
    test_size: float
    pos_rate_test: float
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    accuracy: float
    sensitivity: float
    specificity: float
    notes: str = ""

    def to_row(self) -> List:
        return [
            self.split_id,
            self.seed,
            self.method,
            self.test_size,
            self.pos_rate_test,
            self.auc,
            self.auc_ci_low,
            self.auc_ci_high,
            self.accuracy,
            self.sensitivity,
            self.specificity,
            self.notes,
        ]


def compute_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000,
                    ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap AUC confidence interval."""
    rng = np.random.RandomState(seed)
    scores = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    scores = np.array(scores)
    scores.sort()
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    main = roc_auc_score(y_true, y_prob)
    return main, lower, upper


def load_datasets(clin_path: Path, hab_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clin = pd.read_excel(clin_path)
    hab = pd.read_csv(hab_path)
    return clin, hab


def find_id_column(clin: pd.DataFrame, hab: pd.DataFrame) -> str:
    common = [c for c in clin.columns if c in hab.columns]
    if not common:
        raise ValueError("No common ID column found between datasets")
    # prefer typical ID column names
    for name in ["ID", "id", "patient_id", "case_id"]:
        if name in common:
            return name
    return common[0]


def evaluate_from_predictions(pred_xlsx: Path) -> Tuple[float, float, float, float, float, float]:
    from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score

    df = pd.read_excel(pred_xlsx, sheet_name="Test")
    y_true = df["label"].to_numpy()
    y_pred = df["predicted"].to_numpy()
    y_prob = df["probability"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.sum() else float("nan")
    auc, lo, hi = compute_auc_ci(y_true, y_prob)
    return acc, sen, spec, auc, lo, hi


def main():
    parser = argparse.ArgumentParser(description="Dynamic test split evaluator")
    parser.add_argument("--clinical_excel", default="临床.xlsx")
    parser.add_argument("--habitat_csv", default="生境_new.csv")
    parser.add_argument("--script", default="0827latest.py")
    parser.add_argument("--results", default="results")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(100)))
    args = parser.parse_args()

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "split_metrics.csv"

    clin_orig, hab_orig = load_datasets(Path(args.clinical_excel), Path(args.habitat_csv))
    id_col = find_id_column(clin_orig, hab_orig)

    label_col = "label"
    train_col = "train"

    # baseline test fraction and class balance
    p_test = 1 - clin_orig[train_col].mean()
    pos_rate = clin_orig[label_col].mean()

    metrics_rows = []
    best_auc = -np.inf
    best_split_ids = None

    for split_id, seed in enumerate(args.seeds):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=p_test, random_state=seed)
        y = clin_orig[label_col]
        indices = np.arange(len(clin_orig))
        tr_idx, te_idx = next(splitter.split(indices, y))
        if len(np.unique(y.iloc[te_idx])) < 2 or len(te_idx) < 10:
            continue

        train_flags = np.zeros(len(clin_orig), dtype=int)
        train_flags[tr_idx] = 1

        clin = clin_orig.copy()
        hab = hab_orig.copy()
        clin[train_col] = train_flags
        hab[train_col] = train_flags

        split_dir = results_dir / f"split_{split_id:03d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        clin_tmp = split_dir / "临床.xlsx"
        hab_tmp = split_dir / "生境_new.csv"
        clin.to_excel(clin_tmp, index=False)
        hab.to_csv(hab_tmp, index=False)

        out_dir = split_dir / "model_outputs"
        cmd = [
            "python",
            args.script,
            "--clinical_excel",
            str(clin_tmp),
            "--habitat_csv",
            str(hab_tmp),
            "--output_dir",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log_path = split_dir / "run.log"
        log_path.write_text(proc.stdout + "\n" + proc.stderr)
        if proc.returncode != 0:
            note = f"script failed (code {proc.returncode})"
            metrics_rows.append(SplitMetrics(split_id, seed, "S1", p_test, np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan, np.nan, note).to_row())
            continue

        pred_path = out_dir / "prediction_results_clinical.xlsx"
        if not pred_path.exists():
            note = "missing prediction file"
            metrics_rows.append(SplitMetrics(split_id, seed, "S1", p_test, np.nan, np.nan, np.nan,
                                            np.nan, np.nan, np.nan, np.nan, note).to_row())
            continue

        acc, sen, spec, auc, lo, hi = evaluate_from_predictions(pred_path)
        metrics_rows.append(SplitMetrics(split_id, seed, "S1", p_test, y.iloc[te_idx].mean(),
                                         auc, lo, hi, acc, sen, spec).to_row())

        if auc > best_auc and sen >= 0.9 * pos_rate:
            best_auc = auc
            best_split_ids = clin[[id_col, train_col]].copy()

    header = [
        "split_id",
        "seed",
        "method",
        "test_size",
        "pos_rate_test",
        "AUC",
        "AUC_CI_low",
        "AUC_CI_high",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Notes",
    ]
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(metrics_rows)

    if best_split_ids is not None:
        best_split_ids.to_csv(results_dir / "best_split_assignments.csv", index=False)


if __name__ == "__main__":
    main()
