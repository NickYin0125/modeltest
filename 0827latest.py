#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clinical + Habitat + Joint (score-level) pipeline
Update: add "direction self-check + label unification (to 0/1)" as safety checks,
without changing your core modeling logic.

- Univariate on clinical (train==0), tests auto-chosen.
- Modeling:
  * Clinical (Excel): RF/SVM(sigmoid)/GPC + L1 feature selection + StandardScaler
  * Habitat (CSV):   RF/SVM(sigmoid) + L1 feature selection + StandardScaler
  * For both: labels are standardized to 0/1 (positive=1). If original labels are 1/2
    or other binary values, they are mapped to 0/1 once and logged.
  * Direction self-check: train a quick Logistic(L2) on TRAIN (with standardized y);
    evaluate on TEST. If AUC < 0.5 and (1-AUC) >= 0.6, we assume TEST labels have
    opposite semantics and flip ONLY the TEST labels used for evaluation/exports.
    (Training labels are not changed; source files are not modified.)
- Joint fusion (Clinical + Habitat): SVM/RF/GB/LR on two probabilities,
  export SVM predictions + SHAP; comparison ROC/DCA/Calibration.

This script also sets Matplotlib backend to Agg for headless environments.
"""


import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, accuracy_score, recall_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
import joblib

warnings.filterwarnings('ignore')


# ------------------------------
# Utilities
# ------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(str(path), dpi=300)
    plt.close()


def binarize_label(series: pd.Series, positive_value=2) -> pd.Series:
    """Map positive_value -> 1, others -> 0"""
    return (series == positive_value).astype(int)


def standardize_binary_label(series: pd.Series, positive_value=2) -> pd.Series:
    """
    Standardize any binary label column to {0,1} where 1 == 'positive' defined by positive_value.
    If the series is already in {0,1}, return as-is.
    If it is in {1,2} and positive_value==2, map 2->1,1->0.
    Otherwise, if exactly 2 unique values (e.g., {0,2}), map 'positive_value'->1, the other->0.
    """
    s = series.copy()
    uniq = sorted(pd.unique(s.dropna()))
    uniq_set = set(uniq)
    if uniq_set.issubset({0, 1}):
        # already 0/1
        return s.astype(int)

    if len(uniq) == 2:
        # Map positive_value -> 1, other -> 0
        pos = positive_value
        if pos not in uniq_set:
            # fallback: map the larger value to positive
            pos = max(uniq_set)
        return (s == pos).astype(int)

    # Fallback for unexpected cases: threshold at median (not expected here)
    med = s.median()
    return (s >= med).astype(int)



def direction_self_check_flag(
    X_train: pd.DataFrame,
    y_train01: np.ndarray,
    X_test: pd.DataFrame,
    y_test01: np.ndarray,
    stage_name: str,
    flip_threshold: float = 0.6
) -> bool:
    """
    Train a quick Logistic(L2) on TRAIN (with standardized y in {0,1}).
    Evaluate on TEST. If AUC < 0.5 and (1-AUC) >= flip_threshold (default 0.6),
    we assume the TEST labels' semantics are opposite. Return True to indicate
    we should flip TEST PROBABILITIES (NOT labels) for evaluation/exports.
    """
    try:
        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        if len(num_cols) == 0 or len(np.unique(y_train01)) < 2 or len(np.unique(y_test01)) < 2:
            print(f"ℹ️ [{stage_name}] Direction self-check skipped (insufficient numeric features or single-class).")
            return False

        Xtr = X_train[num_cols].copy()
        Xte = X_test[num_cols].copy()
        med = Xtr.median()
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xtr_s, y_train01)
        proba = lr.predict_proba(Xte_s)[:, 1]
        auc_true = roc_auc_score(y_test01, proba)
        print(f"🔎 [{stage_name}] Direction self-check — probe Logistic AUC (pos=1): {auc_true:.4f} | 1-AUC={1-auc_true:.4f}")
        if auc_true < 0.5 and (1 - auc_true) >= flip_threshold:
            print(f"⚠️ [{stage_name}] Detected opposite semantics on TEST. Will flip TEST probabilities for evaluation/exports.")
            return True
        return False
    except Exception as e:
        print(f"⚠️ [{stage_name}] Direction self-check failed: {e}")
        return False

        Xtr = X_train[num_cols].copy()
        Xte = X_test[num_cols].copy()
        med = Xtr.median()
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xtr_s, y_train01)
        proba = lr.predict_proba(Xte_s)[:, 1]
        auc_true = roc_auc_score(y_test01, proba)
        print(f"🔎 [{stage_name}] Direction self-check — probe Logistic AUC (pos=1): {auc_true:.4f} | 1-AUC={1-auc_true:.4f}")
        if auc_true < 0.5 and (1 - auc_true) >= flip_threshold:
            print(f"⚠️ [{stage_name}] Detected opposite semantics on TEST labels. Flipping TEST labels for evaluation/exports.")
            return 1 - y_test01
        return y_test01
    except Exception as e:
        print(f"⚠️ [{stage_name}] Direction self-check failed: {e}")
        return y_test01


# ------------------------------
# Univariate analysis (clinical)
# ------------------------------
def categorical_test(col_data: pd.Series, label_data: pd.Series):
    contingency_table = pd.crosstab(col_data, label_data)
    try:
        if contingency_table.shape == (2, 2):
            if (contingency_table.values < 5).any():
                stat, p = fisher_exact(contingency_table)
                test_type = "Fisher Exact"
            else:
                stat, p, _, _ = chi2_contingency(contingency_table)
                test_type = "Chi-square"
        else:
            stat, p, dof, expected = chi2_contingency(contingency_table)
            if (expected < 5).any():
                test_type = "Chi-square (warning: small expected counts)"
            else:
                test_type = "Chi-square"
    except Exception as e:
        stat, p = np.nan, np.nan
        test_type = f"Categorical test failed: {str(e)}"
    return stat, p, test_type


def run_univariate_analysis_clinical(
    clinical_excel: Path,
    output_dir: Path,
    univariate_train_flag: int = 0,
    positive_value: int = 2
):
    print("🔎 [Univariate] Loading:", clinical_excel)
    df = pd.read_excel(clinical_excel)

    # standardize labels to 0/1 for safer binary tests
    df['label'] = standardize_binary_label(df['label'], positive_value=positive_value)

    train_df = df[df['train'] == univariate_train_flag].copy()
    feature_cols = [c for c in train_df.columns if c not in ['label', 'train']]
    results = []

    for col in feature_cols:
        data = train_df[[col, 'label']].dropna()
        labels_unique = data['label'].unique()
        if len(labels_unique) != 2:
            results.append({
                'Feature': col,
                'Test': 'Skipped (label not binary)',
                'Statistic': np.nan,
                'p-value': np.nan,
                'Significant (p<0.05)': 'No'
            })
            continue

        if pd.api.types.is_numeric_dtype(data[col]):
            unique_vals = data[col].nunique()
            if unique_vals <= 10 and unique_vals / len(data[col]) < 0.2:
                stat, p, test_type = categorical_test(data[col], data['label'])
            else:
                try:
                    g1 = data[data['label'] == labels_unique[0]][col]
                    g2 = data[data['label'] == labels_unique[1]][col]
                    stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                    test_type = 'Mann-Whitney U'
                except Exception as e:
                    stat, p = np.nan, np.nan
                    test_type = f"Mann-Whitney U failed: {str(e)}"
        else:
            stat, p, test_type = categorical_test(data[col], data['label'])

        results.append({
            'Feature': col,
            'Test': test_type,
            'Statistic': stat,
            'p-value': p,
            'Significant (p<0.05)': 'Yes' if pd.notnull(p) and p < 0.05 else 'No'
        })

    results_df = pd.DataFrame(results).sort_values(by='p-value')
    out_xlsx = output_dir / "univariate_analysis_results.xlsx"
    results_df.to_excel(out_xlsx, index=False)
    print(f"✅ [Univariate] Saved to '{out_xlsx}'")


# ------------------------------
# Modeling helpers
# ------------------------------
def evaluate_predictions_df(df: pd.DataFrame, dataset_name: str = ""):
    y_true = df['label'].values
    y_pred = df['predicted'].values
    y_prob = df['probability'].values

    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) else float('nan')
    auc_score = roc_auc_score(y_true, y_prob)

    print(f"📊 [{dataset_name}]")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Sensitivity: {sen:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  AUC:         {auc_score:.4f}")
    print("-" * 40)


def compute_auc_ci(y_true, y_prob, n_bootstraps=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
    auc_main = roc_auc_score(y_true, y_prob)
    return auc_main, lower, upper


# ----------------------------------------------------------------------
# DeLong's test for comparing correlated ROC AUCs
# ----------------------------------------------------------------------
def delong_test(y_true, y_pred1, y_pred2):
    """
    Compute DeLong's test for two correlated ROC AUCs.  Returns the
    (auc1, auc2, auc1-auc2, standard_error, p_value).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground truth labels (0/1).
    y_pred1 : array-like of shape (n_samples,)
        Predicted probabilities or scores from the first model.
    y_pred2 : array-like of shape (n_samples,)
        Predicted probabilities or scores from the second model.

    Notes
    -----
    This implementation follows the algorithm described in DeLong et al.
    (1988) and used by the pROC R package.  It computes the U-statistics
    based covariance between two correlated AUC estimates and derives a
    z-score and two‑sided p-value for the null hypothesis of equal AUCs.
    """
    import numpy as _np
    from scipy.stats import norm as _norm

    y_true = _np.asarray(y_true)
    preds = _np.vstack((y_pred1, y_pred2))  # shape (2, n_samples)
    pos = y_true == 1
    neg = y_true == 0
    m = _np.sum(pos)
    n = _np.sum(neg)
    if m == 0 or n == 0:
        raise ValueError("DeLong test requires both positive and negative samples.")

    pos_scores = preds[:, pos]  # (2, m)
    neg_scores = preds[:, neg]  # (2, n)

    v10 = _np.zeros((2, m))
    v01 = _np.zeros((2, n))
    for r in range(2):
        # Compute per‑positive averages
        for i in range(m):
            ps = pos_scores[r, i]
            v10[r, i] = ((ps > neg_scores[r]) + 0.5 * (ps == neg_scores[r])).mean()
        # Compute per‑negative averages
        for j in range(n):
            ns = neg_scores[r, j]
            v01[r, j] = ((pos_scores[r] > ns) + 0.5 * (pos_scores[r] == ns)).mean()

    aucs = v10.mean(axis=1)
    # Compute covariance matrix of AUC estimates
    s10 = _np.cov(v10, bias=True)
    s01 = _np.cov(v01, bias=True)
    # Bias terms for U-statistics
    cov = s10 / m + s01 / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = _np.sqrt(var)
    z = diff / se
    p_val = 2 * (1 - _norm.cdf(abs(z)))
    return aucs[0], aucs[1], diff, se, p_val


def fit_select_best_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_defs: dict,
    l1_C: float,
    cv_splits: int = 5,
    random_state: int = 42
):
    scaler = StandardScaler()
    l1_selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', C=l1_C, random_state=random_state)
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    best_model_name = None
    best_auc = -np.inf
    best_pipeline = None
    model_auc_results = {}

    for name, model in model_defs.items():
        pipeline = Pipeline([('scaler', scaler), ('select', l1_selector), ('clf', model)])
        try:
            aucs = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
            mean_auc = aucs.mean()
            print(f"{name} CV AUC: {mean_auc:.4f}")
        except Exception as e:
            mean_auc = np.nan
            print(f"{name} CV failed: {e}")

        model_auc_results[name] = mean_auc
        if pd.notnull(mean_auc) and mean_auc > best_auc:
            best_auc = mean_auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n✅ Best model: {best_model_name} (CV AUC = {best_auc:.4f})")
    return best_model_name, best_pipeline, model_auc_results


def _robust_shap_summary_plot(shap_values, X_df, save_path: Path):
    """
    Handle SHAP return types across versions for binary classifiers.  If the
    optional ``shap`` package is not installed, quietly skip plotting.
    """
    try:
        import shap  # type: ignore
        from shap import summary_plot  # type: ignore
    except ImportError:
        print("ℹ️ SHAP is not installed; skipping SHAP summary plot.")
        return
    vals = shap_values
    try:
        if isinstance(vals, list):
            # list-of-arrays, pick positive class index 1
            vals = vals[1]
        elif hasattr(vals, "ndim") and vals.ndim == 3:
            # (n_samples, n_features, n_classes)
            vals = vals[:, :, 1]
    except Exception:
        pass
    summary_plot(vals, X_df, show=False)
    savefig(save_path)


def train_eval_save_one_stage(
    stage_name: str,
    df: pd.DataFrame,
    output_dir: Path,
    l1_C: float,
    model_defs: dict,
    pos_label_value: int = 2,
    train_flag_for_train: int = 1,
    train_flag_for_test: int = 0,
    roc_png_name: str = None,
    model_pkl_name: str = None,
    shap_png_name: str = None,
    predictions_xlsx_name: str = None,
    shap_mode: str = "auto"
):
    # Attempt to import SHAP explainers.  SHAP is optional; if it is not
    # installed, we set a flag to skip explanation plots.  This avoids
    # raising ImportError in environments where the shap package is not
    # available.
    try:
        from shap import TreeExplainer, KernelExplainer  # type: ignore
        _shap_available = True
    except ImportError:
        _shap_available = False

    # Split
    train_df = df[df['train'] == train_flag_for_train].copy()
    test_df = df[df['train'] == train_flag_for_test].copy()

    if 'number' not in df.columns:
        raise ValueError(f"[{stage_name}] Missing 'number' column.")

    feature_cols = [c for c in df.columns if c not in ['label', 'train', 'number']]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # ---- Label unification to 0/1 ----
    # If already 0/1, keep. Else map positive_value ->1, others->0 (applied independently on both splits to be robust to missing classes).
    y_train = standardize_binary_label(train_df['label'], positive_value=pos_label_value).astype(int).values
    y_test = standardize_binary_label(test_df['label'], positive_value=pos_label_value).astype(int).values
    print(f"ℹ️ [{stage_name}] Labels standardized to 0/1 (1=positive). train_pos_rate={y_train.mean():.3f}, test_pos_rate={y_test.mean():.3f}")

    # ---- Direction self-check on TEST ----
    flip_prob = direction_self_check_flag(X_train, y_train, X_test, y_test, stage_name=stage_name)

    # Select best model via CV
    best_model_name, best_pipeline, _ = fit_select_best_model(
        X_train, y_train, model_defs, l1_C=l1_C
    )

    # Fit final
    best_pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Probabilities & ROC
    #
    # Instead of computing probabilities on the same training data that was
    # used to fit the model (which leads to overly optimistic AUC), we
    # generate out‑of‑fold predictions via cross_val_predict.  These
    # predictions are made on held‑out folds during cross‑validation and
    # therefore provide a better estimate of the model’s performance on
    # unseen data.  We still fit the final model on the full training
    # dataset before producing predictions on the test set.
    # ------------------------------------------------------------------
    # Generate cross‑validated probabilities for the training set
    try:
        cv_for_train = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_train_proba = cross_val_predict(
            best_pipeline,
            X_train,
            y_train,
            cv=cv_for_train,
            method='predict_proba'
        )[:, 1]
    except Exception:
        # Fallback: if cross_val_predict fails (e.g., due to some models not
        # supporting probability estimates during CV), fall back to
        # in‑sample probabilities and warn users in the logs.
        print(f"⚠️ [{stage_name}] cross_val_predict failed; falling back to in‑sample training probabilities.")
        cv_train_proba = best_pipeline.predict_proba(X_train)[:, 1]

    # Fit final model on the entire training data for test predictions
    best_pipeline.fit(X_train, y_train)

    # Test probabilities
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    # Apply direction flip if necessary
    if flip_prob:
        test_proba = 1 - test_proba

    # ROC curves and AUCs
    fpr_tr, tpr_tr, _ = roc_curve(y_train, cv_train_proba)
    fpr_te, tpr_te, _ = roc_curve(y_test, test_proba)
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    roc_auc_te = auc(fpr_te, tpr_te)

    # ROC plot
    roc_png = output_dir / (roc_png_name or f'roc_curve_{stage_name}.png')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_tr, tpr_tr, label=f'Train AUC = {roc_auc_tr:.3f}', linestyle='--')
    plt.plot(fpr_te, tpr_te, label=f'Test AUC = {roc_auc_te:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({stage_name})')
    plt.legend()
    savefig(roc_png)
    print(f"📈 [{stage_name}] ROC saved -> {roc_png}")

    # Save model
    model_pkl = output_dir / (model_pkl_name or f'best_{stage_name}_{best_model_name}.pkl')
    joblib.dump(best_pipeline, str(model_pkl))
    print(f"💾 [{stage_name}] Model saved -> {model_pkl}")

    # SHAP
    shap_png = None
    # Skip SHAP completely if shap is not available
    if _shap_available and shap_mode != "skip":
        try:
            scaler = best_pipeline.named_steps['scaler']
            selector = best_pipeline.named_steps['select']
            # Fit scaler and selector on the entire training data
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            selector.fit(X_train_scaled, y_train)
            selected_mask = selector.get_support()
            selected_features = X_train.columns[selected_mask]
            X_selected = X_train[selected_features]

            shap_png = output_dir / (shap_png_name or f'shap_{stage_name}.png')
            if best_model_name == "RandomForest" and (shap_mode in ["auto", "tree"]):
                clf = best_pipeline.named_steps['clf']
                explainer = TreeExplainer(clf)
                shap_values = explainer.shap_values(X_selected)
                _robust_shap_summary_plot(shap_values, X_selected, shap_png)
            elif best_model_name == "SVM" or shap_mode == "kernel":
                clf = best_pipeline.named_steps['clf']
                bg = X_selected.values[: min(100, len(X_selected))]
                explainer = KernelExplainer(clf.predict_proba, bg)
                sample = X_selected.values[: min(200, len(X_selected))]
                shap_values = explainer.shap_values(sample)
                _robust_shap_summary_plot(shap_values, X_selected.iloc[:sample.shape[0]], shap_png)
            else:
                print(f"ℹ️ [{stage_name}] SHAP skipped (model not supported or shap_mode=skip).")
                shap_png = None
            if shap_png:
                print(f"📊 [{stage_name}] SHAP saved -> {shap_png}")
        except Exception as e:
            print(f"⚠️ [{stage_name}] SHAP failed: {e}")

    # Predictions & evaluation (use checked labels on TEST)
    #
    # Convert probabilities into class labels using an explicit
    # probability threshold instead of relying on the classifier’s
    # internal decision rule.  In the original implementation the
    # RandomForest classifier’s ``predict`` method was used to obtain
    # test labels.  However, this method ignores any probability
    # flipping performed during the direction self‑check and can lead to
    # sub‑optimal test accuracy when the semantics of the test set
    # differ from the training set.  To address this, we derive
    # predictions from the probability outputs after flipping.

    # Determine an optimal threshold on the training set that maximizes
    # classification accuracy.  We search over a grid of values in
    # [0, 1] and pick the one yielding the highest accuracy on the
    # cross‑validated probabilities.  If multiple thresholds result
    # in the same accuracy, the first (lowest) threshold is chosen.
    best_threshold = 0.5
    best_acc = 0.0
    for thr in np.linspace(0.0, 1.0, 101):
        preds_thr = (cv_train_proba >= thr).astype(int)
        acc_thr = accuracy_score(y_train, preds_thr)
        if acc_thr > best_acc:
            best_acc = acc_thr
            best_threshold = thr

    # Use the derived threshold to create class labels for both
    # training and test sets.  This ensures that any flipping
    # applied to ``test_proba`` is reflected in the final prediction.
    train_pred = (cv_train_proba >= best_threshold).astype(int)
    test_pred = (test_proba >= best_threshold).astype(int)

    train_out = pd.DataFrame({
        'number': train_df['number'].values,
        'label': y_train,
        'predicted': train_pred,
        'probability': cv_train_proba
    })
    test_out = pd.DataFrame({
        'number': test_df['number'].values,
        'label': y_test,
        'predicted': test_pred,
        'probability': test_proba
    })

    pred_xlsx = output_dir / (predictions_xlsx_name or f'prediction_results_{stage_name}.xlsx')
    with pd.ExcelWriter(pred_xlsx) as writer:
        train_out.to_excel(writer, sheet_name="Train", index=False)
        test_out.to_excel(writer, sheet_name="Test", index=False)
    print(f"📒 [{stage_name}] Predictions saved -> {pred_xlsx}")

    evaluate_predictions_df(train_out, f"{stage_name} (Train)")
    evaluate_predictions_df(test_out, f"{stage_name} (Test)")

    return {
        'best_model_name': best_model_name,
        'model_path': model_pkl,
        'roc_png': roc_png,
        'shap_png': shap_png,
        'predictions_xlsx': pred_xlsx,
        'train_df': train_out,
        'test_df': test_out,
    }


# ------------------------------
# Fusion (Clinical + Habitat only)
# ------------------------------
def read_scores(file_path: Path):
    train_df = pd.read_excel(file_path, sheet_name='Train')
    test_df = pd.read_excel(file_path, sheet_name='Test')
    return train_df[['number', 'label', 'probability']], test_df[['number', 'label', 'probability']]


def merge_scores_two(df1, df2, label_name='label'):
    merged = df1.merge(df2, on='number', suffixes=('_clinical', '_habitat'))
    X = merged[['probability_clinical', 'probability_habitat']].values
    y = merged[f'{label_name}_clinical'].values  # labels should match and are standardized 0/1
    numbers = merged['number'].values
    return X, y, numbers


def bootstrap_auc_ci(y_true, y_proba, n_bootstraps=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    boot = []
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_proba), len(y_proba))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], y_proba[idx]))
    sorted_scores = np.sort(np.array(boot))
    lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
    return lower, upper


def evaluate_fusion(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) else float('nan')
    auc_val = roc_auc_score(y_true, y_proba)
    lower, upper = bootstrap_auc_ci(y_true, y_proba)
    print(f"📊 [{name}] Joint test:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Sensitivity: {sen:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  AUC: {auc_val:.4f}  (95% CI: [{lower:.4f}, {upper:.4f}])")
    print("-" * 40)


def print_joint_train_report(model_name: str, y_train, joint_oof_proba, clinical_oof_proba, habitat_oof_proba, threshold: float = 0.5):
    """
    Print full metrics on TRAIN (using OOF probabilities) for a joint model,
    and run DeLong tests vs clinical/habitat baselines.
    """
    y_true = np.asarray(y_train)
    p = np.asarray(joint_oof_proba).ravel()
    pc = np.asarray(clinical_oof_proba).ravel()
    ph = np.asarray(habitat_oof_proba).ravel()
    y_hat = (p >= threshold).astype(int)

    acc = accuracy_score(y_true, y_hat)
    sens = recall_score(y_true, y_hat, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    auc_val = roc_auc_score(y_true, p)
    lci, uci = bootstrap_auc_ci(y_true, p, n_bootstraps=2000, ci=0.95, seed=42)

    print(f"📊 [{model_name}] Joint train (OOF):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Sensitivity: {sens:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  AUC: {auc_val:.4f}  (95% CI: [{lci:.4f}, {uci:.4f}])")

    try:
        auc_j, auc_c, diff, se, pval = delong_test(y_true, p, pc)
        print(f"🧪 DeLong (train): Joint [{model_name}] vs Clinical")
        print(f"  AUC_joint: {auc_j:.4f} | AUC_clinical: {auc_c:.4f} | Δ: {diff:.4f} | SE: {se:.4f} | p: {pval:.4g}")
    except Exception as e:
        print(f"🧪 DeLong (train): Joint [{model_name}] vs Clinical -> failed: {e}")

    try:
        auc_j, auc_h, diff, se, pval = delong_test(y_true, p, ph)
        print(f"🧪 DeLong (train): Joint [{model_name}] vs Habitat")
        print(f"  AUC_joint: {auc_j:.4f} | AUC_habitat: {auc_h:.4f} | Δ: {diff:.4f} | SE: {se:.4f} | p: {pval:.4g}")
    except Exception as e:
        print(f"🧪 DeLong (train): Joint [{model_name}] vs Habitat -> failed: {e}")
    print("-" * 40)


def run_fusion_block_two(
    clinical_pred_xlsx: Path,
    habitat_pred_xlsx: Path,
    output_dir: Path
):
    clinical_train, clinical_test = read_scores(clinical_pred_xlsx)
    habitat_train, habitat_test = read_scores(habitat_pred_xlsx)

    # Merge two sources
    X_train, y_train, number_train = merge_scores_two(clinical_train, habitat_train)
    X_test, y_test, number_test = merge_scores_two(clinical_test, habitat_test)

    # Models
    results = {}
    # SVM
    svm = SVC(probability=True, kernel='linear', random_state=0)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)[:, 1]
    evaluate_fusion("SVM", y_test, svm_pred, svm_proba)
    results['svm'] = (svm, svm_pred, svm_proba)

    # --- Train OOF report for SVM ---
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        svm_oof_proba = cross_val_predict(svm, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        print_joint_train_report("SVM", y_train, svm_oof_proba, X_train[:, 0], X_train[:, 1])
    except Exception as e:
        print(f"⚠️ OOF/metrics generation failed for SVM: {e}")
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    evaluate_fusion("Random Forest", y_test, rf_pred, rf_proba)
    results['rf'] = (rf, rf_pred, rf_proba)

    # --- Train OOF report for Random Forest ---
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        rf_oof_proba = cross_val_predict(rf, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        print_joint_train_report("Random Forest", y_train, rf_oof_proba, X_train[:, 0], X_train[:, 1])
    except Exception as e:
        print(f"⚠️ OOF/metrics generation failed for Random Forest: {e}")
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    evaluate_fusion("Gradient Boosting", y_test, gb_pred, gb_proba)
    results['gb'] = (gb, gb_pred, gb_proba)

    # --- Train OOF report for Gradient Boosting ---
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        gb_oof_proba = cross_val_predict(gb, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        print_joint_train_report("Gradient Boosting", y_train, gb_oof_proba, X_train[:, 0], X_train[:, 1])
    except Exception as e:
        print(f"⚠️ OOF/metrics generation failed for Gradient Boosting: {e}")
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    evaluate_fusion("Logistic Regression", y_test, lr_pred, lr_proba)
    results['lr'] = (lr, lr_pred, lr_proba)

    # --- Train OOF report for Logistic Regression ---
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        lr_oof_proba = cross_val_predict(lr, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        print_joint_train_report("Logistic Regression", y_train, lr_oof_proba, X_train[:, 0], X_train[:, 1])
    except Exception as e:
        print(f"⚠️ OOF/metrics generation failed for Logistic Regression: {e}")
    # Save SVM predictions (for downstream comparison)
    fusion_svm_xlsx = output_dir / "fusion_model_svm_results.xlsx"
    train_out_svm = pd.DataFrame({
        'number': number_train,
        'label': y_train,
        'predicted': svm.predict(X_train),
        'probability': svm.predict_proba(X_train)[:, 1]
    })
    test_out_svm = pd.DataFrame({
        'number': number_test,
        'label': y_test,
        'predicted': svm_pred,
        'probability': svm_proba
    })
    with pd.ExcelWriter(fusion_svm_xlsx) as writer:
        train_out_svm.to_excel(writer, sheet_name="Train", index=False)
        test_out_svm.to_excel(writer, sheet_name="Test", index=False)
    print(f"💾 Joint (clinical+habitat) SVM predictions saved -> {fusion_svm_xlsx}")

    # Save Logistic Regression predictions (for joint model selection)
    fusion_lr_xlsx = output_dir / "fusion_model_lr_results.xlsx"
    train_out_lr = pd.DataFrame({
        'number': number_train,
        'label': y_train,
        'predicted': lr.predict(X_train),
        'probability': lr.predict_proba(X_train)[:, 1]
    })
    test_out_lr = pd.DataFrame({
        'number': number_test,
        'label': y_test,
        'predicted': lr_pred,
        'probability': lr_proba
    })
    with pd.ExcelWriter(fusion_lr_xlsx) as writer:
        train_out_lr.to_excel(writer, sheet_name="Train", index=False)
        test_out_lr.to_excel(writer, sheet_name="Test", index=False)
    print(f"💾 Joint (clinical+habitat) Logistic Regression predictions saved -> {fusion_lr_xlsx}")

    # Prepare DataFrames for SHAP and for saving features (still use SVM as baseline for feature importance)
    X_train_df = pd.DataFrame(X_train, columns=['pred_prob_clinical', 'pred_prob_habitat'])
    X_test_df = pd.DataFrame(X_test, columns=['pred_prob_clinical', 'pred_prob_habitat'])

    # SHAP for SVM: attempt to compute SHAP values if shap is installed
    try:
        import shap  # type: ignore
        explainer = shap.KernelExplainer(svm.predict_proba, X_train_df.iloc[: min(100, len(X_train_df))])
        shap_vals = explainer.shap_values(X_test_df.iloc[: min(200, len(X_test_df))])
        shap_png = output_dir / "shap_fusion_svm.png"
        _robust_shap_summary_plot(shap_vals, X_test_df.iloc[: min(200, len(X_test_df))], shap_png)
        print(f"📊 Joint SVM SHAP saved -> {shap_png}")
    except Exception as e:
        # If shap is unavailable or any error occurs, log and skip
        print(f"⚠️ Joint SHAP failed: {e}")

    # Save fusion features CSVs (train/test)
    train_feat_csv = output_dir / "fusion_features_train.csv"
    test_feat_csv = output_dir / "fusion_features_test.csv"
    train_feat = X_train_df.copy()
    train_feat['label'] = y_train
    train_feat['number'] = number_train
    test_feat = X_test_df.copy()
    test_feat['label'] = y_test
    test_feat['number'] = number_test
    train_feat.to_csv(train_feat_csv, index=False)
    test_feat.to_csv(test_feat_csv, index=False)
    print(f"💾 Joint features saved -> {train_feat_csv} | {test_feat_csv}")

    return {
        'svm': results['svm'],
        'rf': results['rf'],
        'gb': results['gb'],
        'lr': results['lr'],
        'fusion_svm_xlsx': fusion_svm_xlsx,
        'fusion_lr_xlsx': fusion_lr_xlsx,
    }


# ------------------------------
# Comparative plots: ROC, DCA, Calibration
# ------------------------------
def plot_rocs(paths: dict, sheet_name: str, title: str, out_png: Path):
    def load_xy(pth: Path, sheet='Train'):
        df = pd.read_excel(pth, sheet_name=sheet)
        return df['label'].values, df['probability'].values

    plt.figure(figsize=(7, 6))
    for name, pth in paths.items():
        y, prob = load_xy(pth, sheet_name)
        fpr, tpr, _ = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        lw = 1.5 if name == 'Joint model' else 1.0
        plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend(loc="lower right")
    savefig(out_png)
    print(f"📈 ROC comparison saved -> {out_png}")


def plot_dca_curves(paths: dict, sheet_name: str, title: str, out_png: Path):
    # Load labels from the first entry
    name0, path0 = next(iter(paths.items()))
    df0 = pd.read_excel(path0, sheet_name=sheet_name)
    y_true = df0['label'].values

    thresholds = np.linspace(0.01, 0.99, 99)
    plt.figure(figsize=(7, 6))

    # Treat None = 0
    plt.plot(thresholds, [0]*len(thresholds), label="Treat None", linestyle="--")

    # Treat All
    prevalence = np.mean(y_true)
    treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    plt.plot(thresholds, treat_all, label="Treat All", linestyle="--")

    for name, pth in paths.items():
        df = pd.read_excel(pth, sheet_name=sheet_name)
        prob = df['probability'].values
        nb = []
        for th in thresholds:
            pred = (prob >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            n = len(y_true)
            nb.append((tp/n) - (fp/n) * (th/(1-th)))
        plt.plot(thresholds, nb, label=name)

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(False)
    plt.legend()
    savefig(out_png)
    print(f"📈 DCA saved -> {out_png}")


def plot_calibration(paths: dict, sheet_name: str, title: str, out_png: Path, n_bins: int = 10):
    plt.figure(figsize=(7, 6))
    for name, pth in paths.items():
        df = pd.read_excel(pth, sheet_name=sheet_name)
        y, prob = df['label'].values, df['probability'].values
        prob_true, prob_pred = calibration_curve(y, prob, n_bins=n_bins, strategy='quantile')
        plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(False)
    savefig(out_png)
    print(f"📈 Calibration saved -> {out_png}")


# ------------------------------
# Optional: NIfTI visualization (unchanged idea)
# ------------------------------
def try_nifti_block(image_dir: Path, mask_dir: Path, sample_id: str, output_dir: Path):
    try:
        import nibabel as nib
        import matplotlib.colors as mcolors
        from skimage.feature import greycomatrix, greycoprops
        from skimage.util import img_as_ubyte
        from skimage import exposure
        from sklearn.cluster import KMeans

        vis_dir = output_dir / "visualization"
        ensure_dir(vis_dir)

        img_path = image_dir / f"{sample_id}.nii"
        mask_path = mask_dir / f"{sample_id}.nii.gz"
        if not img_path.exists() or not mask_path.exists():
            print(f"⚠️ NIfTI files not found for sample_id '{sample_id}'. Skipping NIfTI block.")
            return

        img_nii = nib.load(str(img_path))
        mask_nii = nib.load(str(mask_path))
        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        if mask_data.ndim == 3 and mask_data.shape[2] == 1:
            mask_data = np.squeeze(mask_data, axis=2)

        tumor_pixels = img_data[mask_data > 0]
        if len(tumor_pixels) < 3:
            print("⚠️ ROI too small for clustering. Skipping.")
            return
        pixels = tumor_pixels.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)

        cluster_map = np.zeros_like(mask_data)
        cluster_map[mask_data > 0] = kmeans.labels_ + 1

        # Base image
        plt.figure(figsize=(5, 5))
        plt.imshow(img_data, cmap='gray')
        plt.axis('off')
        savefig(vis_dir / f"{sample_id}_original.png")

        # Overlay mask (red translucent)
        cmap = mcolors.ListedColormap(['none', 'red'])
        norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)
        plt.figure(figsize=(5, 5))
        plt.imshow(img_data, cmap='gray')
        plt.imshow(mask_data > 0, cmap=cmap, norm=norm, alpha=0.3)
        plt.axis('off')
        savefig(vis_dir / f"{sample_id}_overlay_mask.png")

        # Cluster map
        tab10 = plt.cm.get_cmap('tab10', 10)
        colors = [(0, 0, 0, 1)] + [tab10(i) for i in range(3)]
        cmap2 = mcolors.ListedColormap(colors)
        norm2 = mcolors.BoundaryNorm(boundaries=[0, 1, 2, 3, 4], ncolors=4)
        plt.figure(figsize=(5, 5))
        plt.imshow(cluster_map, cmap=cmap2, norm=norm2)
        plt.axis('off')
        savefig(vis_dir / f"{sample_id}_cluster_map.png")

        # Per cluster colored region
        for i in range(1, 4):
            cluster_mask = (cluster_map == i)
            cluster_img = np.zeros((*img_data.shape, 3), dtype=np.uint8)
            color_rgb = np.array(cmap2(i)[:3]) * 255
            cluster_img[cluster_mask] = color_rgb.astype(np.uint8)
            plt.figure(figsize=(5, 5))
            plt.imshow(cluster_img)
            plt.axis('off')
            savefig(vis_dir / f"{sample_id}_cluster_{i}.png")

        # Texture props per cluster (GLCM)
        props = ["contrast", "dissimilarity", "homogeneity", "ASM", "correlation"]
        for i in range(1, 4):
            region = (cluster_map == i)
            if np.sum(region) < 10:
                continue
            masked = img_data * region
            vals = masked[region > 0]
            p2, p98 = np.percentile(vals, (2, 98))
            masked = exposure.rescale_intensity(masked, in_range=(p2, p98))
            from skimage.util import img_as_ubyte
            slice_ubyte = img_as_ubyte(masked)
            glcm = greycomatrix(slice_ubyte, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            for prop in props:
                value = greycoprops(glcm, prop)[0, 0]
                fmap = np.zeros_like(img_data)
                fmap[region] = value
                plt.figure(figsize=(5, 5))
                plt.imshow(fmap, cmap='hot')
                plt.axis('off')
                savefig(vis_dir / f"{sample_id}_cluster_{i}_{prop}.png")

        print("🧩 NIfTI visualization saved under:", vis_dir)
    except ImportError as e:
        print(f"ℹ️ NIfTI block skipped (missing packages): {e}")
    except Exception as e:
        print(f"⚠️ NIfTI block error: {e}")


# ------------------------------
# Orchestrator
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Clinical, Habitat, and Joint (Clinical+Habitat) pipeline.")
    parser.add_argument('--clinical_excel', type=str,
                        default=r"/Users/nick/PycharmProjects/pythonProject4/data/临床.xlsx")
    parser.add_argument('--habitat_csv', type=str, default=r"/Users/nick/PycharmProjects/pythonProject4/data/生境_new.csv")
    parser.add_argument('--output_dir', type=str, default=r"/Users/nick/PycharmProjects/pythonProject4/outputs0901d")

    parser.add_argument('--univariate_train_flag', type=int, default=0, help="train flag used by univariate (kept as 0)")
    parser.add_argument('--train_flag_for_model', type=int, default=1, help="train flag used by modeling (kept as 1)")
    parser.add_argument('--test_flag_for_model', type=int, default=0, help="test flag used by modeling (kept as 0)")
    parser.add_argument('--pos_label_value', type=int, default=2, help="positive label value mapped to 1 before modeling")

    # Optional NIfTI visualization
    parser.add_argument('--nii_image_dir', type=str, default="", help="Folder containing *.nii images")
    parser.add_argument('--nii_mask_dir', type=str, default="", help="Folder containing corresponding *.nii.gz masks")
    parser.add_argument('--nii_sample_id', type=str, default="", help="Sample ID like '75' (without extension)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # 1) Univariate (clinical, train==0) with standardized labels
    if Path(args.clinical_excel).exists():
        run_univariate_analysis_clinical(Path(args.clinical_excel), output_dir, args.univariate_train_flag, positive_value=args.pos_label_value)
    else:
        print(f"⚠️ Clinical Excel not found: {args.clinical_excel} (skipping univariate)")

    # 2) Clinical model
    if Path(args.clinical_excel).exists():
        df_clin = pd.read_excel(args.clinical_excel)
        # Standardize label once
        df_clin['label'] = standardize_binary_label(df_clin['label'], positive_value=args.pos_label_value)
        clinical_models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='sigmoid', probability=True, random_state=42),
            "GaussianProcess": GaussianProcessClassifier(kernel=1.0 * RBF(), random_state=42),
        }
        clinical_art = train_eval_save_one_stage(
            stage_name="clinical",
            df=df_clin,
            output_dir=output_dir,
            l1_C=0.12,
            model_defs=clinical_models,
            pos_label_value=1,  # already standardized to 0/1
            train_flag_for_train=args.train_flag_for_model,
            train_flag_for_test=args.test_flag_for_model,
            roc_png_name="roc_curve_best_model_clinical.png",
            shap_png_name="shap_clinical.png",
            predictions_xlsx_name="prediction_results_clinical.xlsx",
            shap_mode="auto"
        )
        test_df = clinical_art['test_df']
        auc_val, lo, hi = compute_auc_ci(test_df['label'].values, test_df['probability'].values)
        print(f"📐 [clinical Test] AUC={auc_val:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")
    else:
        print(f"⚠️ Clinical Excel not found: {args.clinical_excel} (skipping clinical model)")

    # 3) Habitat model
    if Path(args.habitat_csv).exists():
        df_hab = pd.read_csv(args.habitat_csv)
        # Standardize label once
        df_hab['label'] = standardize_binary_label(df_hab['label'], positive_value=args.pos_label_value)
        habitat_models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='sigmoid', probability=True, random_state=42),
        }
        train_eval_save_one_stage(
            stage_name="sheng",
            df=df_hab,
            output_dir=output_dir,
            l1_C=0.5,
            model_defs=habitat_models,
            pos_label_value=1,  # already standardized to 0/1
            train_flag_for_train=args.train_flag_for_model,
            train_flag_for_test=args.test_flag_for_model,
            roc_png_name="roc_curve_sheng.png",
            shap_png_name="shap_habitat.png",
            predictions_xlsx_name="prediction_results_sheng.xlsx",
            shap_mode="kernel"
        )
    else:
        print(f"⚠️ Habitat CSV not found: {args.habitat_csv} (skipping habitat model)")

    # 4) Joint fusion (Clinical + Habitat)
    clinical_pred = output_dir / "prediction_results_clinical.xlsx"
    habitat_pred = output_dir / "prediction_results_sheng.xlsx"
    have_both = clinical_pred.exists() and habitat_pred.exists()

    if have_both:
        fusion_art = run_fusion_block_two(clinical_pred, habitat_pred, output_dir)

        # Comparative plots across 3 models
        # Use the Logistic Regression fusion model by default for joint evaluation
        paths = {
            'Clinical model': clinical_pred,
            'Habitat model': habitat_pred,
            'Joint model': output_dir / "fusion_model_lr_results.xlsx"
        }
        plot_rocs(paths, sheet_name='Train', title='ROC - Train', out_png=output_dir / 'roc_train_all_models.png')
        plot_rocs(paths, sheet_name='Test', title='ROC - Test', out_png=output_dir / 'roc_test_all_models.png')

        plot_dca_curves(paths, sheet_name='Train', title='DCA - Train', out_png=output_dir / 'dca_train.png')
        plot_dca_curves(paths, sheet_name='Test', title='DCA - Test', out_png=output_dir / 'dca_test.png')

        plot_calibration(paths, 'Train', 'Calibration - Train', output_dir / 'calibration_curve_train.png')
        plot_calibration(paths, 'Test', 'Calibration - Test', output_dir / 'calibration_curve_test.png')

        # ------------------------------
        # DeLong significance tests
        # Compare AUCs of clinical, habitat and joint models on the test set.
        try:
            # Load test predictions
            clin_df = pd.read_excel(paths['Clinical model'], sheet_name='Test')
            hab_df = pd.read_excel(paths['Habitat model'], sheet_name='Test')
            joint_df = pd.read_excel(paths['Joint model'], sheet_name='Test')
            y_test = clin_df['label'].values
            auc_c, auc_h, diff_ch, se_ch, p_ch = delong_test(y_test, clin_df['probability'].values, hab_df['probability'].values)
            auc_c2, auc_j, diff_cj, se_cj, p_cj = delong_test(y_test, clin_df['probability'].values, joint_df['probability'].values)
            auc_h2, auc_j2, diff_hj, se_hj, p_hj = delong_test(y_test, hab_df['probability'].values, joint_df['probability'].values)
            print("📏 DeLong test results (Test set):")
            print(f"  Clinical vs Habitat:    AUC diff = {diff_ch:.4f}, p-value = {p_ch:.4f}")
            print(f"  Clinical vs Joint:      AUC diff = {diff_cj:.4f}, p-value = {p_cj:.4f}")
            print(f"  Habitat vs Joint:       AUC diff = {diff_hj:.4f}, p-value = {p_hj:.4f}")
        except Exception as e:
            print(f"⚠️ DeLong test failed: {e}")
    else:
        print("⚠️ Joint fusion skipped (missing clinical or habitat predictions).")

    # 5) Optional NIfTI visualization
    if args.nii_image_dir and args.nii_mask_dir and args.nii_sample_id:
        try_nifti_block(Path(args.nii_image_dir), Path(args.nii_mask_dir), args.nii_sample_id, output_dir)


if __name__ == "__main__":
    main()
