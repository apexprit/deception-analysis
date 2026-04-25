"""Comprehensive evaluation metrics for deception detection.

This module provides a DeceptionMetrics class that computes a wide range of
classification metrics, fairness metrics, and confidence intervals for
evaluating deception detection models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)
from sklearn.utils import resample
from scipy import stats


logger = logging.getLogger(__name__)


class DeceptionMetrics:
    """Compute and aggregate evaluation metrics for deception detection.

    This class provides methods for computing standard classification metrics,
    per-subject metrics, fairness metrics, and bootstrap confidence intervals.

    Attributes:
        random_state: Random seed for reproducibility.
        n_bootstrap: Number of bootstrap samples for confidence intervals.
        ci_level: Confidence level for intervals (default 0.95).
    """

    def __init__(
        self,
        random_state: int = 42,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
    ) -> None:
        """Initialize metrics calculator.

        Args:
            random_state: Random seed for reproducibility.
            n_bootstrap: Number of bootstrap samples for confidence intervals.
            ci_level: Confidence level for intervals (default 0.95).
        """
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        np.random.seed(random_state)

    def compute_all_metrics(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_prob: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics.

        Args:
            y_true: Ground truth binary labels (0 = truthful, 1 = deceptive).
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities for the positive class (deceptive).
                If provided, ROC‑AUC and average precision are computed.

        Returns:
            Dictionary containing:
                - accuracy, precision, recall, f1, specificity
                - balanced_accuracy
                - auc_roc, average_precision (if y_prob provided)
                - confusion_matrix (as 2x2 array)
                - classification_report (as dict)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.ndim != 1:
            raise ValueError(f"y_true must be 1‑D, got shape {y_true.shape}")
        if y_pred.ndim != 1:
            raise ValueError(f"y_pred must be 1‑D, got shape {y_pred.shape}")
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
            )

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Specificity (true negative rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Probability‑based metrics
        auc_roc = None
        avg_precision = None
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]  # take positive class probabilities
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
            except ValueError as e:
                logger.warning(f"Could not compute ROC‑AUC: {e}")
                auc_roc = None
            try:
                avg_precision = average_precision_score(y_true, y_prob)
            except ValueError as e:
                logger.warning(f"Could not compute average precision: {e}")
                avg_precision = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report as dict
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "balanced_accuracy": float(balanced_acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "support": int(len(y_true)),
        }
        if auc_roc is not None:
            metrics["auc_roc"] = float(auc_roc)
        if avg_precision is not None:
            metrics["average_precision"] = float(avg_precision)

        logger.info(
            f"Computed metrics: accuracy={accuracy:.3f}, f1={f1:.3f}, "
            f"recall={recall:.3f}, precision={precision:.3f}"
        )
        return metrics

    def compute_per_subject_metrics(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        subject_ids: Union[np.ndarray, List[str]],
        y_prob: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metrics separately for each subject.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            subject_ids: Subject identifier for each sample.
            y_prob: Predicted probabilities (optional).

        Returns:
            Dictionary mapping subject_id -> metrics dict.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        subject_ids = np.asarray(subject_ids)

        if len(y_true) != len(subject_ids):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) vs subject_ids ({len(subject_ids)})"
            )

        unique_subjects = np.unique(subject_ids)
        subject_metrics = {}

        for subj in unique_subjects:
            mask = subject_ids == subj
            if np.sum(mask) == 0:
                continue
            subj_y_true = y_true[mask]
            subj_y_pred = y_pred[mask]
            subj_y_prob = y_prob[mask] if y_prob is not None else None

            try:
                metrics = self.compute_all_metrics(
                    subj_y_true, subj_y_pred, subj_y_prob
                )
                subject_metrics[subj] = metrics
            except Exception as e:
                logger.warning(f"Failed to compute metrics for subject {subj}: {e}")
                subject_metrics[subj] = {"error": str(e)}

        logger.info(f"Computed per‑subject metrics for {len(subject_metrics)} subjects")
        return subject_metrics

    def compute_fairness_metrics(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        sensitive_groups: Union[np.ndarray, List[str]],
    ) -> Dict[str, float]:
        """Compute fairness metrics across sensitive groups.

        Currently implemented:
            - Demographic parity difference: max difference in positive rate across groups.
            - Equalized odds difference: max difference in TPR and FPR across groups.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_groups: Group membership for each sample (e.g., gender, ethnicity).

        Returns:
            Dictionary with fairness metrics.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_groups = np.asarray(sensitive_groups)

        unique_groups = np.unique(sensitive_groups)
        group_stats = {}

        for group in unique_groups:
            mask = sensitive_groups == group
            if np.sum(mask) == 0:
                continue
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]

            # Positive rate (predicted)
            pos_rate = np.mean(group_y_pred)

            # True positive rate and false positive rate
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            group_stats[group] = {
                "positive_rate": pos_rate,
                "tpr": tpr,
                "fpr": fpr,
                "support": int(np.sum(mask)),
            }

        # Demographic parity difference
        pos_rates = [stats["positive_rate"] for stats in group_stats.values()]
        demographic_parity_diff = max(pos_rates) - min(pos_rates) if pos_rates else 0.0

        # Equalized odds differences (max TPR diff, max FPR diff)
        tprs = [stats["tpr"] for stats in group_stats.values()]
        fprs = [stats["fpr"] for stats in group_stats.values()]
        equalized_odds_tpr_diff = max(tprs) - min(tprs) if tprs else 0.0
        equalized_odds_fpr_diff = max(fprs) - min(fprs) if fprs else 0.0

        fairness = {
            "demographic_parity_difference": float(demographic_parity_diff),
            "equalized_odds_tpr_difference": float(equalized_odds_tpr_diff),
            "equalized_odds_fpr_difference": float(equalized_odds_fpr_diff),
            "group_stats": group_stats,
        }
        logger.info(
            f"Fairness metrics: demographic parity diff = {demographic_parity_diff:.3f}, "
            f"equalized odds TPR diff = {equalized_odds_tpr_diff:.3f}"
        )
        return fairness

    def bootstrap_confidence_intervals(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_prob: Optional[Union[np.ndarray, List[float]]] = None,
        n_bootstrap: Optional[int] = None,
        ci: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for metrics.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities (optional).
            n_bootstrap: Override default number of bootstrap samples.
            ci: Override default confidence level.

        Returns:
            Dictionary mapping metric name -> {'lower': ..., 'upper': ..., 'mean': ...}.
        """
        n_bootstrap = n_bootstrap or self.n_bootstrap
        ci_level = ci or self.ci_level
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_prob = np.asarray(y_prob) if y_prob is not None else None

        n_samples = len(y_true)
        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
        ]
        if y_prob is not None:
            metric_names.extend(["auc_roc", "average_precision"])

        bootstrap_metrics = {name: [] for name in metric_names}

        for i in range(n_bootstrap):
            indices = resample(
                np.arange(n_samples),
                replace=True,
                n_samples=n_samples,
                random_state=self.random_state + i,
            )
            b_y_true = y_true[indices]
            b_y_pred = y_pred[indices]
            b_y_prob = y_prob[indices] if y_prob is not None else None

            try:
                metrics = self.compute_all_metrics(b_y_true, b_y_pred, b_y_prob)
                for name in metric_names:
                    if name in metrics:
                        bootstrap_metrics[name].append(metrics[name])
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")

        # Compute confidence intervals
        ci_results = {}
        alpha = (1 - ci_level) / 2
        for name, values in bootstrap_metrics.items():
            if len(values) == 0:
                ci_results[name] = {"lower": np.nan, "upper": np.nan, "mean": np.nan}
                continue
            lower = np.percentile(values, 100 * alpha)
            upper = np.percentile(values, 100 * (1 - alpha))
            mean = np.mean(values)
            ci_results[name] = {
                "lower": float(lower),
                "upper": float(upper),
                "mean": float(mean),
            }

        logger.info(
            f"Computed bootstrap CIs for {len(ci_results)} metrics "
            f"({n_bootstrap} samples, CI={ci_level})"
        )
        return ci_results