"""Cross‑validation strategies for deception detection.

This module provides subject‑aware and stratified cross‑validation
splits suitable for multimodal deception analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    LeaveOneGroupOut,
    cross_val_predict,
)
from sklearn.base import BaseEstimator

from .metrics import DeceptionMetrics


logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross‑validation with stratification and subject‑aware splits.

    This class provides methods for performing k‑fold cross‑validation
    while respecting subject boundaries (no data leakage) and maintaining
    class balance.

    Attributes:
        random_state: Random seed for reproducibility.
        n_splits: Default number of folds.
        metrics: Instance of DeceptionMetrics for evaluation.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_splits: int = 5,
        metrics: Optional[DeceptionMetrics] = None,
    ) -> None:
        """Initialize cross‑validator.

        Args:
            random_state: Random seed for reproducibility.
            n_splits: Default number of folds.
            metrics: Optional DeceptionMetrics instance. If None, a default is created.
        """
        self.random_state = random_state
        self.n_splits = n_splits
        self.metrics = metrics or DeceptionMetrics(random_state=random_state)

    def stratified_k_fold(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[int]],
        n_splits: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified k‑fold indices.

        Args:
            X: Feature matrix (any shape, only length matters).
            y: Target labels used for stratification.
            n_splits: Override default number of folds.
            random_state: Override default random seed.

        Returns:
            List of (train_indices, test_indices) for each fold.
        """
        n_splits = n_splits or self.n_splits
        random_state = random_state or self.random_state

        y = np.asarray(y)
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = []
        for train_idx, test_idx in skf.split(X, y):
            splits.append((train_idx, test_idx))

        logger.info(
            f"Generated {n_splits} stratified folds "
            f"(random_state={random_state}, n_samples={len(y)})"
        )
        return splits

    def subject_aware_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[int]],
        subject_ids: Union[np.ndarray, List[str]],
        n_splits: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate k‑fold splits where the same subject does not appear in both train and test.

        Uses GroupKFold with subject_ids as groups.

        Args:
            X: Feature matrix.
            y: Target labels (unused for grouping, kept for compatibility).
            subject_ids: Subject identifier for each sample.
            n_splits: Override default number of folds.

        Returns:
            List of (train_indices, test_indices) for each fold.
        """
        n_splits = n_splits or self.n_splits
        subject_ids = np.asarray(subject_ids)

        # Ensure we have enough unique groups
        unique_subjects = np.unique(subject_ids)
        if len(unique_subjects) < n_splits:
            logger.warning(
                f"Only {len(unique_subjects)} unique subjects, reducing n_splits to {len(unique_subjects)}"
            )
            n_splits = len(unique_subjects)

        gkf = GroupKFold(n_splits=n_splits)
        splits = []
        for train_idx, test_idx in gkf.split(X, y, groups=subject_ids):
            splits.append((train_idx, test_idx))

        logger.info(
            f"Generated {n_splits} subject‑aware folds "
            f"(unique subjects={len(unique_subjects)})"
        )
        return splits

    def leave_one_subject_out(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[int]],
        subject_ids: Union[np.ndarray, List[str]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Leave‑One‑Subject‑Out (LOSO) cross‑validation generator.

        Each fold uses all subjects except one as training, and the left‑out
        subject as test.

        Args:
            X: Feature matrix.
            y: Target labels.
            subject_ids: Subject identifier for each sample.

        Returns:
            List of (train_indices, test_indices) for each fold.
        """
        subject_ids = np.asarray(subject_ids)
        unique_subjects = np.unique(subject_ids)

        splits = []
        for subj in unique_subjects:
            test_mask = subject_ids == subj
            train_mask = ~test_mask
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            splits.append((train_idx, test_idx))

        logger.info(
            f"Generated {len(splits)} LOSO folds (unique subjects={len(unique_subjects)})"
        )
        return splits

    def run_cross_validation(
        self,
        classifier: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[int]],
        n_splits: Optional[int] = None,
        subject_ids: Optional[Union[np.ndarray, List[str]]] = None,
        method: str = "stratified",
        return_predictions: bool = True,
    ) -> Dict[str, Any]:
        """Run cross‑validation and compute metrics across folds.

        Args:
            classifier: A scikit‑learn compatible classifier.
            X: Feature matrix.
            y: Target labels.
            n_splits: Override default number of folds.
            subject_ids: Required for 'subject_aware' and 'loso' methods.
            method: One of 'stratified', 'subject_aware', 'loso'.
            return_predictions: If True, aggregate predictions across folds.

        Returns:
            Dictionary with:
                - mean_metrics: dict of mean metric values across folds
                - std_metrics: dict of standard deviations
                - per_fold_metrics: list of metric dicts for each fold
                - aggregated_predictions: dict with y_true, y_pred, y_prob (if return_predictions)
                - fold_indices: list of (train_idx, test_idx) for each fold
        """
        n_splits = n_splits or self.n_splits
        y = np.asarray(y)

        # Generate fold indices
        if method == "stratified":
            folds = self.stratified_k_fold(X, y, n_splits=n_splits)
        elif method == "subject_aware":
            if subject_ids is None:
                raise ValueError("subject_ids required for subject_aware method")
            folds = self.subject_aware_split(X, y, subject_ids, n_splits=n_splits)
        elif method == "loso":
            if subject_ids is None:
                raise ValueError("subject_ids required for loso method")
            folds = self.leave_one_subject_out(X, y, subject_ids)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'stratified', 'subject_aware', 'loso'."
            )

        per_fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        fold_indices = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            logger.info(f"Fold {fold_idx+1}/{len(folds)}: train={len(train_idx)}, test={len(test_idx)}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train classifier
            classifier.fit(X_train, y_train)

            # Predict
            y_pred = classifier.predict(X_test)
            y_prob = (
                classifier.predict_proba(X_test)[:, 1]
                if hasattr(classifier, "predict_proba")
                else None
            )

            # Compute metrics
            metrics = self.metrics.compute_all_metrics(y_test, y_pred, y_prob)
            per_fold_metrics.append(metrics)

            # Aggregate for overall analysis
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            if y_prob is not None:
                all_y_prob.extend(y_prob.tolist())

            fold_indices.append((train_idx.tolist(), test_idx.tolist()))

        # Compute mean and std across folds
        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
        ]
        if any("auc_roc" in m for m in per_fold_metrics):
            metric_names.append("auc_roc")
        if any("average_precision" in m for m in per_fold_metrics):
            metric_names.append("average_precision")

        mean_metrics = {}
        std_metrics = {}
        for name in metric_names:
            values = []
            for fold_metrics in per_fold_metrics:
                if name in fold_metrics:
                    values.append(fold_metrics[name])
            if values:
                mean_metrics[name] = float(np.mean(values))
                std_metrics[name] = float(np.std(values))
            else:
                mean_metrics[name] = np.nan
                std_metrics[name] = np.nan

        # Aggregate predictions
        aggregated = {}
        if return_predictions:
            aggregated["y_true"] = all_y_true
            aggregated["y_pred"] = all_y_pred
            if all_y_prob:
                aggregated["y_prob"] = all_y_prob

        result = {
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "per_fold_metrics": per_fold_metrics,
            "aggregated_predictions": aggregated,
            "fold_indices": fold_indices,
            "method": method,
            "n_splits": len(folds),
        }

        logger.info(
            f"Cross‑validation completed ({method}, {len(folds)} folds). "
            f"Mean accuracy = {mean_metrics.get('accuracy', 0):.3f} ± {std_metrics.get('accuracy', 0):.3f}"
        )
        return result