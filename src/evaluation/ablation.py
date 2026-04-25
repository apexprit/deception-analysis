"""Ablation study for feature group importance in deception detection.

This module provides systematic removal of feature groups (facial, audio,
temporal, cross‑modal) to quantify their contribution to model performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from .cross_validation import CrossValidator
from .metrics import DeceptionMetrics


logger = logging.getLogger(__name__)


class AblationStudy:
    """Conduct ablation studies by removing feature groups.

    This class evaluates the contribution of different feature groups
    (facial, audio, temporal, cross‑modal) to overall model performance.

    Attributes:
        feature_names: List of all feature names in the dataset.
        feature_groups: Dictionary mapping group names to lists of feature indices or names.
        random_state: Random seed for reproducibility.
        n_splits: Number of cross‑validation folds.
        metrics: Instance of DeceptionMetrics for evaluation.
    """

    def __init__(
        self,
        feature_names: List[str],
        feature_groups: Dict[str, List[str]],
        random_state: int = 42,
        n_splits: int = 5,
        metrics: Optional[DeceptionMetrics] = None,
    ) -> None:
        """Initialize ablation study.

        Args:
            feature_names: List of all feature names in the dataset.
            feature_groups: Dictionary mapping group names to lists of feature names.
                Expected keys: 'facial', 'audio', 'temporal', 'cross_modal'.
            random_state: Random seed for reproducibility.
            n_splits: Number of cross‑validation folds.
            metrics: Optional DeceptionMetrics instance.
        """
        self.feature_names = feature_names
        self.feature_groups = feature_groups
        self.random_state = random_state
        self.n_splits = n_splits
        self.metrics = metrics or DeceptionMetrics(random_state=random_state)

        # Validate that all group features are in feature_names
        for group_name, features in feature_groups.items():
            for f in features:
                if f not in feature_names:
                    logger.warning(
                        f"Feature '{f}' from group '{group_name}' not found in feature_names"
                    )

        # Build index mapping
        self.feature_to_index = {name: i for i, name in enumerate(feature_names)}
        self.group_indices = {}
        for group_name, features in feature_groups.items():
            indices = []
            for f in features:
                if f in self.feature_to_index:
                    indices.append(self.feature_to_index[f])
            self.group_indices[group_name] = indices

    def _select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        include_groups: Optional[List[str]] = None,
        exclude_groups: Optional[List[str]] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Select features based on included/excluded groups.

        Args:
            X: Original feature matrix (n_samples × n_features).
            include_groups: List of group names to keep (all others removed).
            exclude_groups: List of group names to remove (all others kept).

        Returns:
            Feature matrix with selected columns.
        """
        if include_groups is not None and exclude_groups is not None:
            raise ValueError("Cannot specify both include_groups and exclude_groups")

        if isinstance(X, pd.DataFrame):
            columns = X.columns.tolist()
        else:
            columns = list(range(X.shape[1]))

        # Determine which indices to keep
        if include_groups is not None:
            keep_indices = set()
            for group in include_groups:
                keep_indices.update(self.group_indices.get(group, []))
        elif exclude_groups is not None:
            all_indices = set(range(len(self.feature_names)))
            remove_indices = set()
            for group in exclude_groups:
                remove_indices.update(self.group_indices.get(group, []))
            keep_indices = all_indices - remove_indices
        else:
            # Keep all features
            keep_indices = set(range(len(self.feature_names)))

        # Convert to sorted list
        keep_indices = sorted(keep_indices)

        if isinstance(X, pd.DataFrame):
            selected_cols = [columns[i] for i in keep_indices]
            return X[selected_cols]
        else:
            return X[:, keep_indices]

    def run_ablation(
        self,
        classifier_class: type,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List[int]],
        feature_names: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        n_splits: Optional[int] = None,
        subject_ids: Optional[Union[np.ndarray, List[str]]] = None,
        cv_method: str = "stratified",
    ) -> Dict[str, Dict[str, Any]]:
        """Run ablation study across feature groups.

        Args:
            classifier_class: A scikit‑learn classifier class (not instance).
            X: Feature matrix.
            y: Target labels.
            feature_names: Override stored feature names.
            feature_groups: Override stored feature groups.
            n_splits: Override default number of folds.
            subject_ids: Optional subject IDs for subject‑aware CV.
            cv_method: Cross‑validation method ('stratified', 'subject_aware', 'loso').

        Returns:
            Dictionary with keys:
                - full_model: metrics with all features
                - minus_facial: metrics without facial features
                - minus_audio: metrics without audio features
                - minus_temporal: metrics without temporal features
                - minus_cross_modal: metrics without cross‑modal features
                - facial_only: metrics with only facial features
                - audio_only: metrics with only audio features
                - temporal_only: metrics with only temporal features
            Each value is a dict with 'mean_metrics', 'std_metrics', 'per_fold_metrics'.
        """
        if feature_names is not None:
            self.feature_names = feature_names
        if feature_groups is not None:
            self.feature_groups = feature_groups
            self.group_indices = {}
            for group_name, features in feature_groups.items():
                indices = []
                for f in features:
                    if f in self.feature_to_index:
                        indices.append(self.feature_to_index[f])
                self.group_indices[group_name] = indices

        n_splits = n_splits or self.n_splits
        cv = CrossValidator(
            random_state=self.random_state,
            n_splits=n_splits,
            metrics=self.metrics,
        )

        # Define ablation conditions
        conditions = {
            "full_model": {"include_groups": None, "exclude_groups": None},
            "minus_facial": {"exclude_groups": ["facial"]},
            "minus_audio": {"exclude_groups": ["audio"]},
            "minus_temporal": {"exclude_groups": ["temporal"]},
            "minus_cross_modal": {"exclude_groups": ["cross_modal"]},
            "facial_only": {"include_groups": ["facial"]},
            "audio_only": {"include_groups": ["audio"]},
            "temporal_only": {"include_groups": ["temporal"]},
        }

        results = {}

        for cond_name, cond_args in conditions.items():
            logger.info(f"Running ablation condition: {cond_name}")

            # Select features
            X_cond = self._select_features(
                X,
                include_groups=cond_args.get("include_groups"),
                exclude_groups=cond_args.get("exclude_groups"),
            )

            # Skip if no features remain
            if X_cond.shape[1] == 0:
                logger.warning(f"No features left for condition {cond_name}, skipping")
                results[cond_name] = {
                    "mean_metrics": {},
                    "std_metrics": {},
                    "per_fold_metrics": [],
                }
                continue

            # Run cross‑validation
            classifier = classifier_class()
            cv_result = cv.run_cross_validation(
                classifier,
                X_cond,
                y,
                n_splits=n_splits,
                subject_ids=subject_ids,
                method=cv_method,
                return_predictions=False,
            )

            results[cond_name] = {
                "mean_metrics": cv_result["mean_metrics"],
                "std_metrics": cv_result["std_metrics"],
                "per_fold_metrics": cv_result["per_fold_metrics"],
                "n_features": X_cond.shape[1],
            }

        logger.info("Ablation study completed.")
        return results

    def compute_feature_importance_ranking(
        self,
        classifier: BaseEstimator,
        feature_names: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Compute feature importance ranking from a trained classifier.

        Supports classifiers with `feature_importances_` attribute (tree‑based)
        or `coef_` attribute (linear models).

        Args:
            classifier: Trained scikit‑learn classifier.
            feature_names: Override stored feature names.

        Returns:
            List of (feature_name, importance_score) sorted descending.
        """
        if feature_names is None:
            feature_names = self.feature_names

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            # For linear models, take absolute coefficients averaged across classes
            coef = classifier.coef_
            if coef.ndim == 2:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        else:
            logger.warning(
                "Classifier has no feature_importances_ or coef_; returning uniform importances"
            )
            importances = np.ones(len(feature_names))

        # Ensure length matches
        if len(importances) != len(feature_names):
            logger.error(
                f"Importance length ({len(importances)}) != feature count ({len(feature_names)})"
            )
            # Truncate or pad
            if len(importances) > len(feature_names):
                importances = importances[: len(feature_names)]
            else:
                importances = np.pad(
                    importances,
                    (0, len(feature_names) - len(importances)),
                    mode="constant",
                    constant_values=0.0,
                )

        ranked = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info(f"Computed feature importance ranking for {len(ranked)} features")
        return ranked