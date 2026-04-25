"""Core Gradient Boosting classifier for deception detection.

This module implements a robust, production-oriented classifier wrapper that
standardizes features, handles missing values, performs probability calibration,
and persists all metadata required for reproducible inference.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import ModelConfig
from src.utils.io import load_model as io_load_model
from src.utils.io import save_model as io_save_model


class DeceptionClassifier:
    """Gradient Boosting-based deception classifier with probability calibration.

    Uses `sklearn` GradientBoostingClassifier as the base estimator and wraps it
    with calibration for better probabilistic interpretation.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize classifier with model config parameters.

        Args:
            config: Optional model configuration. If not provided, defaults are used.
        """

        self.config: ModelConfig = config or ModelConfig()
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )
        self.scaler = StandardScaler()

        calibration_method = getattr(self.config, "calibration_method", "sigmoid")
        self.calibrated_model = CalibratedClassifierCV(
            estimator=GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                subsample=self.config.subsample,
                random_state=self.config.random_state,
            ),
            method=calibration_method,
            cv=min(max(2, getattr(self.config, "cross_validation_folds", 5)), 5),
        )

        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_: bool = False
        self.training_metadata_: Dict[str, Any] = {}

    def prepare_features(
        self,
        features_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from a feature dataframe.

        Args:
            features_df: DataFrame with extracted features.
            feature_columns: Optional explicit feature column list.

        Returns:
            Tuple[np.ndarray, List[str]]: Feature matrix and ordered feature names.
        """

        if features_df is None or features_df.empty:
            return np.empty((0, 0), dtype=np.float64), []

        excluded_columns = {
            "frame_idx",
            "timestamp",
            "segment_idx",
            "start_time",
            "end_time",
            "label",
            "subject_id",
        }

        if feature_columns is None:
            candidate_cols = [c for c in features_df.columns if c not in excluded_columns]
            # Research rationale: constraining to numeric-only columns prevents accidental
            # leakage of identifiers and ensures stable scaling behavior.
            feature_columns = [
                c for c in candidate_cols if pd.api.types.is_numeric_dtype(features_df[c])
            ]

        if not feature_columns:
            return np.empty((len(features_df), 0), dtype=np.float64), []

        selected = features_df[feature_columns].copy()
        for col in selected.columns:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")

        selected = selected.replace([np.inf, -np.inf], np.nan)
        medians = selected.median(numeric_only=True)
        selected = selected.fillna(medians)
        selected = selected.fillna(0.0)

        # Clip to robust quantiles to reduce effect of extreme outliers.
        lower = selected.quantile(0.01)
        upper = selected.quantile(0.99)
        selected = selected.clip(lower=lower, upper=upper, axis=1)

        return selected.to_numpy(dtype=np.float64), list(selected.columns)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Train the gradient boosting classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary labels (0=truthful, 1=deceptive).
            feature_names: Ordered feature names.

        Returns:
            Dict[str, float]: Training metrics and dataset summary.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if len(y) == 0:
            raise ValueError("Cannot train with zero samples.")

        self.feature_names_ = list(feature_names)

        X_scaled = self.scaler.fit_transform(X)

        unique_classes, counts = np.unique(y, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique_classes, counts)}

        # Research rationale: inverse-frequency weighting mitigates skewed priors and
        # helps gradient boosting focus on under-represented deceptive/truthful samples.
        class_weights = {
            cls: len(y) / (len(unique_classes) * count)
            for cls, count in zip(unique_classes, counts)
            if count > 0
        }
        sample_weight = np.array([class_weights.get(label, 1.0) for label in y], dtype=np.float64)

        self.model.fit(X_scaled, y, sample_weight=sample_weight)

        can_calibrate = len(unique_classes) > 1 and len(y) >= max(10, len(unique_classes) * 3)
        if can_calibrate:
            self.calibrated_model.fit(X_scaled, y, sample_weight=sample_weight)

        train_preds = self.predict(X)
        train_probs = self.predict_proba(X)

        metrics: Dict[str, float] = {
            "train_accuracy": float(accuracy_score(y, train_preds)),
            "n_samples": float(len(y)),
            "n_features": float(X.shape[1]),
            "class_distribution": class_distribution,  # type: ignore[assignment]
        }

        try:
            metrics["train_log_loss"] = float(log_loss(y, train_probs, labels=[0, 1]))
        except ValueError:
            metrics["train_log_loss"] = float("nan")

        if len(unique_classes) > 1:
            try:
                metrics["train_auc"] = float(roc_auc_score(y, train_probs[:, 1]))
            except ValueError:
                metrics["train_auc"] = float("nan")
        else:
            metrics["train_auc"] = float("nan")

        cv_folds = min(getattr(self.config, "cross_validation_folds", 5), len(y))
        if cv_folds >= 2 and len(unique_classes) > 1:
            pipeline = make_pipeline(
                StandardScaler(),
                GradientBoostingClassifier(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    subsample=self.config.subsample,
                    random_state=self.config.random_state,
                ),
            )
            cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring="accuracy")
            metrics["cv_accuracy_mean"] = float(np.mean(cv_scores))
            metrics["cv_accuracy_std"] = float(np.std(cv_scores))
        else:
            metrics["cv_accuracy_mean"] = float("nan")
            metrics["cv_accuracy_std"] = float("nan")

        self.training_metadata_ = metrics
        self.is_fitted_ = True
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary deception labels for input features."""

        if not self.is_fitted_:
            raise ValueError("Model is not trained. Call train() first.")

        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float64))
        if hasattr(self.calibrated_model, "calibrated_classifiers_"):
            return self.calibrated_model.predict(X_scaled)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict deception probabilities for input features.

        Returns:
            np.ndarray: Probability matrix [truthful_prob, deceptive_prob].
        """

        if not self.is_fitted_:
            raise ValueError("Model is not trained. Call train() first.")

        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float64))
        if hasattr(self.calibrated_model, "calibrated_classifiers_"):
            return self.calibrated_model.predict_proba(X_scaled)
        return self.model.predict_proba(X_scaled)

    def predict_deception_score(self, X: np.ndarray) -> np.ndarray:
        """Predict scalar deception score in [0, 1] for each sample."""

        proba = self.predict_proba(X)
        return proba[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Get sorted feature importance values from the trained base model."""

        if not self.is_fitted_:
            raise ValueError("Model is not trained. Call train() first.")

        if self.feature_names_ is None:
            raise ValueError("Feature names are unavailable.")

        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            raise ValueError("Model does not expose feature importances.")

        importance_df = pd.DataFrame(
            {"feature": self.feature_names_, "importance": np.asarray(importances, dtype=np.float64)}
        )
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str) -> None:
        """Save model, scaler, calibration wrapper, and metadata using joblib."""

        payload = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names_,
            "is_fitted": self.is_fitted_,
            "config": asdict(self.config),
            "training_metadata": self.training_metadata_,
        }
        io_save_model(payload, path)

    def load(self, path: str) -> None:
        """Load model, scaler, calibration wrapper, and metadata from file."""

        payload = io_load_model(path)
        self.model = payload["model"]
        self.calibrated_model = payload.get("calibrated_model", self.calibrated_model)
        self.scaler = payload["scaler"]
        self.feature_names_ = payload.get("feature_names")
        self.is_fitted_ = bool(payload.get("is_fitted", True))
        self.training_metadata_ = dict(payload.get("training_metadata", {}))

        config_dict = payload.get("config", {})
        if isinstance(config_dict, dict):
            self.config = ModelConfig(**config_dict)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and configuration details."""

        return {
            "is_fitted": self.is_fitted_,
            "n_features": 0 if self.feature_names_ is None else len(self.feature_names_),
            "feature_names": self.feature_names_ or [],
            "config": asdict(self.config),
            "training_metadata": self.training_metadata_,
        }

