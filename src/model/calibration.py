"""Subject-adaptive calibration utilities for deception detection.

This module implements personal calibration to account for inter-subject
behavioral variability, a major confound in affective/deception modeling.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.model.classifier import DeceptionClassifier
from src.utils.config import AppConfig, get_default_config
from src.utils.io import ensure_dir


class SubjectCalibrator:
    """Subject-specific calibration for deception detection.

    Adapts a global model to individual subjects by learning personal baselines
    and applying feature-space plus score-space adjustments.
    """

    def __init__(self, base_classifier: DeceptionClassifier, config: Optional[AppConfig] = None):
        """Initialize calibrator.

        Args:
            base_classifier: Trained global classifier.
            config: Optional app-level configuration.
        """

        self.base_classifier = base_classifier
        self.config: AppConfig = config or get_default_config()
        self.subject_profiles: Dict[str, Dict[str, float]] = {}

    def compute_baseline(self, subject_features: pd.DataFrame, subject_id: str) -> Dict[str, float]:
        """Compute a subject's baseline behavior from truthful samples.

        Args:
            subject_features: DataFrame containing subject feature rows.
            subject_id: Subject identifier.

        Returns:
            Dict[str, float]: Baseline profile with mean/std per feature.
        """

        if subject_features is None or subject_features.empty:
            return {"subject_id": subject_id, "n_samples": 0.0}

        X, feature_names = self.base_classifier.prepare_features(subject_features)
        if X.size == 0 or len(feature_names) == 0:
            return {"subject_id": subject_id, "n_samples": float(len(subject_features))}

        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)

        profile: Dict[str, float] = {
            "subject_id": subject_id,  # type: ignore[assignment]
            "n_samples": float(X.shape[0]),
        }
        for i, name in enumerate(feature_names):
            profile[f"{name}_mean"] = float(means[i])
            profile[f"{name}_std"] = float(max(stds[i], 1e-8))

        # Baseline deception tendency is used for threshold adaptation.
        if self.base_classifier.is_fitted_:
            scores = self.base_classifier.predict_deception_score(X)
            profile["baseline_deception_mean"] = float(np.mean(scores))
            profile["baseline_deception_std"] = float(max(np.std(scores), 1e-8))
        else:
            profile["baseline_deception_mean"] = 0.5
            profile["baseline_deception_std"] = 0.1

        return profile

    def calibrate_features(
        self,
        features: np.ndarray,
        subject_id: str,
        feature_names: List[str],
    ) -> np.ndarray:
        """Apply subject-specific z-score normalization.

        Args:
            features: Raw feature array.
            subject_id: Subject identifier.
            feature_names: Ordered feature names.

        Returns:
            np.ndarray: Calibrated feature array.
        """

        X = np.asarray(features, dtype=np.float64)
        if X.size == 0:
            return X

        profile = self.subject_profiles.get(subject_id)
        if profile is None:
            return X

        calibrated = X.copy()
        for i, name in enumerate(feature_names):
            mean_key = f"{name}_mean"
            std_key = f"{name}_std"
            if mean_key in profile and std_key in profile:
                mu = float(profile[mean_key])
                sigma = float(profile[std_key])
                calibrated[:, i] = (calibrated[:, i] - mu) / (sigma + 1e-8)

        return calibrated

    def calibrate_prediction(self, raw_proba: float, subject_id: str) -> float:
        """Apply subject-specific logistic threshold adjustment.

        Args:
            raw_proba: Raw deception probability from global model.
            subject_id: Subject identifier.

        Returns:
            float: Calibrated deception probability in [0, 1].
        """

        p = float(np.clip(raw_proba, 1e-6, 1.0 - 1e-6))
        profile = self.subject_profiles.get(subject_id)
        if profile is None:
            return p

        baseline_mean = float(profile.get("baseline_deception_mean", 0.5))
        baseline_std = float(profile.get("baseline_deception_std", 0.1))
        n_samples = float(profile.get("n_samples", 0.0))

        # Research rationale: move the decision surface in logit space, where
        # additive shifts correspond to multiplicative odds adjustment.
        confidence = float(np.clip(np.log1p(n_samples) / np.log1p(200.0), 0.0, 1.0))
        shift_strength = confidence * np.clip((baseline_mean - 0.5) / (baseline_std + 1e-8), -2.0, 2.0)
        shift_strength *= 0.15

        logit = np.log(p / (1.0 - p))
        adjusted = logit - shift_strength
        calibrated = 1.0 / (1.0 + np.exp(-adjusted))
        return float(np.clip(calibrated, 0.0, 1.0))

    def add_subject_profile(self, subject_id: str, truthful_features: pd.DataFrame) -> None:
        """Register a subject profile using truthful behavior samples.

        Args:
            subject_id: Subject identifier.
            truthful_features: Truthful-only feature dataframe for the subject.
        """

        profile = self.compute_baseline(truthful_features, subject_id)
        self.subject_profiles[subject_id] = profile

    def save_profiles(self, path: str) -> None:
        """Save all subject profiles to a JSON file.

        Args:
            path: Output JSON path.
        """

        out_path = Path(path)
        ensure_dir(out_path.parent)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.subject_profiles, f, indent=2, sort_keys=True)

    def load_profiles(self, path: str) -> None:
        """Load subject profiles from JSON file.

        Args:
            path: Input JSON path.
        """

        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Subject profile file not found: {in_path}")
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid subject profile format. Expected JSON object.")
        self.subject_profiles = data

    def get_subject_ids(self) -> List[str]:
        """Return registered subject identifiers."""

        return sorted(self.subject_profiles.keys())

    def calibrate_subject_model(
        self,
        subject_id: str,
        subject_features: np.ndarray,
        subject_labels: np.ndarray,
        feature_names: List[str],
    ) -> DeceptionClassifier:
        """Fine-tune a personal model for a specific subject.

        Hard examples (where base model disagrees) are upweighted to specialize
        the model for subject-specific decision boundaries.

        Args:
            subject_id: Subject identifier.
            subject_features: Subject feature matrix.
            subject_labels: Subject labels.
            feature_names: Feature names used by the model.

        Returns:
            DeceptionClassifier: Subject-adapted classifier.
        """

        X = np.asarray(subject_features, dtype=np.float64)
        y = np.asarray(subject_labels).astype(int)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("Invalid subject data shapes for calibration.")
        if len(y) == 0:
            raise ValueError("Cannot calibrate subject model with zero samples.")

        personal = DeceptionClassifier(config=self.base_classifier.config)

        # Transfer model state when available to keep inductive bias from global data.
        if self.base_classifier.is_fitted_:
            try:
                personal.scaler = clone(self.base_classifier.scaler)
                personal.scaler.mean_ = np.copy(self.base_classifier.scaler.mean_)
                personal.scaler.scale_ = np.copy(self.base_classifier.scaler.scale_)
                personal.scaler.var_ = np.copy(self.base_classifier.scaler.var_)
                personal.scaler.n_features_in_ = self.base_classifier.scaler.n_features_in_
                if hasattr(self.base_classifier.scaler, "n_samples_seen_"):
                    personal.scaler.n_samples_seen_ = self.base_classifier.scaler.n_samples_seen_
            except Exception:
                # Fallback to refitting scaler below.
                pass

        # Subject-space calibration (if profile exists) before fine-tuning.
        X_cal = self.calibrate_features(X, subject_id, feature_names)

        # Derive sample weights emphasizing hard examples.
        weights = np.ones(len(y), dtype=np.float64)
        if self.base_classifier.is_fitted_:
            base_pred = self.base_classifier.predict(X)
            disagreement = (base_pred != y).astype(np.float64)
            weights += 2.0 * disagreement

        X_scaled = personal.scaler.fit_transform(X_cal)
        personal.model.fit(X_scaled, y, sample_weight=weights)

        # Calibrator requires both classes and enough support.
        unique = np.unique(y)
        if len(unique) > 1 and len(y) >= 10:
            personal.calibrated_model.fit(X_scaled, y, sample_weight=weights)

        personal.feature_names_ = list(feature_names)
        personal.is_fitted_ = True
        personal.training_metadata_ = {
            "subject_id": subject_id,
            "n_samples": float(len(y)),
            "class_distribution": {int(c): int((y == c).sum()) for c in unique},
            "config": asdict(personal.config),
        }
        return personal

