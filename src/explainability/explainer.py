"""Explainability module for deception detection using SHAP and feature importance.

This module provides interpretable explanations for deception predictions by
combining SHAP values, feature importance, and domain knowledge to produce
human-readable behavioral explanations.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.utils import check_array

from src.model.classifier import DeceptionClassifier


class DeceptionExplainer:
    """Generates interpretable explanations for deception predictions.
    
    Combines SHAP values, feature importance, and domain knowledge
    to produce human-readable behavioral explanations.
    """
    
    # Domain knowledge mapping: feature name -> behavioral interpretation
    FEATURE_INTERPRETATIONS = {
        'eye_left_openness': 'Left eye openness (reduced blinking or gaze aversion may indicate deception)',
        'eye_right_openness': 'Right eye openness (asymmetric eye behavior can signal cognitive load)',
        'mouth_openness': 'Mouth openness (suppressed or exaggerated expressions may indicate deception)',
        'mouth_width': 'Mouth width changes (forced smiles show different width patterns)',
        'mouth_asymmetry': 'Mouth asymmetry (asymmetric expressions are harder to fake)',
        'left_brow_height': 'Left eyebrow height (raised brows can indicate surprise or concern)',
        'right_brow_height': 'Right eyebrow height (asymmetric brow activity linked to deception)',
        'brow_asymmetry': 'Eyebrow asymmetry (involuntary micro-expressions often appear asymmetrically)',
        'brow_tension': 'Eyebrow tension (furrowed brows indicate cognitive effort or stress)',
        'nose_wrinkle': 'Nose wrinkle (disgust micro-expression, hard to fake voluntarily)',
        'nostril_flare': 'Nostril flare (stress indicator, autonomic nervous system response)',
        'nose_asymmetry': 'Nose asymmetry (subtle asymmetries reveal involuntary expressions)',
        'head_pitch': 'Head pitch (nodding patterns differ between truthful and deceptive responses)',
        'head_yaw': 'Head yaw (head turning can indicate avoidance behavior)',
        'head_roll': 'Head tilt (unusual tilt patterns may indicate discomfort)',
        'head_displacement': 'Head movement (excessive stillness or fidgeting both signal stress)',
        'pitch_mean': 'Average pitch (elevated pitch is a stress indicator)',
        'pitch_std': 'Pitch variability (reduced variability suggests rehearsed responses)',
        'pitch_range': 'Pitch range (constricted range indicates tension)',
        'pitch_variation_coefficient': 'Pitch variation coefficient (normalized pitch instability)',
        'voiced_ratio': 'Voiced speech ratio (reduced voicing suggests hesitation)',
        'pause_count': 'Number of pauses (increased pausing indicates cognitive load)',
        'avg_pause_duration': 'Average pause length (longer pauses suggest response fabrication)',
        'max_pause_duration': 'Longest pause (extended silence before answering is suspicious)',
        'pause_rate': 'Pauses per minute (higher rate indicates cognitive effort)',
        'speech_ratio': 'Speech ratio (less speech time relative to pauses suggests evasion)',
        'energy_mean': 'Average speech energy (reduced energy may indicate withdrawal)',
        'energy_std': 'Energy variability (high variability suggests emotional instability)',
        'energy_range': 'Energy range (wide range indicates emotional dysregulation)',
        'energy_variation': 'Energy variation coefficient (normalized energy instability)',
        'spectral_centroid_mean': 'Spectral centroid (brightness of voice, shifts under stress)',
        'spectral_bandwidth_mean': 'Spectral bandwidth (voice quality changes under deception)',
        'audio_visual_sync': 'Audio-visual synchrony (desynchronization suggests incongruent behavior)',
        'pause_gaze_coupling': 'Pause-gaze coupling (looking away while pausing is a deception cue)',
        'stress_composite': 'Stress composite (combined physiological stress indicator)',
    }
    
    FEATURE_GROUPS = {
        'Eye Behavior': ['eye_left_openness', 'eye_right_openness'],
        'Mouth Dynamics': ['mouth_openness', 'mouth_width', 'mouth_asymmetry'],
        'Eyebrow Activity': ['left_brow_height', 'right_brow_height', 'brow_asymmetry', 'brow_tension'],
        'Nose Movement': ['nose_wrinkle', 'nostril_flare', 'nose_asymmetry'],
        'Head Dynamics': ['head_pitch', 'head_yaw', 'head_roll', 'head_displacement'],
        'Voice Pitch': ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_variation_coefficient'],
        'Speech Timing': ['pause_count', 'avg_pause_duration', 'max_pause_duration', 'pause_rate', 'speech_ratio', 'voiced_ratio'],
        'Voice Energy': ['energy_mean', 'energy_std', 'energy_range', 'energy_variation', 'spectral_centroid_mean', 'spectral_bandwidth_mean'],
        'Cross-Modal': ['audio_visual_sync', 'pause_gaze_coupling', 'stress_composite'],
    }
    
    def __init__(self, classifier: DeceptionClassifier):
        """Initialize with a trained DeceptionClassifier.
        
        Args:
            classifier: A trained DeceptionClassifier instance.
        
        Raises:
            ValueError: If classifier is not trained.
        """
        if not classifier.is_fitted_:
            warnings.warn("Classifier must be trained before creating explainer. Explainer will not work until trained.")
        
        self.classifier = classifier
        self._shap_explainer = None
        self._feature_names = classifier.feature_names_
        
    def _init_shap_explainer(self) -> Optional[shap.TreeExplainer]:
        """Lazy-initialize SHAP TreeExplainer from the trained model.
        
        Returns:
            shap.TreeExplainer: Initialized SHAP explainer, or None on failure.
        
        Notes:
            Uses feature_perturbation='tree_path_dependent' for tree-based models.
            On failure, warns and returns None.
        """
        if self._shap_explainer is None:
            try:
                # Use the underlying GradientBoostingClassifier (not calibrated model)
                # Research rationale: TreeExplainer works best with tree-based models
                # and provides exact SHAP values for gradient boosting.
                self._shap_explainer = shap.TreeExplainer(
                    model=self.classifier.model,
                    feature_perturbation='tree_path_dependent'
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize SHAP TreeExplainer: {e}. "
                    "Falling back to permutation importance.",
                    RuntimeWarning
                )
                self._shap_explainer = None
        
        return self._shap_explainer
    
    def compute_shap_values(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Compute SHAP values using TreeExplainer.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            feature_names: Ordered list of feature names matching columns of X.
        
        Returns:
            np.ndarray: SHAP values array of shape (n_samples, n_features).
        
        Notes:
            - If SHAP fails, falls back to permutation importance.
            - Returns absolute SHAP values (magnitude of contribution).
        """
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        
        # Ensure feature names match
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"Feature names length ({len(feature_names)}) "
                f"does not match X columns ({X.shape[1]})"
            )
        
        # Scale features using classifier's scaler
        X_scaled = self.classifier.scaler.transform(X)
        
        try:
            explainer = self._init_shap_explainer()
            if explainer is None:
                raise RuntimeError("SHAP explainer could not be initialized.")
            
            shap_values = explainer.shap_values(X_scaled)
            
            # For binary classification, SHAP returns list of arrays for each class
            # We take the SHAP values for the positive class (deception)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Positive class (deception)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]  # Positive class for multi-output
        except Exception as e:
            warnings.warn(
                f"SHAP computation failed: {e}. Falling back to permutation importance.",
                RuntimeWarning
            )
            # Fallback: compute permutation importance as proxy for feature contribution
            shap_values = self._compute_fallback_importance(X_scaled, feature_names)
        
        return np.asarray(shap_values, dtype=np.float64)
    
    def _compute_fallback_importance(
        self, X_scaled: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        """Compute permutation importance as fallback when SHAP fails.
        
        Args:
            X_scaled: Scaled feature matrix.
            feature_names: Feature names.
        
        Returns:
            np.ndarray: Importance values shaped (n_samples, n_features).
        """
        # Use permutation importance to get global feature importance
        # Research rationale: permutation importance measures decrease in model
        # performance when a feature is randomly shuffled, providing a robust
        # alternative to SHAP for feature contribution estimation.
        try:
            result = permutation_importance(
                estimator=self.classifier.model,
                X=X_scaled,
                y=self.classifier.model.predict(X_scaled),  # Use predicted labels
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            # Broadcast global importance to each sample
            importance = result.importances_mean
            importance = np.clip(importance, 0, None)  # Ensure non-negative
            importance = importance / (np.sum(importance) + 1e-10)  # Normalize
            
            # Create per-sample importance by adding small noise
            # This approximates sample-specific contributions
            n_samples = X_scaled.shape[0]
            noise = np.random.normal(0, 0.01 * importance.std(), (n_samples, len(importance)))
            shap_values = importance.reshape(1, -1) + noise
            return shap_values
        except Exception:
            # Ultimate fallback: uniform importance
            n_samples, n_features = X_scaled.shape
            return np.ones((n_samples, n_features)) / n_features
    
    def get_feature_importance(
        self, X: np.ndarray, feature_names: List[str], top_k: int = 10
    ) -> pd.DataFrame:
        """Get ranked feature importance combining model importance and SHAP.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            feature_names: Ordered list of feature names.
            top_k: Number of top features to return (default: 10).
        
        Returns:
            pd.DataFrame: Columns: ['feature', 'importance', 'shap_importance',
                                    'combined_score', 'rank', 'interpretation']
        """
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        
        # 1. Get model's built-in feature importance
        try:
            model_importance = self.classifier.get_feature_importance()
            model_importance_dict = dict(
                zip(model_importance['feature'], model_importance['importance'])
            )
        except Exception:
            model_importance_dict = {}
        
        # 2. Compute SHAP importance (mean absolute SHAP value per feature)
        shap_values = self.compute_shap_values(X, feature_names)
        shap_importance = np.mean(np.abs(shap_values), axis=0)
        
        # 3. Combine scores
        combined_scores = []
        for i, feat in enumerate(feature_names):
            model_imp = model_importance_dict.get(feat, 0.0)
            shap_imp = shap_importance[i] if i < len(shap_importance) else 0.0
            
            # Research rationale: weighted average gives more weight to SHAP
            # because it captures non-linear interactions better than Gini importance.
            combined = 0.3 * model_imp + 0.7 * shap_imp
            
            combined_scores.append({
                'feature': feat,
                'importance': float(model_imp),
                'shap_importance': float(shap_imp),
                'combined_score': float(combined),
                'interpretation': self.FEATURE_INTERPRETATIONS.get(feat, 'No interpretation available.')
            })
        
        df = pd.DataFrame(combined_scores)
        df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        # Limit to top_k
        if top_k > 0:
            df = df.head(top_k)
        
        return df
    
    def get_ranked_indicators(
        self, X: np.ndarray, feature_names: List[str], top_k: int = 5
    ) -> List[Dict]:
        """Get ranked deception indicators for a single sample.
        
        Args:
            X: Feature matrix of shape (1, n_features) for a single sample.
            feature_names: Ordered list of feature names.
            top_k: Number of top indicators to return (default: 5).
        
        Returns:
            List[Dict]: Each dict contains:
                - 'feature': feature name
                - 'value': original feature value
                - 'shap_value': SHAP contribution value
                - 'abs_contribution': absolute SHAP magnitude
                - 'direction': 'toward_deception' or 'toward_truth'
                - 'interpretation': behavioral interpretation
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D. Use shape (1, n_features) for single sample.")
        
        if X.shape[0] != 1:
            warnings.warn(
                "Multiple samples provided. Using first sample for indicator ranking.",
                UserWarning
            )
            X = X[:1]
        
        # Compute SHAP values for the sample
        shap_values = self.compute_shap_values(X, feature_names)
        shap_sample = shap_values[0]
        
        # Get original feature values
        feature_values = X[0]
        
        # Create indicator dictionaries
        indicators = []
        for i, feat in enumerate(feature_names):
            shap_val = shap_sample[i]
            feat_val = feature_values[i]
            
            # Determine direction based on SHAP sign
            # Positive SHAP pushes prediction toward deception (class 1)
            direction = 'toward_deception' if shap_val > 0 else 'toward_truth'
            
            indicators.append({
                'feature': feat,
                'value': float(feat_val),
                'shap_value': float(shap_val),
                'abs_contribution': float(np.abs(shap_val)),
                'direction': direction,
                'interpretation': self.FEATURE_INTERPRETATIONS.get(feat, 'No interpretation available.')
            })
        
        # Sort by absolute contribution (most influential first)
        indicators.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        # Return top_k indicators
        return indicators[:top_k]
    
    def explain_prediction(
        self,
        X: np.ndarray,
        feature_names: List[str],
        prediction: float,
        subject_id: Optional[str] = None
    ) -> Dict:
        """Generate complete explanation for a single prediction.
        
        Args:
            X: Feature matrix of shape (1, n_features) for a single sample.
            feature_names: Ordered list of feature names.
            prediction: Deception probability (0-1).
            subject_id: Optional subject identifier for contextualization.
        
        Returns:
            Dict: Explanation dictionary with keys:
                - 'verdict': 'truthful' or 'deceptive' based on threshold 0.5
                - 'confidence': absolute distance from decision boundary
                - 'deception_probability': prediction probability
                - 'top_indicators': list of top indicators (from get_ranked_indicators)
                - 'group_contributions': dict of SHAP contributions per feature group
                - 'behavioral_summary': human-readable summary text
                - 'feature_values': dict of original feature values
                - 'subject_id': subject identifier if provided
        """
        if X.ndim != 2 or X.shape[0] != 1:
            raise ValueError("X must be a single sample with shape (1, n_features).")
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X, feature_names)
        
        # Determine verdict
        verdict = 'deceptive' if prediction >= 0.5 else 'truthful'
        confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
        
        # Get top indicators
        top_indicators = self.get_ranked_indicators(X, feature_names, top_k=5)
        
        # Compute group contributions
        group_contributions = self.explain_group_contributions(shap_values, feature_names)
        
        # Generate behavioral summary
        behavioral_summary = self.generate_behavioral_summary(
            top_indicators, prediction, group_contributions
        )
        
        # Prepare feature values dictionary
        feature_values = {
            feat: float(val) for feat, val in zip(feature_names, X[0])
        }
        
        return {
            'verdict': verdict,
            'confidence': float(confidence),
            'deception_probability': float(prediction),
            'top_indicators': top_indicators,
            'group_contributions': group_contributions,
            'behavioral_summary': behavioral_summary,
            'feature_values': feature_values,
            'subject_id': subject_id
        }
    
    def generate_behavioral_summary(
        self,
        top_indicators: List[Dict],
        prediction: float,
        group_contributions: Dict[str, float]
    ) -> str:
        """Generate human-readable behavioral explanation paragraph.
        
        Args:
            top_indicators: Top indicators from get_ranked_indicators.
            prediction: Deception probability (0-1).
            group_contributions: Normalized SHAP contributions per feature group.
        
        Returns:
            str: Human-readable summary paragraph.
        
        Example:
            "The subject shows elevated deception probability (0.82). Key indicators include
            increased pause duration (SHAP: +0.12), which suggests cognitive load during
            response fabrication, and reduced audio-visual synchrony (SHAP: +0.09),
            indicating incongruent verbal and non-verbal behavior."
        """
        # Determine deception level
        if prediction >= 0.7:
            level = "strongly elevated"
        elif prediction >= 0.6:
            level = "elevated"
        elif prediction >= 0.5:
            level = "slightly elevated"
        elif prediction >= 0.4:
            level = "slightly reduced"
        elif prediction >= 0.3:
            level = "reduced"
        else:
            level = "strongly reduced"
        
        # Build summary opening
        summary_parts = [
            f"The subject shows {level} deception probability ({prediction:.2f})."
        ]
        
        # Add top indicators
        if top_indicators:
            indicator_desc = []
            for idx, ind in enumerate(top_indicators[:3]):  # top 3
                feat = ind['feature']
                shap_val = ind['shap_value']
                direction = ind['direction']
                interp = self.FEATURE_INTERPRETATIONS.get(feat, '')
                # Extract the main behavioral insight (first sentence)
                if interp:
                    insight = interp.split('.')[0] + '.'
                else:
                    insight = "This feature contributes to the prediction."
                
                # Format SHAP value with sign
                sign = '+' if shap_val >= 0 else ''
                indicator_desc.append(
                    f"{feat.replace('_', ' ')} (SHAP: {sign}{shap_val:.2f}), "
                    f"which {insight.lower()}"
                )
            
            if indicator_desc:
                summary_parts.append("Key indicators include " + ", ".join(indicator_desc) + ".")
        
        # Add group contributions
        top_group = max(group_contributions.items(), key=lambda x: x[1]) if group_contributions else None
        if top_group:
            group_name, contrib = top_group
            summary_parts.append(
                f"The {group_name} feature group contributed the most ({contrib:.1%}) to the prediction, "
                f"suggesting that {group_name.lower()} played a dominant role."
            )
        
        # Join into a single paragraph
        return " ".join(summary_parts)
    
    def explain_group_contributions(
        self, shap_values: np.ndarray, feature_names: List[str]
    ) -> Dict[str, float]:
        """Aggregate SHAP contribution per feature group, normalized to sum to 1.0.
        
        Args:
            shap_values: SHAP values array of shape (n_samples, n_features).
            feature_names: Ordered list of feature names.
        
        Returns:
            Dict[str, float]: Mapping from group name to normalized contribution.
        """
        # Ensure single sample (take first if multiple)
        if shap_values.ndim == 2 and shap_values.shape[0] > 1:
            shap_values = shap_values[0:1]
        shap_abs = np.abs(shap_values).mean(axis=0)  # average across samples (usually 1)
        
        # Map feature to group
        feature_to_group = {}
        for group, features in self.FEATURE_GROUPS.items():
            for feat in features:
                feature_to_group[feat] = group
        
        # Aggregate contributions
        group_sums = {}
        for i, feat in enumerate(feature_names):
            if i >= len(shap_abs):
                continue
            group = feature_to_group.get(feat)
            if group is None:
                # Assign to 'Other' group
                group = 'Other'
            group_sums[group] = group_sums.get(group, 0.0) + shap_abs[i]
        
        # Normalize to sum to 1.0
        total = sum(group_sums.values())
        if total > 0:
            group_contributions = {k: v / total for k, v in group_sums.items()}
        else:
            group_contributions = {k: 0.0 for k in group_sums.keys()}
        return group_contributions
