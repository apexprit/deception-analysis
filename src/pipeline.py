"""Main pipeline for multimodal deception analysis."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from src.audio.extractor import AudioFeatureExtractor
from src.audio.fusion import MultimodalFusion
from src.explainability.explainer import DeceptionExplainer
from src.explainability.visualizer import ExplainabilityVisualizer
from src.facial.extractor import FacialFeatureExtractor
from src.facial.visualizer import FacialVisualizer
from src.model.calibration import SubjectCalibrator
from src.model.classifier import DeceptionClassifier
from src.temporal.analyzer import TemporalAnalyzer
from src.temporal.visualizer import TemporalVisualizer
from src.utils.config import AppConfig, get_default_config
from src.utils.io import extract_audio_from_video, save_results


class DeceptionPipeline:
    """Main pipeline for multimodal deception analysis."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize pipeline with all sub-modules.

        Args:
            config: Optional application configuration. If None, default config is used.
        """
        self.config = config or get_default_config()
        self.logger = logging.getLogger(__name__)

        # Initialize all components with config
        self.facial_extractor = FacialFeatureExtractor(self.config.features)
        self.audio_extractor = AudioFeatureExtractor(self.config.audio)
        self.fusion = MultimodalFusion(self.config)
        self.classifier = DeceptionClassifier(self.config.model)
        self.calibrator = SubjectCalibrator(self.classifier, self.config)
        self.temporal_analyzer = TemporalAnalyzer(self.config.temporal)
        self.explainer = DeceptionExplainer(self.classifier)

        # Visualizers (no config needed)
        self.facial_visualizer = FacialVisualizer()
        self.temporal_visualizer = TemporalVisualizer()
        self.explainability_visualizer = ExplainabilityVisualizer()

        self.logger.info("DeceptionPipeline initialized with default config.")

    def analyze_video(
        self,
        video_path: str,
        subject_id: Optional[str] = None,
        generate_visualizations: bool = True,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Full analysis pipeline for a single video.

        Steps:
        1. Extract facial features from video
        2. Extract audio from video (using io utilities), then extract audio features
        3. Fuse facial and audio features (align + cross-modal)
        4. Run classifier prediction
        5. Apply subject calibration if subject_id provided
        6. Run temporal analysis on probability trajectory
        7. Generate explanations using SHAP
        8. Generate visualizations if requested
        9. Compile results into a comprehensive report dict

        Args:
            video_path: Path to the video file.
            subject_id: Optional subject identifier for calibration.
            generate_visualizations: Whether to generate visualization plots.
            output_dir: Directory to save visualizations and intermediate results.
                If None, a temporary directory is used.

        Returns:
            dict with keys:
            - 'deception_probability': float
            - 'prediction': str ('truthful' or 'deceptive')
            - 'confidence': float
            - 'facial_features': DataFrame
            - 'audio_features': DataFrame
            - 'fused_features': DataFrame
            - 'temporal_analysis': dict
            - 'explanation': dict
            - 'calibrated': bool
            - 'visualizations': dict of file paths (if generated)
            - 'metadata': dict
        """
        self.logger.info(f"Starting analysis of video: {video_path}")
        result: Dict[str, Union[float, str, pd.DataFrame, dict, bool, list]] = {}
        visualizations = {}
        metadata = {"video_path": video_path, "subject_id": subject_id}

        # Determine output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="deception_analysis_")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Extract facial features
            self.logger.info("Extracting facial features...")
            facial_df = self.facial_extractor.process_video(video_path)
            result["facial_features"] = facial_df
            metadata["facial_frames"] = len(facial_df)
            self.logger.info(f"Extracted {len(facial_df)} facial frames.")

            # 2. Extract audio features
            self.logger.info("Extracting audio features...")
            # Extract audio from video
            audio_temp = output_path / "audio.wav"
            try:
                audio_path = extract_audio_from_video(video_path, audio_temp)
            except Exception as e:
                self.logger.warning(f"Audio extraction failed: {e}. Using empty audio features.")
                audio_df = pd.DataFrame()
            else:
                # Extract audio features
                audio_df = self.audio_extractor.process_audio_segments(
                    str(audio_path), segment_duration=2.0
                )
            result["audio_features"] = audio_df
            metadata["audio_segments"] = len(audio_df)

            # 3. Fuse features
            self.logger.info("Fusing facial and audio features...")
            aligned_df = self.fusion.align_features(
                facial_df, audio_df, video_fps=self.config.features.fps
            )
            fused_df = self.fusion.extract_fusion_features(aligned_df)
            result["fused_features"] = fused_df

            # 4. Run classifier prediction
            self.logger.info("Running classifier prediction...")
            if not self.classifier.is_fitted_:
                self.logger.warning("Classifier not trained. Using default probability 0.5.")
                proba = 0.5
                prediction = "truthful"
                confidence = 0.5
                per_frame_probs = pd.Series([0.5] * len(fused_df))
            else:
                # Prepare features for classification
                X, feature_names = self.classifier.prepare_features(fused_df)
                if X.size == 0:
                    self.logger.warning("No features for classification.")
                    proba = 0.5
                    prediction = "truthful"
                    confidence = 0.5
                    per_frame_probs = pd.Series([0.5] * len(fused_df))
                else:
                    # Overall prediction (average across frames)
                    proba = float(self.classifier.predict_deception_score(X).mean())
                    prediction = "deceptive" if proba >= self.config.model.decision_threshold else "truthful"
                    confidence = abs(proba - 0.5) * 2  # map 0.5->0, 0/1->1
                    # Per-frame probabilities for temporal analysis
                    per_frame_probs = pd.Series(self.classifier.predict_deception_score(X))

            result["deception_probability"] = proba
            result["prediction"] = prediction
            result["confidence"] = confidence
            metadata["decision_threshold"] = self.config.model.decision_threshold

            # 5. Subject calibration
            calibrated = False
            if subject_id is not None and self.classifier.is_fitted_:
                self.logger.info(f"Applying subject calibration for {subject_id}...")
                try:
                    # Update baseline with current features
                    self.calibrator.add_subject_profile(subject_id, fused_df)
                    # Calibrate predictions
                    import numpy as np
                    calibrated_preds = np.array([
                        self.calibrator.calibrate_prediction(p, subject_id)
                        for p in per_frame_probs.values
                    ])
                    if calibrated_preds is not None:
                        calibrated_proba = float(calibrated_preds.mean())
                        calibrated_pred = (
                            "deceptive"
                            if calibrated_proba >= self.config.model.decision_threshold
                            else "truthful"
                        )
                        result["deception_probability"] = calibrated_proba
                        result["prediction"] = calibrated_pred
                        result["confidence"] = abs(calibrated_proba - 0.5) * 2
                        calibrated = True
                        per_frame_probs = pd.Series(calibrated_preds)
                except Exception as e:
                    self.logger.error(f"Subject calibration failed: {e}")
            result["calibrated"] = calibrated

            # 6. Temporal analysis
            self.logger.info("Performing temporal analysis...")
            timestamps = fused_df.get("timestamp", pd.Series(range(len(fused_df))))
            temporal_result = self.temporal_analyzer.analyze(
                per_frame_probs.values, timestamps.values
            )
            
            # Map temporal results for API
            if isinstance(timestamps, pd.Series):
                temporal_result["timestamps"] = timestamps.tolist()
            else:
                temporal_result["timestamps"] = list(timestamps)
                
            if "smoothed_probabilities" in temporal_result:
                import numpy as np
                if isinstance(temporal_result["smoothed_probabilities"], np.ndarray):
                    temporal_result["probability_trajectory"] = temporal_result["smoothed_probabilities"].tolist()
                else:
                    temporal_result["probability_trajectory"] = list(temporal_result["smoothed_probabilities"])
            
            result["temporal_analysis"] = temporal_result

            # 7. Generate explanations
            self.logger.info("Generating explanations...")
            if self.classifier.is_fitted_ and X.size > 0:
                try:
                    # Explainer expects a single sample (1, n_features). 
                    # We use the mean of all frames to explain the overall video prediction.
                    import numpy as np
                    X_mean = np.nanmean(X, axis=0, keepdims=True)
                    explanation = self.explainer.explain_prediction(
                        X=X_mean,
                        feature_names=feature_names,
                        prediction=proba,
                        subject_id=subject_id
                    )
                    result["explanation"] = explanation
                except Exception as e:
                    self.logger.warning(f"Explanation generation failed: {e}")
                    result["explanation"] = {"error": str(e)}
            else:
                result["explanation"] = {"info": "Classifier not trained or no features."}

            # 8. Generate visualizations
            if generate_visualizations:
                self.logger.info("Generating visualizations...")
                try:
                    # Facial landmarks visualization
                    facial_viz_path = output_path / "facial_landmarks.png"
                    self.facial_visualizer.plot_landmarks(facial_df, str(facial_viz_path))
                    visualizations["facial_landmarks"] = str(facial_viz_path)
                except Exception as e:
                    self.logger.warning(f"Facial visualization failed: {e}")

                try:
                    # Temporal probability trajectory
                    temporal_viz_path = output_path / "temporal_trajectory.png"
                    self.temporal_visualizer.plot_trajectory(
                        per_frame_probs.values,
                        timestamps.values,
                        str(temporal_viz_path),
                    )
                    visualizations["temporal_trajectory"] = str(temporal_viz_path)
                except Exception as e:
                    self.logger.warning(f"Temporal visualization failed: {e}")

                if self.classifier.is_fitted_ and X.size > 0:
                    try:
                        # SHAP summary plot
                        shap_viz_path = output_path / "shap_summary.png"
                        self.explainability_visualizer.plot_shap_summary(
                            self.explainer, X, str(shap_viz_path)
                        )
                        visualizations["shap_summary"] = str(shap_viz_path)
                    except Exception as e:
                        self.logger.warning(f"SHAP visualization failed: {e}")

            result["visualizations"] = visualizations

            # 9. Metadata
            metadata.update(
                {
                    "video_fps": self.config.features.fps,
                    "audio_sample_rate": self.config.audio.sample_rate,
                    "pipeline_version": "1.0",
                }
            )
            result["metadata"] = metadata

            self.logger.info(f"Analysis completed. Prediction: {prediction} (prob={proba:.3f})")
            return result

        except Exception as e:
            self.logger.error(f"Pipeline analysis failed: {e}")
            # Return partial results with error flag
            error_result = {
                "error": str(e),
                "deception_probability": 0.5,
                "prediction": "error",
                "confidence": 0.0,
                "facial_features": pd.DataFrame(),
                "audio_features": pd.DataFrame(),
                "fused_features": pd.DataFrame(),
                "temporal_analysis": {},
                "explanation": {},
                "calibrated": False,
                "visualizations": {},
                "metadata": metadata,
            }
            return error_result

    def analyze_batch(
        self,
        video_paths: List[str],
        subject_ids: Optional[List[str]] = None,
        generate_visualizations: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[dict]:
        """Analyze multiple videos. Wraps analyze_video with progress tracking.

        Args:
            video_paths: List of video file paths.
            subject_ids: Optional list of subject IDs (same length as video_paths).
            generate_visualizations: Whether to generate visualizations per video.
            output_dir: Base output directory. Subdirectories will be created per video.

        Returns:
            List of analysis result dicts, one per video.
        """
        if subject_ids is not None and len(subject_ids) != len(video_paths):
            raise ValueError("subject_ids must have same length as video_paths")

        results = []
        for idx, video_path in enumerate(tqdm(video_paths, desc="Analyzing videos")):
            subject_id = subject_ids[idx] if subject_ids else None
            video_output_dir = None
            if output_dir is not None:
                video_name = Path(video_path).stem
                video_output_dir = str(Path(output_dir) / video_name)
                Path(video_output_dir).mkdir(parents=True, exist_ok=True)

            result = self.analyze_video(
                video_path,
                subject_id=subject_id,
                generate_visualizations=generate_visualizations,
                output_dir=video_output_dir,
            )
            results.append(result)

        return results

    def train(
        self,
        video_paths: List[str],
        labels: List[int],
        subject_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Train the classifier on labeled video data.

        Steps:
        1. Extract features from all videos
        2. Fuse features
        3. Train classifier
        4. Compute training metrics
        5. Save model if output_dir provided

        Args:
            video_paths: List of video file paths.
            labels: Binary labels (0=truthful, 1=deceptive).
            subject_ids: Optional list of subject IDs for calibration baseline.
            output_dir: Directory to save trained model and training results.

        Returns:
            dict with training metrics and model info.
        """
        if len(video_paths) != len(labels):
            raise ValueError("video_paths and labels must have same length")
        if subject_ids is not None and len(subject_ids) != len(video_paths):
            raise ValueError("subject_ids must have same length as video_paths")

        self.logger.info(f"Starting training on {len(video_paths)} videos.")

        # 1. Extract features from all videos
        all_fused_dfs = []
        for idx, video_path in enumerate(tqdm(video_paths, desc="Extracting features")):
            try:
                # Extract facial features
                facial_df = self.facial_extractor.process_video(video_path)

                # Extract audio features
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    audio_path = extract_audio_from_video(video_path, tmp.name)
                    audio_df = self.audio_extractor.process_audio_segments(str(audio_path), segment_duration=2.0)

                # Fuse
                aligned_df = self.fusion.align_features(
                    facial_df, audio_df, video_fps=self.config.features.fps
                )
                fused_df = self.fusion.extract_fusion_features(aligned_df)
                fused_df["label"] = labels[idx]
                if subject_ids is not None:
                    fused_df["subject_id"] = subject_ids[idx]
                all_fused_dfs.append(fused_df)
            except Exception as e:
                self.logger.warning(f"Failed to extract features from {video_path}: {e}")
                continue

        if not all_fused_dfs:
            raise ValueError("No features extracted from any video.")

        # Concatenate all features
        full_df = pd.concat(all_fused_dfs, ignore_index=True)

        # 2. Prepare feature matrix and labels
        X, feature_names = self.classifier.prepare_features(full_df)
        y = full_df["label"].values

        if X.size == 0:
            raise ValueError("No valid features after preparation.")

        # 3. Train classifier
        self.logger.info(f"Training classifier with {X.shape[0]} samples, {X.shape[1]} features.")
        metrics = self.classifier.train(X, y, feature_names)

        # 4. Update calibrator with subject baselines if subject_ids provided
        if subject_ids is not None and self.classifier.is_fitted_:
            self.logger.info("Updating subject baselines...")
            for subject_id in set(subject_ids):
                subject_mask = full_df["subject_id"] == subject_id
                if subject_mask.any():
                    subject_features = full_df[subject_mask]
                    try:
                        self.calibrator.add_subject_profile(subject_id, subject_features)
                    except Exception as e:
                        self.logger.warning(f"Failed to update baseline for {subject_id}: {e}")

        # 5. Save model if output_dir provided
        if output_dir is not None:
            model_path = Path(output_dir) / "deception_classifier.joblib"
            self.classifier.save(str(model_path))
            metrics["model_path"] = str(model_path)
            self.logger.info(f"Model saved to {model_path}")

            # Save training results
            results_path = Path(output_dir) / "training_results.json"
            save_results(metrics, str(results_path))

        self.logger.info("Training completed.")
        return metrics

    def generate_report(self, analysis_result: dict, output_path: str) -> str:
        """Generate a comprehensive text/JSON report from analysis results.

        Creates a formatted report including:
        - Overall deception assessment
        - Key behavioral indicators
        - Temporal analysis summary
        - Feature importance highlights
        - Suspicious intervals if detected

        Args:
            analysis_result: Result dict from analyze_video.
            output_path: Path to save the report (JSON).

        Returns:
            Path to the saved report file.
        """
        from src.utils.io import save_results

        # Build report structure
        report = {
            "summary": {
                "prediction": analysis_result.get("prediction", "unknown"),
                "deception_probability": analysis_result.get("deception_probability", 0.0),
                "confidence": analysis_result.get("confidence", 0.0),
                "calibrated": analysis_result.get("calibrated", False),
            },
            "temporal_analysis": analysis_result.get("temporal_analysis", {}),
            "explanation": analysis_result.get("explanation", {}),
            "metadata": analysis_result.get("metadata", {}),
        }

        # Add feature importance if available
        if self.classifier.is_fitted_:
            try:
                importance_df = self.classifier.get_feature_importance()
                top_features = importance_df.head(10).to_dict(orient="records")
                report["feature_importance"] = top_features
            except Exception as e:
                self.logger.warning(f"Could not compute feature importance: {e}")
                report["feature_importance"] = []

        # Add suspicious intervals from temporal analysis
        temporal = analysis_result.get("temporal_analysis", {})
        if "suspicious_intervals" in temporal:
            report["suspicious_intervals"] = temporal["suspicious_intervals"]
        if "spikes" in temporal:
            report["spikes"] = temporal["spikes"]

        # Save report as JSON
        save_results(report, output_path)
        self.logger.info(f"Report saved to {output_path}")
        return output_path

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained classifier model.

        Args:
            model_path: Path to the saved model file.
        """
        self.classifier.load(model_path)
        # Re-initialize explainer with the loaded classifier
        self.explainer = DeceptionExplainer(self.classifier)
        # Re-initialize calibrator with the loaded classifier
        self.calibrator = SubjectCalibrator(self.classifier, self.config)
        self.logger.info(f"Model loaded from {model_path}")

    def save_model(self, model_path: str) -> None:
        """Save the current classifier model.

        Args:
            model_path: Path where the model should be saved.
        """
        self.classifier.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
