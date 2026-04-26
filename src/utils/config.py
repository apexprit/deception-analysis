"""Configuration objects for the deception analysis system.

This module centralizes path, feature extraction, model, and temporal analysis
settings so downstream components can share a consistent configuration surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ProjectConfig:
    """Filesystem paths used throughout the project.

    Attributes:
        root_dir: Root directory for the project repository.
        src_dir: Source package directory.
        data_dir: Directory containing datasets and samples.
        sample_data_dir: Directory containing small sample inputs.
        model_dir: Directory for serialized model artifacts.
        results_dir: Directory for metrics, reports, and predictions.
        docs_dir: Documentation directory.
        notebooks_dir: Exploratory notebooks directory.
        demo_dir: Demo application directory.
    """

    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    src_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    sample_data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    docs_dir: Path = field(init=False)
    notebooks_dir: Path = field(init=False)
    demo_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Derive project paths from the configured root directory."""

        object.__setattr__(self, "src_dir", self.root_dir / "src")
        object.__setattr__(self, "data_dir", self.root_dir / "data")
        object.__setattr__(self, "sample_data_dir", self.root_dir / "data" / "sample")
        object.__setattr__(self, "model_dir", self.root_dir / "models")
        object.__setattr__(self, "results_dir", self.root_dir / "results")
        object.__setattr__(self, "docs_dir", self.root_dir / "docs")
        object.__setattr__(self, "notebooks_dir", self.root_dir / "notebooks")
        object.__setattr__(self, "demo_dir", self.root_dir / "demo")


@dataclass(frozen=True)
class FeatureConfig:
    """Facial and behavioral feature extraction parameters."""

    fps: float = 30.0
    eye_openness_threshold: float = 0.22
    mouth_openness_threshold: float = 0.35
    eyebrow_tension_threshold: float = 0.18
    gaze_shift_threshold: float = 0.15
    head_pose_threshold_degrees: float = 20.0
    blink_duration_threshold_frames: int = 3
    micro_expression_window_frames: int = 15
    landmark_smoothing_alpha: float = 0.6
    min_face_detection_confidence: float = 0.6
    min_face_tracking_confidence: float = 0.5
    face_mesh_refine_landmarks: bool = True


@dataclass(frozen=True)
class AudioConfig:
    """Audio feature extraction parameters for speech analysis."""

    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mfcc: int = 13
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 8000.0
    trim_top_db: int = 30
    silence_threshold_db: float = -40.0
    pitch_floor_hz: float = 75.0
    pitch_ceiling_hz: float = 600.0
    jitter_window_seconds: float = 0.04
    speaking_rate_window_seconds: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    """Machine learning model and calibration parameters."""

    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 5
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    subsample: float = 0.9
    random_state: int = 42
    validation_size: float = 0.2
    calibration_method: str = "isotonic"
    cross_validation_folds: int = 5
    class_weight_strategy: str = "balanced"
    decision_threshold: float = 0.5


@dataclass(frozen=True)
class TemporalConfig:
    """Temporal analysis parameters for deception probability trajectories."""

    window_size: int = 30
    spike_threshold: float = 0.7
    min_suspicious_duration: int = 10
    smoothing_window: int = 5
    baseline_window_size: int = 60
    subject_adaptation_rate: float = 0.15
    cooldown_frames: int = 15
    trend_slope_threshold: float = 0.02
    aggregation_percentiles: Tuple[int, int, int] = (50, 75, 90)


@dataclass(frozen=True)
class AppConfig:
    """Combined configuration object for the full system."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)


def get_default_config() -> AppConfig:
    """Return the default combined configuration for the project.

    Returns:
        AppConfig: A fully populated immutable configuration object.
    """

    return AppConfig()
