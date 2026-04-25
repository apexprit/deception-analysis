"""Multimodal fusion utilities for facial and audio features."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.config import AppConfig, get_default_config


class MultimodalFusion:
    """Fuses facial and audio features for multimodal deception analysis."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the fusion module.

        Args:
            config: Optional application configuration.
        """

        self.config = config or get_default_config()

    def align_features(self, facial_df: pd.DataFrame, audio_df: pd.DataFrame, video_fps: float = 30.0) -> pd.DataFrame:
        """Align facial frame-level and audio segment-level features by time.

        Args:
            facial_df: Frame-level facial feature DataFrame.
            audio_df: Segment-level audio feature DataFrame with start and end times.
            video_fps: Fallback video FPS if timestamps are not present.

        Returns:
            Combined DataFrame aligned to facial frame granularity.
        """

        if facial_df.empty:
            return facial_df.copy()
        aligned = facial_df.copy().reset_index(drop=True)
        if "timestamp" not in aligned.columns:
            aligned["timestamp"] = aligned.get("frame_idx", pd.Series(aligned.index)).astype(float) / float(video_fps)
        if audio_df.empty:
            return aligned

        audio_features = [col for col in audio_df.columns if col not in {"segment_idx", "start_time", "end_time"}]
        for column in audio_features:
            aligned[column] = np.nan

        for idx, timestamp in aligned["timestamp"].items():
            matches = audio_df[(audio_df["start_time"] <= timestamp) & (timestamp < audio_df["end_time"])]
            if matches.empty and timestamp == aligned["timestamp"].iloc[-1]:
                matches = audio_df[audio_df["end_time"] <= timestamp].tail(1)
            if not matches.empty:
                aligned.loc[idx, audio_features] = matches.iloc[0][audio_features].to_numpy(dtype=float)
        return aligned.ffill().bfill()

    @staticmethod
    def _rolling_corr(left: pd.Series, right: pd.Series, window: int = 30) -> pd.Series:
        """Compute robust rolling correlation with finite fallbacks."""

        corr = left.rolling(window=window, min_periods=max(3, window // 5)).corr(right)
        return corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Return z-scored values with zero variance protection."""

        std = float(series.std(skipna=True) or 0.0)
        if std <= 1e-8:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series.fillna(series.mean()) - float(series.mean())) / std

    def extract_fusion_features(self, aligned_df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-modal fusion features.

        Args:
            aligned_df: DataFrame containing aligned facial and audio features.

        Returns:
            DataFrame with original columns plus fusion-specific features.
        """

        fused = aligned_df.copy()
        if fused.empty:
            for name in self.get_fusion_feature_names():
                fused[name] = []
            return fused

        mouth = fused.get("mouth_openness", pd.Series(np.zeros(len(fused)), index=fused.index)).astype(float)
        energy = fused.get("energy_mean", pd.Series(np.zeros(len(fused)), index=fused.index)).astype(float)
        eye = fused.get("eye_left_openness", pd.Series(np.zeros(len(fused)), index=fused.index)).astype(float)
        pause = fused.get("pause_count", pd.Series(np.zeros(len(fused)), index=fused.index)).astype(float)
        pitch = fused.get("pitch_mean", pd.Series(np.zeros(len(fused)), index=fused.index)).astype(float)
        brow = fused.get("brow_tension", pd.Series(np.zeros(len(fused)), index=fused.index)).astype(float)

        # Audio-visual synchrony flags mismatches between visible articulation and speech energy.
        fused["audio_visual_sync"] = self._rolling_corr(mouth, energy)
        # Pause-gaze coupling captures whether vocal hesitation coincides with eye behavior changes.
        fused["pause_gaze_coupling"] = self._rolling_corr(eye, pause)
        # Composite stress combines vocal pitch, facial brow tension, and energy spikes.
        fused["stress_composite"] = (0.4 * self._zscore(pitch) + 0.35 * self._zscore(brow) + 0.25 * self._zscore(energy)).astype(float)
        return fused

    def get_fusion_feature_names(self) -> List[str]:
        """Return names of fusion-specific features."""

        return ["audio_visual_sync", "pause_gaze_coupling", "stress_composite"]

