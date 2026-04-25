"""Audio and speech feature extraction using librosa."""

from __future__ import annotations

from typing import Dict, List, Optional

import librosa
import numpy as np
import pandas as pd

from src.utils.config import AudioConfig, get_default_config


class AudioFeatureExtractor:
    """Extracts speech and audio features for deception analysis using librosa."""

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize the audio extractor.

        Args:
            config: Optional audio configuration. Defaults to application audio config.
        """

        self.config = config or get_default_config().audio

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        """Convert non-finite numeric outputs to zero."""

        return float(value) if np.isfinite(value) else 0.0

    def _empty_features(self) -> Dict[str, float]:
        """Return a zero-filled feature dictionary for invalid audio."""

        return {name: 0.0 for name in self.get_feature_names()}

    def extract_pitch_variation(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract pitch/f0 features using probabilistic YIN.

        Args:
            y: Audio waveform.
            sr: Sample rate.

        Returns:
            Dictionary with pitch summary and voiced-frame ratio.
        """

        if y.size < self.config.hop_length or np.allclose(y, 0.0):
            return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0, "pitch_variation_coefficient": 0.0, "voiced_ratio": 0.0}
        try:
            f0, _, voiced_prob = librosa.pyin(y, fmin=65.0, fmax=2093.0, sr=sr, hop_length=self.config.hop_length)
        except Exception:
            return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0, "pitch_variation_coefficient": 0.0, "voiced_ratio": 0.0}
        voiced = f0[np.isfinite(f0)]
        if voiced.size == 0:
            return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0, "pitch_variation_coefficient": 0.0, "voiced_ratio": 0.0}
        mean = float(np.mean(voiced))
        std = float(np.std(voiced))
        # Pitch elevation and instability can reflect stress or high cognitive load during deceptive speech.
        return {
            "pitch_mean": mean,
            "pitch_std": std,
            "pitch_range": float(np.max(voiced) - np.min(voiced)),
            "pitch_variation_coefficient": self._finite_or_zero(std / mean if mean else 0.0),
            "voiced_ratio": float(np.mean(np.asarray(voiced_prob) > 0.5)) if voiced_prob is not None else float(voiced.size / f0.size),
        }

    def extract_pause_analysis(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze speech pauses and timing with adaptive energy VAD.

        Args:
            y: Audio waveform.
            sr: Sample rate.

        Returns:
            Dictionary with pause count, duration, rate, and speech ratio.
        """

        duration = len(y) / float(sr) if sr else 0.0
        if y.size == 0 or duration <= 0.0:
            return {"pause_count": 0.0, "avg_pause_duration": 0.0, "max_pause_duration": 0.0, "pause_rate": 0.0, "speech_ratio": 0.0}
        rms = librosa.feature.rms(y=y, frame_length=self.config.n_fft, hop_length=self.config.hop_length)[0]
        if rms.size == 0 or np.allclose(rms, 0.0):
            return {"pause_count": 1.0, "avg_pause_duration": duration, "max_pause_duration": duration, "pause_rate": 60.0 / duration, "speech_ratio": 0.0}
        threshold = max(float(np.median(rms) * 0.6), float(np.max(rms) * 0.03))
        silent = rms <= threshold
        frame_seconds = self.config.hop_length / float(sr)
        min_pause_frames = max(1, int(np.ceil(0.3 / frame_seconds)))
        pauses: List[float] = []
        run = 0
        for is_silent in silent:
            if is_silent:
                run += 1
            else:
                if run >= min_pause_frames:
                    pauses.append(run * frame_seconds)
                run = 0
        if run >= min_pause_frames:
            pauses.append(run * frame_seconds)
        pause_time = float(np.sum(pauses)) if pauses else 0.0
        # Pauses and hesitations are deception-relevant when they indicate formulation difficulty.
        return {
            "pause_count": float(len(pauses)),
            "avg_pause_duration": float(np.mean(pauses)) if pauses else 0.0,
            "max_pause_duration": float(np.max(pauses)) if pauses else 0.0,
            "pause_rate": float(len(pauses) / (duration / 60.0)) if duration > 0 else 0.0,
            "speech_ratio": float(np.clip(1.0 - pause_time / duration, 0.0, 1.0)),
        }

    def extract_speech_energy(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract speech energy and spectral intensity features.

        Args:
            y: Audio waveform.
            sr: Sample rate.

        Returns:
            Dictionary with RMS energy and spectral descriptors.
        """

        if y.size == 0:
            return {"energy_mean": 0.0, "energy_std": 0.0, "energy_range": 0.0, "energy_variation": 0.0, "spectral_centroid_mean": 0.0, "spectral_bandwidth_mean": 0.0}
        rms = librosa.feature.rms(y=y, frame_length=self.config.n_fft, hop_length=self.config.hop_length)[0]
        mean = float(np.mean(rms)) if rms.size else 0.0
        std = float(np.std(rms)) if rms.size else 0.0
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length)[0]
        # Vocal intensity spikes can pair with facial stress markers in multimodal deception cues.
        return {
            "energy_mean": mean,
            "energy_std": std,
            "energy_range": float(np.max(rms) - np.min(rms)) if rms.size else 0.0,
            "energy_variation": self._finite_or_zero(std / mean if mean else 0.0),
            "spectral_centroid_mean": float(np.mean(centroid)) if centroid.size else 0.0,
            "spectral_bandwidth_mean": float(np.mean(bandwidth)) if bandwidth.size else 0.0,
        }

    def extract_mfcc_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract MFCC and delta-MFCC speech characterization features.

        Args:
            y: Audio waveform.
            sr: Sample rate.

        Returns:
            Dictionary containing MFCC means, standard deviations, and delta means.
        """

        features: Dict[str, float] = {}
        if y.size == 0:
            for idx in range(self.config.n_mfcc):
                features[f"mfcc_{idx}_mean"] = 0.0
                features[f"mfcc_{idx}_std"] = 0.0
                features[f"delta_mfcc_{idx}_mean"] = 0.0
            return features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.n_mfcc, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        delta = librosa.feature.delta(mfcc) if mfcc.shape[1] >= 3 else np.zeros_like(mfcc)
        # MFCCs capture voice quality shifts that may accompany stress or controlled speech.
        for idx in range(self.config.n_mfcc):
            features[f"mfcc_{idx}_mean"] = float(np.mean(mfcc[idx]))
            features[f"mfcc_{idx}_std"] = float(np.std(mfcc[idx]))
            features[f"delta_mfcc_{idx}_mean"] = float(np.mean(delta[idx]))
        return features

    def process_audio(self, audio_path: str) -> Dict[str, float]:
        """Process a complete audio file and extract all features.

        Args:
            audio_path: Path to an audio file readable by librosa.

        Returns:
            Combined audio feature dictionary.
        """

        try:
            y, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        except Exception:
            return self._empty_features()
        if y.size == 0:
            return self._empty_features()
        features: Dict[str, float] = {}
        features.update(self.extract_pitch_variation(y, sr))
        features.update(self.extract_pause_analysis(y, sr))
        features.update(self.extract_speech_energy(y, sr))
        features.update(self.extract_mfcc_features(y, sr))
        return {name: self._finite_or_zero(float(features.get(name, 0.0))) for name in self.get_feature_names()}

    def process_audio_segments(self, audio_path: str, segment_duration: float = 2.0) -> pd.DataFrame:
        """Process audio in fixed-duration segments for temporal analysis.

        Args:
            audio_path: Path to an audio file readable by librosa.
            segment_duration: Segment length in seconds.

        Returns:
            DataFrame with segment index, timing, and audio feature columns.
        """

        try:
            y, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        except Exception:
            columns = ["segment_idx", "start_time", "end_time"] + self.get_feature_names()
            return pd.DataFrame(columns=columns)
        segment_samples = max(1, int(segment_duration * sr))
        records: List[Dict[str, float]] = []
        for segment_idx, start in enumerate(range(0, len(y), segment_samples)):
            end = min(start + segment_samples, len(y))
            segment = y[start:end]
            record: Dict[str, float] = {
                "segment_idx": float(segment_idx),
                "start_time": start / float(sr),
                "end_time": end / float(sr),
            }
            if segment.size < max(16, self.config.hop_length // 2):
                record.update(self._empty_features())
            else:
                record.update(self.extract_pitch_variation(segment, sr))
                record.update(self.extract_pause_analysis(segment, sr))
                record.update(self.extract_speech_energy(segment, sr))
                record.update(self.extract_mfcc_features(segment, sr))
            records.append(record)
        return pd.DataFrame(records, columns=["segment_idx", "start_time", "end_time"] + self.get_feature_names())

    def get_feature_names(self) -> List[str]:
        """Return ordered names of all produced audio features."""

        names = [
            "pitch_mean",
            "pitch_std",
            "pitch_range",
            "pitch_variation_coefficient",
            "voiced_ratio",
            "pause_count",
            "avg_pause_duration",
            "max_pause_duration",
            "pause_rate",
            "speech_ratio",
            "energy_mean",
            "energy_std",
            "energy_range",
            "energy_variation",
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
        ]
        for idx in range(self.config.n_mfcc):
            names.extend([f"mfcc_{idx}_mean", f"mfcc_{idx}_std", f"delta_mfcc_{idx}_mean"])
        return names

