"""Temporal deception probability analysis utilities.

This module is model-agnostic and operates on probability trajectories to detect
spikes, contiguous suspicious regions, and aggregate temporal statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.utils.config import TemporalConfig


class TemporalAnalyzer:
    """Frame-by-frame temporal deception analysis with spike detection."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        """Initialize temporal analyzer.

        Args:
            config: Optional temporal configuration.
        """

        self.config = config or TemporalConfig()
        self.window_size = int(getattr(self.config, "window_size", 30))
        self.spike_threshold = float(getattr(self.config, "spike_threshold", 0.7))
        self.min_suspicious_duration = int(getattr(self.config, "min_suspicious_duration", 10))
        self.smoothing_alpha = float(getattr(self.config, "smoothing_alpha", 0.3))

    def smooth_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply exponential moving average smoothing.

        Args:
            probabilities: Raw deception probabilities.

        Returns:
            np.ndarray: Smoothed probabilities.
        """

        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.size == 0:
            return np.array([], dtype=np.float64)
        if probs.size == 1:
            return np.clip(probs.copy(), 0.0, 1.0)

        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)

        alpha = float(np.clip(self.smoothing_alpha, 1e-3, 1.0))
        smoothed = np.empty_like(probs)
        smoothed[0] = probs[0]
        for i in range(1, len(probs)):
            # EMA is used because it denoises while preserving abrupt changes,
            # which are behaviorally meaningful in deception dynamics.
            smoothed[i] = alpha * probs[i] + (1.0 - alpha) * smoothed[i - 1]

        return smoothed

    def detect_spikes(self, probabilities: np.ndarray, timestamps: np.ndarray) -> List[Dict]:
        """Detect high-confidence temporal spikes.

        Args:
            probabilities: Raw deception probability sequence.
            timestamps: Timestamps aligned with probabilities.

        Returns:
            List[Dict]: Spike descriptors.
        """

        probs = np.asarray(probabilities, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64)
        n = min(len(probs), len(times))
        if n == 0:
            return []

        probs = probs[:n]
        times = times[:n]
        smoothed = self.smooth_probabilities(probs)

        if n == 1:
            if smoothed[0] >= self.spike_threshold:
                return [
                    {
                        "frame_idx": 0,
                        "timestamp": float(times[0]),
                        "probability": float(smoothed[0]),
                        "magnitude": float(smoothed[0] - self.spike_threshold),
                        "rate_of_change": 0.0,
                    }
                ]
            return []

        deriv = np.diff(smoothed, prepend=smoothed[0])
        deriv_std = float(np.std(deriv))
        deriv_threshold = 2.0 * deriv_std

        spikes: List[Dict] = []
        for i in range(n):
            prob_high = smoothed[i] >= self.spike_threshold
            roc_high = abs(deriv[i]) >= deriv_threshold if deriv_std > 0 else False
            if prob_high or roc_high:
                spikes.append(
                    {
                        "frame_idx": int(i),
                        "timestamp": float(times[i]),
                        "probability": float(smoothed[i]),
                        "magnitude": float(max(0.0, smoothed[i] - self.spike_threshold)),
                        "rate_of_change": float(deriv[i]),
                    }
                )

        return spikes

    def localize_suspicious_intervals(
        self,
        probabilities: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[Dict]:
        """Identify contiguous suspicious intervals.

        Args:
            probabilities: Raw deception probability sequence.
            timestamps: Timestamps aligned with probabilities.

        Returns:
            List[Dict]: Interval dictionaries.
        """

        probs = np.asarray(probabilities, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64)
        n = min(len(probs), len(times))
        if n == 0:
            return []

        probs = probs[:n]
        times = times[:n]
        smoothed = self.smooth_probabilities(probs)
        mask = smoothed >= self.spike_threshold

        if not np.any(mask):
            return []

        padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
        transitions = np.diff(padded)
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0] - 1

        intervals: List[Dict] = []
        for start, end in zip(starts, ends):
            duration_frames = int(end - start + 1)
            if duration_frames < self.min_suspicious_duration:
                continue

            segment = smoothed[start : end + 1]
            avg_p = float(np.mean(segment))
            max_p = float(np.max(segment))
            if max_p < 0.8:
                severity = "low"
            elif max_p < 0.9:
                severity = "medium"
            else:
                severity = "high"

            start_t = float(times[start])
            end_t = float(times[end])
            duration_s = float(max(0.0, end_t - start_t))

            intervals.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "start_time": start_t,
                    "end_time": end_t,
                    "duration_frames": duration_frames,
                    "duration_seconds": duration_s,
                    "avg_probability": avg_p,
                    "max_probability": max_p,
                    "severity": severity,
                }
            )

        return intervals

    def compute_temporal_features(
        self,
        probabilities: np.ndarray,
        timestamps: np.ndarray,
    ) -> Dict[str, float]:
        """Compute aggregate temporal statistics.

        Args:
            probabilities: Raw deception probability sequence.
            timestamps: Timestamps aligned with probabilities.

        Returns:
            Dict[str, float]: Aggregate temporal metrics.
        """

        probs = np.asarray(probabilities, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64)
        n = min(len(probs), len(times))
        if n == 0:
            return {
                "mean_probability": 0.0,
                "max_probability": 0.0,
                "std_probability": 0.0,
                "spike_count": 0.0,
                "suspicious_time_ratio": 0.0,
                "longest_suspicious_duration": 0.0,
                "transition_count": 0.0,
                "temporal_entropy": 0.0,
                "trend_slope": 0.0,
            }

        probs = probs[:n]
        times = times[:n]
        smoothed = self.smooth_probabilities(probs)

        spikes = self.detect_spikes(probs, times)
        intervals = self.localize_suspicious_intervals(probs, times)

        total_duration = float(max(1e-8, times[-1] - times[0])) if n > 1 else 1e-8
        suspicious_duration = float(sum(item["duration_seconds"] for item in intervals))
        longest_duration = float(max([item["duration_seconds"] for item in intervals], default=0.0))

        binary = (smoothed >= 0.5).astype(np.int8)
        transition_count = float(np.sum(np.abs(np.diff(binary)))) if n > 1 else 0.0

        hist, _ = np.histogram(smoothed, bins=10, range=(0.0, 1.0), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12))) if hist.size else 0.0

        if n > 1 and np.std(times) > 0:
            slope = float(np.polyfit(times, smoothed, deg=1)[0])
        else:
            slope = 0.0

        return {
            "mean_probability": float(np.mean(smoothed)),
            "max_probability": float(np.max(smoothed)),
            "std_probability": float(np.std(smoothed)),
            "spike_count": float(len(spikes)),
            "suspicious_time_ratio": float(np.clip(suspicious_duration / total_duration, 0.0, 1.0)),
            "longest_suspicious_duration": longest_duration,
            "transition_count": transition_count,
            "temporal_entropy": entropy,
            "trend_slope": slope,
        }

    def analyze(self, probabilities: np.ndarray, timestamps: np.ndarray, fps: float = 30.0) -> Dict:
        """Perform complete temporal analysis.

        Args:
            probabilities: Raw deception probability sequence.
            timestamps: Optional timestamp sequence. If empty, generated using fps.
            fps: Frame rate for synthetic timestamps.

        Returns:
            Dict: Full temporal analysis output.
        """

        probs = np.asarray(probabilities, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64)

        if probs.size == 0:
            return {
                "smoothed_probabilities": np.array([], dtype=np.float64),
                "spikes": [],
                "suspicious_intervals": [],
                "temporal_features": self.compute_temporal_features(np.array([]), np.array([])),
                "verdict": "truthful",
                "confidence": 0.0,
            }

        if times.size == 0 or len(times) != len(probs):
            safe_fps = fps if fps > 0 else 30.0
            times = np.arange(len(probs), dtype=np.float64) / float(safe_fps)

        smoothed = self.smooth_probabilities(probs)
        spikes = self.detect_spikes(probs, times)
        intervals = self.localize_suspicious_intervals(probs, times)
        features = self.compute_temporal_features(probs, times)

        mean_prob = float(features.get("mean_probability", 0.0))
        verdict = "deceptive" if mean_prob > 0.5 else "truthful"
        confidence = float(np.clip(abs(mean_prob - 0.5) * 2.0, 0.0, 1.0))

        return {
            "smoothed_probabilities": smoothed,
            "spikes": spikes,
            "suspicious_intervals": intervals,
            "temporal_features": features,
            "verdict": verdict,
            "confidence": confidence,
        }

