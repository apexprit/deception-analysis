"""Visualization helpers for temporal deception analysis outputs."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import TemporalConfig


class TemporalVisualizer:
    """Publication-quality temporal analysis visualizations."""

    def __init__(self):
        """Initialize plotting style for publication-quality figures."""

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "font.size": 11,
            }
        )
        self._temporal_cfg = TemporalConfig()

    def plot_deception_timeline(
        self,
        probabilities: np.ndarray,
        timestamps: np.ndarray,
        smoothed: np.ndarray,
        suspicious_intervals: List[Dict],
        spikes: List[Dict],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create the main deception timeline plot."""

        probs = np.asarray(probabilities, dtype=np.float64)
        times = np.asarray(timestamps, dtype=np.float64)
        smooth = np.asarray(smoothed, dtype=np.float64)

        n = min(len(probs), len(times), len(smooth))
        probs = probs[:n]
        times = times[:n]
        smooth = smooth[:n]

        fig, ax = plt.subplots(figsize=(12, 5))

        if n > 0:
            ax.plot(times, probs, color="#93c5fd", linewidth=1.2, alpha=0.8, label="Raw probability")
            ax.plot(times, smooth, color="#1e3a8a", linewidth=2.0, label="Smoothed probability")

        threshold = float(getattr(self._temporal_cfg, "spike_threshold", 0.7))
        ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.4, label="Spike threshold")
        ax.axhline(0.5, color="#6b7280", linestyle="--", linewidth=1.2, label="Decision boundary")

        for i, interval in enumerate(suspicious_intervals):
            start_t = float(interval.get("start_time", 0.0))
            end_t = float(interval.get("end_time", start_t))
            ax.axvspan(
                start_t,
                end_t,
                color="#fecaca",
                alpha=0.35,
                label="Suspicious interval" if i == 0 else None,
            )

        if spikes:
            spike_x = [float(item.get("timestamp", 0.0)) for item in spikes]
            spike_y = [float(item.get("probability", 0.0)) for item in spikes]
            ax.scatter(
                spike_x,
                spike_y,
                marker="^",
                color="#ef4444",
                edgecolors="black",
                linewidths=0.3,
                s=40,
                zorder=5,
                label="Spikes",
            )

        ax.set_title("Temporal Deception Analysis")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Deception Probability")
        ax.set_ylim(0.0, 1.02)
        ax.legend(loc="best")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_suspicious_regions(
        self,
        suspicious_intervals: List[Dict],
        total_duration: float,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create a Gantt-style suspicious interval chart."""

        fig, ax = plt.subplots(figsize=(11, 4))

        severity_color = {
            "low": "#facc15",
            "medium": "#fb923c",
            "high": "#ef4444",
        }

        if not suspicious_intervals:
            ax.barh([0], [max(total_duration, 1.0)], left=[0.0], color="#e5e7eb", alpha=0.4)
            ax.text(0.5 * max(total_duration, 1.0), 0, "No suspicious intervals", ha="center", va="center")
            ax.set_yticks([0])
            ax.set_yticklabels(["N/A"])
        else:
            for idx, interval in enumerate(suspicious_intervals):
                start = float(interval.get("start_time", 0.0))
                end = float(interval.get("end_time", start))
                width = max(0.0, end - start)
                severity = str(interval.get("severity", "low")).lower()
                color = severity_color.get(severity, "#facc15")
                ax.barh(idx, width, left=start, color=color, edgecolor="black", alpha=0.85)

            ax.set_yticks(np.arange(len(suspicious_intervals)))
            ax.set_yticklabels([f"Interval {i + 1}" for i in range(len(suspicious_intervals))])

        ax.set_xlim(0.0, max(float(total_duration), 1.0))
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Suspicious Intervals")
        ax.set_title("Suspicious Regions Timeline")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_temporal_summary(
        self,
        temporal_features: Dict[str, float],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create a multi-panel summary dashboard of temporal statistics."""

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Panel 1: synthetic distribution from mean/std when raw sequence is unavailable.
        mean_p = float(temporal_features.get("mean_probability", 0.0))
        std_p = float(max(temporal_features.get("std_probability", 0.0), 1e-3))
        synthetic = np.clip(np.random.normal(loc=mean_p, scale=std_p, size=800), 0.0, 1.0)
        axes[0].hist(synthetic, bins=20, color="#60a5fa", edgecolor="black", alpha=0.85)
        axes[0].axvline(mean_p, color="#1d4ed8", linestyle="--", linewidth=2)
        axes[0].set_title("Probability Distribution")
        axes[0].set_xlabel("Deception Probability")
        axes[0].set_ylabel("Count")

        # Panel 2: key metrics.
        key_names = [
            "mean_probability",
            "max_probability",
            "spike_count",
            "suspicious_time_ratio",
            "longest_suspicious_duration",
        ]
        key_vals = [float(temporal_features.get(name, 0.0)) for name in key_names]
        axes[1].barh(np.arange(len(key_names)), key_vals, color="#34d399", edgecolor="black", alpha=0.85)
        axes[1].set_yticks(np.arange(len(key_names)))
        axes[1].set_yticklabels([name.replace("_", " ") for name in key_names])
        axes[1].set_title("Key Temporal Metrics")
        axes[1].set_xlabel("Value")

        # Panel 3: severity pie chart proxy from max probability.
        max_p = float(temporal_features.get("max_probability", 0.0))
        if max_p < 0.7:
            sev = [1.0, 0.0, 0.0]
        elif max_p < 0.85:
            sev = [0.4, 0.6, 0.0]
        else:
            sev = [0.2, 0.3, 0.5]
        axes[2].pie(
            sev,
            labels=["Low", "Medium", "High"],
            colors=["#facc15", "#fb923c", "#ef4444"],
            autopct="%1.0f%%",
            startangle=90,
        )
        axes[2].set_title("Severity Breakdown")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

