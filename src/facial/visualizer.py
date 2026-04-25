"""Visualization utilities for facial landmarks and feature timelines."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class FacialVisualizer:
    """Visualizes facial landmarks and feature values on video frames."""

    def __init__(self):
        """Define landmark groups and display thresholds for overlays."""

        self.eye_landmarks = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
        self.mouth_landmarks = [13, 14, 61, 291]
        self.eyebrow_landmarks = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
        self.eye_threshold = 0.22
        self.mouth_threshold = 0.35
        self.brow_threshold = 0.18

    @staticmethod
    def _as_array(landmarks) -> np.ndarray:
        """Convert supported landmark representations to an array."""

        if landmarks is None:
            return np.empty((0, 3), dtype=np.float32)
        if isinstance(landmarks, np.ndarray):
            return landmarks
        if hasattr(landmarks, "landmark"):
            return np.array([[point.x, point.y, point.z] for point in landmarks.landmark], dtype=np.float32)
        return np.asarray(landmarks, dtype=np.float32)

    def draw_landmarks(self, frame: np.ndarray, landmarks, feature_values: Dict[str, float]) -> np.ndarray:
        """Draw face mesh landmarks and highlight active feature groups.

        Args:
            frame: BGR image frame to annotate.
            landmarks: Landmark array or MediaPipe landmark list.
            feature_values: Dictionary of extracted facial feature values.

        Returns:
            Annotated frame copy.
        """

        output = frame.copy()
        points = self._as_array(landmarks)
        if points.size == 0:
            cv2.putText(output, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return output

        height, width = output.shape[:2]
        if np.nanmax(points[:, :2]) <= 1.5:
            draw_points = np.column_stack((points[:, 0] * width, points[:, 1] * height)).astype(int)
        else:
            draw_points = points[:, :2].astype(int)

        for x_coord, y_coord in draw_points:
            cv2.circle(output, (int(x_coord), int(y_coord)), 1, (0, 180, 0), -1)

        eye_open = min(
            float(feature_values.get("eye_left_openness", 1.0)),
            float(feature_values.get("eye_right_openness", 1.0)),
        )
        if eye_open < self.eye_threshold:
            # Eye closure highlights possible blinking or avoidance-related stress markers.
            for idx in self.eye_landmarks:
                cv2.circle(output, tuple(draw_points[idx]), 3, (255, 0, 0), -1)

        if float(feature_values.get("mouth_openness", 0.0)) > self.mouth_threshold:
            # Mouth movement highlights speech articulation shifts and possible hesitation.
            for idx in self.mouth_landmarks:
                cv2.circle(output, tuple(draw_points[idx]), 4, (0, 0, 255), -1)

        if float(feature_values.get("brow_tension", 0.0)) > self.brow_threshold:
            # Brow tension highlights facial stress or emotion-suppression cues.
            for idx in self.eyebrow_landmarks:
                cv2.circle(output, tuple(draw_points[idx]), 3, (0, 255, 255), -1)

        overlay_items = [
            ("Eye", eye_open),
            ("Mouth", float(feature_values.get("mouth_openness", np.nan))),
            ("Brow", float(feature_values.get("brow_tension", np.nan))),
            ("Yaw", float(feature_values.get("head_yaw", np.nan))),
        ]
        for row, (name, value) in enumerate(overlay_items):
            cv2.putText(output, f"{name}: {value:.3f}", (20, 30 + row * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return output

    def create_feature_timeline(self, features_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """Create a multi-panel timeline plot of facial features.

        Args:
            features_df: DataFrame containing frame-level facial features.
            save_path: Optional path where the figure should be saved.

        Returns:
            Matplotlib figure containing four synchronized subplots.
        """

        sns.set_theme(style="whitegrid")
        time = features_df["timestamp"] if "timestamp" in features_df else features_df.index
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(time, features_df.get("eye_left_openness"), label="Left eye")
        axes[0].plot(time, features_df.get("eye_right_openness"), label="Right eye")
        axes[0].set_ylabel("Eye openness")
        axes[0].legend(loc="best")

        axes[1].plot(time, features_df.get("mouth_openness"), label="Openness")
        axes[1].plot(time, features_df.get("mouth_width"), label="Width")
        axes[1].set_ylabel("Mouth movement")
        axes[1].legend(loc="best")

        axes[2].plot(time, features_df.get("brow_tension"), label="Brow tension", color="tab:orange")
        axes[2].set_ylabel("Eyebrow tension")
        axes[2].legend(loc="best")

        for column in ["head_pitch", "head_yaw", "head_roll"]:
            axes[3].plot(time, features_df.get(column), label=column.replace("head_", "").title())
        axes[3].set_ylabel("Degrees")
        axes[3].set_xlabel("Time (s)")
        axes[3].legend(loc="best")

        fig.suptitle("Facial Feature Timeline")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

