"""Facial micro-expression feature extraction using MediaPipe Face Mesh."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import FeatureConfig, get_default_config
from src.utils.io import load_video


class FacialFeatureExtractor:
    """Extracts micro-expression features from video using MediaPipe Face Mesh."""

    def _get_mp_solutions(self):
        """Lazily load mediapipe to avoid global import errors."""
        try:
            import mediapipe as mp
            return mp.solutions
        except Exception as e:
            raise ImportError(
                "MediaPipe 'solutions' module not found. Original error: " + str(e)
            )

    _LEFT_EYE = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    _LEFT_BROW = [70, 63, 105, 66, 107]
    _RIGHT_BROW = [336, 296, 334, 293, 300]
    _NOSE = [1, 168, 98, 327]
    _HEAD_POINTS = [1, 33, 263, 61, 291, 199]
    _FEATURE_NAMES = [
        "eye_left_openness",
        "eye_right_openness",
        "mouth_openness",
        "mouth_width",
        "mouth_asymmetry",
        "left_brow_height",
        "right_brow_height",
        "brow_asymmetry",
        "brow_tension",
        "nose_wrinkle",
        "nostril_flare",
        "nose_asymmetry",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "head_displacement",
    ]

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the MediaPipe Face Mesh extractor.
        
        Gracefully handles missing MediaPipe dependencies by setting an 'available' flag.
        """

        self.config = config or get_default_config().features
        self.available = False
        self.face_mesh = None
        
        try:
            solutions = self._get_mp_solutions()
            self.mp_face_mesh = solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.available = True
        except (ImportError, AttributeError) as e:
            print(f"\u26a0\ufe0f WARNING: Facial features disabled. MediaPipe error: {e}")

    @staticmethod
    def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Return Euclidean distance between two landmark points."""
        return float(np.linalg.norm(point_a[:3] - point_b[:3]))

    @staticmethod
    def _face_diagonal(landmarks: np.ndarray) -> float:
        """Return face bounding-box diagonal for scale normalization."""
        if landmarks.size == 0:
            return 1.0
        mins = np.nanmin(landmarks[:, :2], axis=0)
        maxs = np.nanmax(landmarks[:, :2], axis=0)
        diagonal = float(np.linalg.norm(maxs - mins))
        return diagonal if diagonal > 1e-6 else 1.0

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        """Return a bounded ratio while avoiding division by zero."""
        if denominator <= 1e-8 or not np.isfinite(denominator):
            return 0.0
        value = numerator / denominator
        return float(np.clip(value, 0.0, 1.0))

    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run Face Mesh on a BGR frame and return normalized landmarks."""

        if not self.available or self.face_mesh is None or frame is None or frame.size == 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 5.0:
            # Extremely blurry frames can cause unstable micro-expression measurements.
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        height, width = frame.shape[:2]
        landmarks = np.array(
            [[point.x * width, point.y * height, point.z * width] for point in results.multi_face_landmarks[0].landmark],
            dtype=np.float32,
        )
        if self._face_diagonal(landmarks) < 25.0:
            # Very small faces do not provide reliable deception-relevant movements.
            return None
        return landmarks

    def extract_eye_openness(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """Calculate left and right eye openness ratios.

        Args:
            landmarks: Face Mesh landmarks with shape ``(468, 3)``.

        Returns:
            Tuple containing left and right Eye Aspect Ratio values.
        """

        def ear(indices: List[int]) -> float:
            p1, p2, p3, p4, p5, p6 = landmarks[indices]
            vertical = self._distance(p2, p6) + self._distance(p3, p5)
            horizontal = 2.0 * self._distance(p1, p4)
            # Blink rate and eye narrowing are behavioral cues often reviewed for stress or concealment.
            return self._safe_ratio(vertical, horizontal)

        return ear(self._LEFT_EYE), ear(self._RIGHT_EYE)

    def extract_mouth_movement(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate scale-normalized mouth movement features.

        Args:
            landmarks: Face Mesh landmarks with shape ``(468, 3)``.

        Returns:
            Dictionary containing mouth openness, width, and asymmetry.
        """

        diagonal = self._face_diagonal(landmarks)
        upper_lip, lower_lip = landmarks[13], landmarks[14]
        left_corner, right_corner = landmarks[61], landmarks[291]
        center = (upper_lip + lower_lip) / 2.0
        left_distance = self._distance(center, left_corner)
        right_distance = self._distance(center, right_corner)
        # Mouth tension, dry-mouth behavior, and articulation changes can accompany cognitive load.
        return {
            "mouth_openness": self._distance(upper_lip, lower_lip) / diagonal,
            "mouth_width": self._distance(left_corner, right_corner) / diagonal,
            "mouth_asymmetry": abs(left_distance - right_distance) / diagonal,
        }

    def extract_eyebrow_tension(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate eyebrow raise, asymmetry, and squeeze features.

        Args:
            landmarks: Face Mesh landmarks with shape ``(468, 3)``.

        Returns:
            Dictionary containing brow height, asymmetry, and tension.
        """

        diagonal = self._face_diagonal(landmarks)
        nose_bridge = landmarks[168]
        left_brow = landmarks[self._LEFT_BROW]
        right_brow = landmarks[self._RIGHT_BROW]
        left_height = float(np.mean(nose_bridge[1] - left_brow[:, 1]) / diagonal)
        right_height = float(np.mean(nose_bridge[1] - right_brow[:, 1]) / diagonal)
        inner_distance = self._distance(landmarks[107], landmarks[336]) / diagonal
        # Brow squeeze and asymmetry are micro-expression proxies for stress, concern, or suppression.
        return {
            "left_brow_height": left_height,
            "right_brow_height": right_height,
            "brow_asymmetry": abs(left_height - right_height),
            "brow_tension": max(0.0, 1.0 - inner_distance),
        }

    def extract_nose_movement(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate nose wrinkle, nostril flare, and asymmetry features.

        Args:
            landmarks: Face Mesh landmarks with shape ``(468, 3)``.

        Returns:
            Dictionary containing nose movement measurements.
        """

        diagonal = self._face_diagonal(landmarks)
        nose_tip, nose_bridge = landmarks[1], landmarks[168]
        left_nostril, right_nostril = landmarks[98], landmarks[327]
        center = (left_nostril + right_nostril) / 2.0
        # Nose wrinkling or nostril flare can indicate disgust, arousal, or regulated emotion.
        return {
            "nose_wrinkle": abs(float(nose_tip[1] - nose_bridge[1])) / diagonal,
            "nostril_flare": self._distance(left_nostril, right_nostril) / diagonal,
            "nose_asymmetry": abs(self._distance(center, left_nostril) - self._distance(center, right_nostril)) / diagonal,
        }

    def extract_head_dynamics(
        self, landmarks: np.ndarray, prev_landmarks: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate approximate head pose and inter-frame movement.

        Args:
            landmarks: Current Face Mesh landmarks with shape ``(468, 3)``.
            prev_landmarks: Optional previous frame landmarks for displacement.

        Returns:
            Dictionary containing pitch, yaw, roll, and displacement.
        """

        diagonal = self._face_diagonal(landmarks)
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]
        chin = landmarks[199]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]

        eye_center = (left_eye + right_eye) / 2.0
        mouth_center = (mouth_left + mouth_right) / 2.0
        eye_vector = right_eye - left_eye
        face_height = max(self._distance(eye_center, chin), 1e-6)
        face_width = max(self._distance(left_eye, right_eye), 1e-6)

        roll = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
        yaw = float(((nose_tip[0] - eye_center[0]) / face_width) * 45.0)
        pitch = float(((nose_tip[1] - mouth_center[1]) / face_height) * 45.0)
        displacement = 0.0
        if prev_landmarks is not None and prev_landmarks.shape == landmarks.shape:
            # Sudden head movement or freezing can be relevant when paired with verbal hesitation.
            displacement = float(np.mean(np.linalg.norm(landmarks[self._HEAD_POINTS] - prev_landmarks[self._HEAD_POINTS], axis=1)) / diagonal)

        return {
            "head_pitch": pitch,
            "head_yaw": yaw,
            "head_roll": roll,
            "head_displacement": displacement,
        }

    def process_frame(
        self, frame: np.ndarray, prev_landmarks: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, float]]:
        """Process a single BGR frame and extract all facial features.

        Args:
            frame: BGR video frame.
            prev_landmarks: Optional previous frame landmarks for dynamics.

        Returns:
            Combined feature dictionary, or ``None`` when no reliable face is detected.
        """

        landmarks = self._extract_landmarks(frame)
        if landmarks is None:
            return None

        left_eye, right_eye = self.extract_eye_openness(landmarks)
        features: Dict[str, float] = {
            "eye_left_openness": left_eye,
            "eye_right_openness": right_eye,
        }
        features.update(self.extract_mouth_movement(landmarks))
        features.update(self.extract_eyebrow_tension(landmarks))
        features.update(self.extract_nose_movement(landmarks))
        features.update(self.extract_head_dynamics(landmarks, prev_landmarks))
        features["_landmarks"] = landmarks  # Internal handoff for process_video only.
        return features

    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> pd.DataFrame:
        """Process a video and extract frame-level facial features.

        Args:
            video_path: Path to a video file.
            max_frames: Optional maximum number of frames to process.

        Returns:
            DataFrame with frame index, timestamp, and feature columns. Missing
            detections are represented with ``NaN`` feature values.
        """

        capture, metadata = load_video(video_path)
        fps = float(metadata.get("fps") or 30.0)
        total_frames = int(metadata.get("frame_count") or 0)
        limit = min(total_frames, max_frames) if max_frames is not None and total_frames else max_frames or total_frames
        records: List[Dict[str, float]] = []
        prev_landmarks: Optional[np.ndarray] = None

        try:
            progress = tqdm(total=limit if limit else None, desc="Extracting facial features")
            frame_idx = 0
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                record: Dict[str, float] = {"frame_idx": float(frame_idx), "timestamp": frame_idx / fps}
                features = self.process_frame(frame, prev_landmarks)
                if features is None:
                    record.update({name: np.nan for name in self.get_feature_names()})
                    prev_landmarks = None
                else:
                    landmarks = features.pop("_landmarks")
                    record.update(features)
                    prev_landmarks = landmarks
                records.append(record)
                frame_idx += 1
                progress.update(1)
            progress.close()
        finally:
            capture.release()

        columns = ["frame_idx", "timestamp"] + self.get_feature_names()
        return pd.DataFrame(records, columns=columns)

    def get_feature_names(self) -> List[str]:
        """Return ordered names of all produced facial features."""

        return list(self._FEATURE_NAMES)

    def close(self) -> None:
        """Release MediaPipe resources."""

        self.face_mesh.close()

