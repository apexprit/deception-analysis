"""Input/output helpers for videos, audio extraction, models, and results."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import cv2
import joblib


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        Path: The created or existing directory path.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_video(path: PathLike) -> Tuple[cv2.VideoCapture, Dict[str, Union[float, int]]]:
    """Load a video file with OpenCV and return its capture object and metadata.

    Args:
        path: Path to the input video file.

    Returns:
        Tuple[cv2.VideoCapture, Dict[str, Union[float, int]]]: OpenCV capture object
        and metadata containing fps, frame_count, width, and height.

    Raises:
        FileNotFoundError: If the video path does not exist.
        ValueError: If OpenCV cannot open the video file.
    """

    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Unable to open video file: {video_path}")

    metadata: Dict[str, Union[float, int]] = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    return cap, metadata


def extract_audio_from_video(video_path: PathLike, output_path: PathLike) -> Path:
    """Extract an audio track from a video using ffmpeg.

    Args:
        video_path: Path to the source video file.
        output_path: Path where the extracted audio file should be written.

    Returns:
        Path: Path to the extracted audio file.

    Raises:
        FileNotFoundError: If the source video path does not exist.
        RuntimeError: If ffmpeg fails to extract the audio track.
    """

    source = Path(video_path)
    destination = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    ensure_dir(destination.parent)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "22050",
        "-ac",
        "1",
        str(destination),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed for {source}: {process.stderr.strip()}"
        )
    return destination


def save_model(model: Any, path: PathLike) -> Path:
    """Serialize a scikit-learn compatible model with joblib.

    Args:
        model: Fitted model or pipeline object to serialize.
        path: Destination file path.

    Returns:
        Path: Path to the saved model artifact.
    """

    model_path = Path(path)
    ensure_dir(model_path.parent)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: PathLike) -> Any:
    """Load a scikit-learn compatible model serialized with joblib.

    Args:
        path: Path to the serialized model artifact.

    Returns:
        Any: Loaded model object.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
    """

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def save_results(results: Mapping[str, Any], path: PathLike) -> Path:
    """Save a results mapping as a JSON file.

    Args:
        results: JSON-serializable results dictionary or mapping.
        path: Destination JSON file path.

    Returns:
        Path: Path to the saved results file.
    """

    results_path = Path(path)
    ensure_dir(results_path.parent)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(dict(results), file, indent=2, sort_keys=True, default=str)
    return results_path
