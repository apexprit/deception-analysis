#!/usr/bin/env python3
"""
CLI script for running deception analysis on new videos.

Usage:
    python inference.py --video ./data/interview.mp4 --model ./models/deception_model.pkl --subject_id S01 --visualize --output_dir ./results
    python inference.py --video_dir ./data/batch --model ./models/deception_model.pkl --subject_file subjects.csv --output_dir ./results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from src.pipeline import DeceptionPipeline
from src.utils.config import AppConfig, get_default_config


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(config_path: Optional[str]) -> AppConfig:
    """Load configuration from JSON file or use defaults."""
    if config_path is None:
        return get_default_config()
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = get_default_config()
        from dataclasses import fields
        for field in fields(config):
            if field.name in config_dict:
                setattr(config, field.name, config_dict[field.name])
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        return get_default_config()


def load_subject_mapping(subject_file: str) -> Dict[str, str]:
    """Load CSV mapping video_path -> subject_id."""
    df = pd.read_csv(subject_file)
    if "video_path" not in df.columns or "subject_id" not in df.columns:
        raise ValueError("CSV must contain columns 'video_path' and 'subject_id'")
    return dict(zip(df["video_path"], df["subject_id"]))


def find_videos(video_dir: str, extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]) -> List[str]:
    """Recursively find video files in directory."""
    video_dir = Path(video_dir)
    video_paths = []
    for ext in extensions:
        video_paths.extend(video_dir.rglob(f"*{ext}"))
    return sorted([str(p) for p in video_paths])


def print_result_summary(result: Dict[str, Any], threshold: float = 0.5) -> None:
    """Print a formatted summary of analysis result."""
    prob = result.get("deception_probability", 0.5)
    pred = result.get("prediction", "unknown")
    confidence = result.get("confidence", 0.0)
    calibrated = result.get("calibrated", False)
    print("\n" + "="*60)
    print("DECEPTION ANALYSIS RESULT")
    print("="*60)
    print(f"Prediction:          {pred.upper()}")
    print(f"Deception Probability: {prob:.3f} (threshold = {threshold})")
    print(f"Confidence:          {confidence:.3f}")
    if calibrated:
        print("Calibration:         Applied (subject-specific)")
    else:
        print("Calibration:         Not applied")
    # Key indicators
    temporal = result.get("temporal_analysis", {})
    if "suspicious_intervals" in temporal and temporal["suspicious_intervals"]:
        intervals = temporal["suspicious_intervals"]
        print(f"Suspicious Intervals: {len(intervals)} detected")
    if "spikes" in temporal and temporal["spikes"]:
        print(f"Probability Spikes:   {len(temporal['spikes'])}")
    # Feature importance
    explanation = result.get("explanation", {})
    if "top_features" in explanation:
        top = explanation["top_features"][:3]
        print("Top Contributing Features:")
        for feat, imp in top:
            print(f"  - {feat}: {imp:.3f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Run deception analysis on video(s).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video",
        help="Single video path to analyze.",
    )
    input_group.add_argument(
        "--video_dir",
        help="Directory containing videos to analyze (batch mode).",
    )

    # Model and config
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--config",
        help="Path to JSON config file (optional).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Deception probability threshold.",
    )

    # Subject identification
    parser.add_argument(
        "--subject_id",
        help="Subject identifier for calibration (single video mode).",
    )
    parser.add_argument(
        "--subject_file",
        help="CSV mapping video_path to subject_id (batch mode).",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory to save results and visualizations.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate all visualizations.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output.",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logging.error(f"Model file does not exist: {model_path}")
        sys.exit(1)

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)
    # Override decision threshold if provided
    config.model.decision_threshold = args.threshold
    logging.info(f"Decision threshold set to {args.threshold}")

    # Initialize pipeline
    pipeline = DeceptionPipeline(config)

    # Load pre-trained model
    try:
        pipeline.load_model(str(model_path))
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Determine video list and subject IDs
    video_paths = []
    subject_ids = None

    if args.video:
        video_paths = [args.video]
        if args.subject_id:
            subject_ids = [args.subject_id]
        elif args.subject_file:
            logging.warning("--subject_file ignored in single video mode. Use --subject_id.")
    else:  # video_dir
        video_paths = find_videos(args.video_dir)
        if not video_paths:
            logging.error(f"No video files found in {args.video_dir}")
            sys.exit(1)
        logging.info(f"Found {len(video_paths)} videos in {args.video_dir}")
        if args.subject_file:
            try:
                mapping = load_subject_mapping(args.subject_file)
                subject_ids = [mapping.get(Path(v).name, None) for v in video_paths]
                # If mapping uses relative paths, we could match by basename
                # For simplicity, we assume mapping uses video file names.
            except Exception as e:
                logging.error(f"Failed to load subject mapping: {e}")
                sys.exit(1)
        elif args.subject_id:
            logging.warning("--subject_id ignored in batch mode. Use --subject_file.")
            subject_ids = None

    # Run analysis
    results = []
    if len(video_paths) == 1:
        # Single video analysis
        video_path = video_paths[0]
        subject_id = subject_ids[0] if subject_ids else None
        logging.info(f"Analyzing video: {video_path}")
        try:
            result = pipeline.analyze_video(
                video_path=video_path,
                subject_id=subject_id,
                generate_visualizations=args.visualize,
                output_dir=str(output_dir),
            )
            results.append(result)
            # Print summary
            print_result_summary(result, args.threshold)
            # Generate report if requested
            if args.report:
                report_path = output_dir / f"{Path(video_path).stem}_report.json"
                pipeline.generate_report(result, str(report_path))
                logging.info(f"Report saved to {report_path}")
        except Exception as e:
            logging.error(f"Analysis failed for {video_path}: {e}")
            sys.exit(1)
    else:
        # Batch analysis
        logging.info(f"Starting batch analysis of {len(video_paths)} videos...")
        try:
            results = pipeline.analyze_batch(
                video_paths=video_paths,
                subject_ids=subject_ids,
                generate_visualizations=args.visualize,
                output_dir=str(output_dir),
            )
            logging.info("Batch analysis completed.")
        except Exception as e:
            logging.error(f"Batch analysis failed: {e}")
            sys.exit(1)

        # Print batch summary
        print("\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        deceptive_count = sum(1 for r in results if r.get("prediction") == "deceptive")
        total = len(results)
        print(f"Videos analyzed: {total}")
        print(f"Deceptive predictions: {deceptive_count} ({deceptive_count/total*100:.1f}%)")
        print(f"Truthful predictions: {total - deceptive_count}")
        avg_prob = sum(r.get("deception_probability", 0.5) for r in results) / total
        print(f"Average deception probability: {avg_prob:.3f}")
        print("="*60)

        # Save batch results to CSV
        batch_csv = output_dir / "batch_results.csv"
        rows = []
        for vid, res in zip(video_paths, results):
            rows.append({
                "video_path": vid,
                "prediction": res.get("prediction", "unknown"),
                "deception_probability": res.get("deception_probability", 0.5),
                "confidence": res.get("confidence", 0.0),
                "calibrated": res.get("calibrated", False),
            })
        df = pd.DataFrame(rows)
        df.to_csv(batch_csv, index=False)
        logging.info(f"Batch results saved to {batch_csv}")

        # Generate report for each video if requested
        if args.report:
            for vid, res in zip(video_paths, results):
                report_path = output_dir / f"{Path(vid).stem}_report.json"
                pipeline.generate_report(res, str(report_path))
            logging.info(f"Individual reports saved to {output_dir}")

    # Save overall results as JSON
    if results:
        results_json = output_dir / "analysis_results.json"
        # Convert DataFrames to dict for serialization (skip them)
        serializable = []
        for res in results:
            ser = {}
            for k, v in res.items():
                if isinstance(v, pd.DataFrame):
                    ser[k] = v.to_dict(orient="records")
                else:
                    ser[k] = v
            serializable.append(ser)
        with open(results_json, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logging.info(f"Full results saved to {results_json}")

    logging.info("Analysis completed successfully.")


if __name__ == "__main__":
    main()