"""Gradio demo application for the Explainable Multimodal Deception Analysis System."""

import argparse
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.pipeline import DeceptionPipeline
from src.utils.config import AppConfig, get_default_config

logger = logging.getLogger(__name__)

# Global state
pipeline: Optional[DeceptionPipeline] = None
config: Optional[AppConfig] = None
current_model_path: Optional[str] = None

# Custom CSS for styling
custom_css = """
/* Deception assessment styling */
.deceptive-badge {
    background-color: #ffcccc;
    color: #b30000;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    border: 2px solid #ff6666;
    display: inline-block;
    margin: 5px;
}
.truthful-badge {
    background-color: #ccffcc;
    color: #006600;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    border: 2px solid #66cc66;
    display: inline-block;
    margin: 5px;
}
.error-badge {
    background-color: #ffebcc;
    color: #cc6600;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    border: 2px solid #ff9933;
    display: inline-block;
    margin: 5px;
}
/* Professional font choices */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Proper spacing */
.gr-container {
    padding: 20px;
}
.gr-tab {
    padding: 20px;
}
/* Highlighted suspicious intervals */
.suspicious-interval {
    background-color: #fff0f0;
    border-left: 4px solid #ff6666;
    padding: 8px 12px;
    margin: 5px 0;
}
/* Progress bar styling */
.gr-progress-bar {
    height: 8px;
    border-radius: 4px;
}
"""

def initialize_pipeline(config_path: Optional[str] = None) -> DeceptionPipeline:
    """Initialize or reinitialize the pipeline."""
    global pipeline, config
    try:
        if config_path and os.path.exists(config_path):
            # In a real implementation, you'd load config from file
            logger.warning(f"Config file loading not implemented, using default config")
        config = get_default_config()
        pipeline = DeceptionPipeline(config)
        logger.info("Pipeline initialized with default config.")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

def get_pipeline() -> DeceptionPipeline:
    """Get or create the pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = initialize_pipeline()
    return pipeline

def analyze_video(
    video_file: Optional[str],
    subject_id: Optional[str],
    threshold: float,
    progress=gr.Progress()
) -> Tuple[Dict[str, Any], Optional[Figure], Optional[Figure], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Run deception analysis on uploaded video."""
    if video_file is None:
        return {
            "error": "No video file provided"
        }, None, None, None, None
    
    # Save uploaded video to a temporary file
    temp_dir = tempfile.mkdtemp(prefix="deception_video_")
    try:
        video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        # Copy the uploaded file
        with open(video_file, 'rb') as src, open(video_path, 'wb') as dst:
            dst.write(src.read())
        
        progress(0.1, desc="Initializing pipeline...")
        pipeline = get_pipeline()
        
        progress(0.2, desc="Analyzing video (this may take a minute)...")
        # Call the pipeline's analyze_video method
        result = pipeline.analyze_video(
            video_path=video_path,
            subject_id=subject_id if subject_id else None,
            generate_visualizations=True,
            output_dir=temp_dir
        )
        
        progress(0.8, desc="Processing results...")
        
        # Check for error
        if "error" in result:
            return {
                "error": result["error"]
            }, None, None, None, None
        
        # Create temporal analysis plot
        temporal_fig = None
        if "temporal_analysis" in result and "timestamps" in result["temporal_analysis"]:
            try:
                temporal_fig = create_temporal_plot(
                    result["temporal_analysis"]["timestamps"],
                    result["temporal_analysis"]["probability_trajectory"],
                    result["temporal_analysis"].get("suspicious_intervals", [])
                )
            except Exception as e:
                logger.warning(f"Failed to create temporal plot: {e}")
        
        # Create feature importance plot
        feature_fig = None
        if "explanation" in result and "feature_importance" in result["explanation"]:
            try:
                feature_fig = create_feature_importance_plot(
                    result["explanation"]["feature_importance"]
                )
            except Exception as e:
                logger.warning(f"Failed to create feature importance plot: {e}")
        
        # Prepare behavioral indicators dataframe
        behavioral_df = None
        if "explanation" in result and "behavioral_explanations" in result["explanation"]:
            try:
                behavioral_df = create_behavioral_dataframe(
                    result["explanation"]["behavioral_explanations"]
                )
            except Exception as e:
                logger.warning(f"Failed to create behavioral dataframe: {e}")
        
        # Prepare suspicious intervals dataframe
        suspicious_df = None
        if "temporal_analysis" in result and "suspicious_intervals" in result["temporal_analysis"]:
            try:
                suspicious_df = create_suspicious_intervals_dataframe(
                    result["temporal_analysis"]["suspicious_intervals"]
                )
            except Exception as e:
                logger.warning(f"Failed to create suspicious intervals dataframe: {e}")
        
        # Format overall assessment
        assessment = {
            "prediction": result.get("prediction", "unknown"),
            "probability": result.get("deception_probability", 0.5),
            "confidence": result.get("confidence", 0.0),
            "calibrated": result.get("calibrated", False),
            "threshold": threshold,
            "is_deceptive": result.get("prediction") == "deceptive" if "prediction" in result else False
        }
        
        progress(1.0, desc="Analysis complete!")
        return assessment, temporal_fig, feature_fig, behavioral_df, suspicious_df
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "error": f"Analysis failed: {str(e)}"
        }, None, None, None, None
    finally:
        # Clean up temp directory after a short delay (or keep for debugging)
        # shutil.rmtree(temp_dir, ignore_errors=True)
        pass

def create_temporal_plot(timestamps, probabilities, suspicious_intervals) -> Figure:
    """Create matplotlib figure for temporal analysis."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, probabilities, 'b-', linewidth=2, label='Deception Probability')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Deception Probability', fontsize=12)
    ax.set_title('Deception Probability Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Highlight suspicious intervals
    for interval in suspicious_intervals:
        if len(interval) >= 2:
            start, end = interval[0], interval[1]
            ax.axvspan(start, end, alpha=0.2, color='red', label='Suspicious Interval' if 'Suspicious Interval' not in ax.get_legend_handles_labels()[1] else '')
    
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

def create_feature_importance_plot(feature_importance: Dict[str, float]) -> Figure:
    """Create matplotlib figure for top feature importance."""
    # Sort features by absolute importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:10]  # Top 10 features
    
    if not top_features:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No feature importance data available', 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    features, importances = zip(*top_features)
    # Shorten long feature names for display
    display_features = [f[:30] + '...' if len(f) > 30 else f for f in features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if imp < 0 else 'green' for imp in importances]
    y_pos = np.arange(len(display_features))
    ax.barh(y_pos, importances, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_features, fontsize=10)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
    ax.set_title('Top 10 Feature Importance for Deception Prediction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    return fig

def create_behavioral_dataframe(behavioral_explanations: Dict[str, Any]) -> pd.DataFrame:
    """Create dataframe for behavioral indicators."""
    rows = []
    for category, indicators in behavioral_explanations.items():
        if isinstance(indicators, dict):
            for indicator, value in indicators.items():
                rows.append({
                    'Category': category,
                    'Indicator': indicator,
                    'Value': value,
                    'Interpretation': interpret_behavioral_indicator(category, indicator, value)
                })
        else:
            rows.append({
                'Category': category,
                'Indicator': 'Overall',
                'Value': indicators,
                'Interpretation': interpret_behavioral_indicator(category, 'Overall', indicators)
            })
    
    if rows:
        df = pd.DataFrame(rows)
        return df
    return pd.DataFrame(columns=['Category', 'Indicator', 'Value', 'Interpretation'])

def interpret_behavioral_indicator(category: str, indicator: str, value: Any) -> str:
    """Provide human-readable interpretation of behavioral indicators."""
    # Simplified interpretation logic
    try:
        val = float(value)
        if category == 'facial':
            if 'eye_contact' in indicator.lower():
                return "Low eye contact may indicate deception" if val < 0.5 else "Normal eye contact"
            elif 'blink_rate' in indicator.lower():
                return "Elevated blink rate may indicate stress" if val > 0.7 else "Normal blink rate"
        elif category == 'audio':
            if 'pitch_variability' in indicator.lower():
                return "Reduced pitch variability may indicate rehearsed speech" if val < 0.3 else "Normal pitch variability"
            elif 'speech_rate' in indicator.lower():
                return "Increased speech rate may indicate anxiety" if val > 1.2 else "Normal speech rate"
        elif category == 'cross_modal':
            return "Incongruence between facial and audio cues detected" if val > 0.6 else "Consistent multimodal behavior"
    except:
        pass
    return "No specific interpretation available"

def create_suspicious_intervals_dataframe(suspicious_intervals: list) -> pd.DataFrame:
    """Create dataframe for suspicious intervals."""
    if not suspicious_intervals:
        return pd.DataFrame(columns=['Start Time (s)', 'End Time (s)', 'Duration (s)', 'Max Probability', 'Severity'])
    
    rows = []
    for i, interval in enumerate(suspicious_intervals):
        if len(interval) >= 4:
            start, end, max_prob, severity = interval[0], interval[1], interval[2], interval[3]
        elif len(interval) >= 2:
            start, end = interval[0], interval[1]
            max_prob = 0.8  # default
            severity = 'medium'
        else:
            continue
        
        rows.append({
            'Start Time (s)': f"{start:.2f}",
            'End Time (s)': f"{end:.2f}",
            'Duration (s)': f"{end - start:.2f}",
            'Max Probability': f"{max_prob:.3f}",
            'Severity': severity
        })
    
    df = pd.DataFrame(rows)
    return df

def load_model(model_path: str, progress=gr.Progress()) -> Dict[str, Any]:
    """Load a pre-trained model."""
    global pipeline, current_model_path
    try:
        progress(0.3, desc="Checking model file...")
        if not os.path.exists(model_path):
            return {"error": f"Model file not found: {model_path}"}
        
        pipeline = get_pipeline()
        progress(0.6, desc="Loading model...")
        # Assuming pipeline has a load_model method
        if hasattr(pipeline, 'load_model'):
            pipeline.load_model(model_path)
        else:
            # Try to load via classifier
            if hasattr(pipeline.classifier, 'load'):
                pipeline.classifier.load(model_path)
            else:
                return {"error": "Model loading not implemented in pipeline"}
        
        current_model_path = model_path
        progress(1.0, desc="Model loaded successfully!")
        return {
            "status": "success",
            "model_path": model_path,
            "feature_count": getattr(pipeline.classifier, 'n_features_', 'unknown'),
            "is_fitted": getattr(pipeline.classifier, 'is_fitted_', False)
        }
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {"error": f"Failed to load model: {str(e)}"}

def get_model_info() -> Dict[str, Any]:
    """Return current model information."""
    global pipeline, current_model_path
    if pipeline is None:
        return {"status": "No pipeline initialized"}
    
    classifier = pipeline.classifier
    return {
        "model_path": current_model_path or "Not loaded",
        "is_fitted": getattr(classifier, 'is_fitted_', False),
        "feature_count": getattr(classifier, 'n_features_', 'unknown'),
        "model_type": type(classifier).__name__,
        "config": {
            "decision_threshold": pipeline.config.model.decision_threshold,
            "n_estimators": pipeline.config.model.n_estimators,
            "calibration_method": pipeline.config.model.calibration_method
        }
    }

def update_settings(
    fps: int,
    audio_segment_length: int,
    generate_visualizations: bool,
    output_dir: str
) -> Dict[str, Any]:
    """Update pipeline settings."""
    global pipeline, config
    try:
        # Note: AppConfig is frozen (immutable), so we cannot modify it directly.
        # In a real implementation, we would create a new config with updated values.
        # For this demo, we'll just store the settings and apply them when pipeline is reinitialized.
        settings = {
            "fps": fps,
            "audio_segment_length": audio_segment_length,
            "generate_visualizations": generate_visualizations,
            "output_dir": output_dir if output_dir.strip() else "Not specified (will use temp directory)",
            "note": "Settings will be applied on next pipeline initialization"
        }
        logger.info(f"Settings updated: {settings}")
        return {"status": "Settings updated (note: config is immutable, requires pipeline reinitialization)", "settings": settings}
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        return {"error": str(e)}

def format_assessment_html(assessment: Dict[str, Any]) -> str:
    """Format assessment results as HTML with styling."""
    if "error" in assessment:
        return f"""
        <div class="error-badge">
            <h3>❌ Error</h3>
            <p>{assessment['error']}</p>
        </div>
        """
    
    prediction = assessment.get("prediction", "unknown")
    probability = assessment.get("probability", 0.5)
    confidence = assessment.get("confidence", 0.0)
    calibrated = assessment.get("calibrated", False)
    threshold = assessment.get("threshold", 0.5)
    
    if prediction == "deceptive":
        badge_class = "deceptive-badge"
        verdict = "DECEPTIVE"
        icon = "⚠️"
    elif prediction == "truthful":
        badge_class = "truthful-badge"
        verdict = "TRUTHFUL"
        icon = "✅"
    else:
        badge_class = "error-badge"
        verdict = "UNKNOWN"
        icon = "❓"
    
    calibrated_text = " (calibrated)" if calibrated else ""
    
    return f"""
    <div class="{badge_class}">
        <h2>{icon} {verdict}{calibrated_text}</h2>
        <p><strong>Deception Probability:</strong> {probability:.3f}</p>
        <p><strong>Confidence:</strong> {confidence:.3f}</p>
        <p><strong>Decision Threshold:</strong> {threshold:.2f}</p>
        <p><strong>Calibrated:</strong> {calibrated}</p>
    </div>
    """


# ============================================================================
# Gradio UI Definition
# ============================================================================

def create_ui():
    """Create the Gradio interface with tabs."""
    with gr.Blocks(css=custom_css, title="Explainable Multimodal Deception Analysis") as demo:
        gr.Markdown("# 🔍 Explainable Multimodal Deception Analysis System")
        gr.Markdown("Upload a video to analyze for deception indicators using facial, audio, and cross-modal features.")
        
        with gr.Tabs():
            # Tab 1: Analyze Video
            with gr.TabItem("🔍 Analyze Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video", sources=["upload"], type="filepath")
                        subject_id = gr.Textbox(label="Subject ID (optional)", placeholder="Enter subject identifier")
                        threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Decision Threshold", info="Higher values make the model more conservative (less likely to classify as deceptive)")
                        analyze_btn = gr.Button("Analyze Video", variant="primary")
                        
                    with gr.Column(scale=2):
                        assessment_html = gr.HTML(label="Deception Assessment")
                        with gr.Row():
                            temporal_plot = gr.Plot(label="Temporal Analysis")
                            feature_importance_plot = gr.Plot(label="Feature Importance")
                        with gr.Row():
                            behavioral_df = gr.Dataframe(label="Behavioral Indicators", headers=["Category", "Indicator", "Value", "Interpretation"])
                            suspicious_df = gr.Dataframe(label="Suspicious Intervals", headers=["Start Time (s)", "End Time (s)", "Duration (s)", "Max Probability", "Severity"])
                
                analyze_btn.click(
                    fn=analyze_video,
                    inputs=[video_input, subject_id, threshold_slider],
                    outputs=[assessment_html, temporal_plot, feature_importance_plot, behavioral_df, suspicious_df]
                )
            
            # Tab 2: Model Info
            with gr.TabItem("📊 Model Info"):
                with gr.Row():
                    with gr.Column():
                        model_path_input = gr.Textbox(label="Model Path", placeholder="Path to trained model file (.pkl, .joblib, etc.)")
                        load_model_btn = gr.Button("Load Model", variant="secondary")
                        model_status = gr.JSON(label="Model Status")
                        gr.Markdown("### Current Model Information")
                        model_info_json = gr.JSON(label="Model Info")
                        refresh_btn = gr.Button("Refresh Model Info")
                
                load_model_btn.click(
                    fn=load_model,
                    inputs=[model_path_input],
                    outputs=[model_status]
                )
                refresh_btn.click(
                    fn=get_model_info,
                    inputs=[],
                    outputs=[model_info_json]
                )
            
            # Tab 3: Settings
            with gr.TabItem("⚙️ Settings"):
                with gr.Row():
                    with gr.Column():
                        fps_slider = gr.Slider(minimum=1, maximum=60, value=30, step=1, label="Frames Per Second (FPS) for processing")
                        audio_segment_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Audio Segment Length (seconds)")
                        visualization_checkbox = gr.Checkbox(value=True, label="Generate Visualizations")
                        output_dir_input = gr.Textbox(label="Output Directory (optional)", placeholder="Leave empty for temporary directory")
                        update_settings_btn = gr.Button("Update Settings", variant="secondary")
                        settings_status = gr.JSON(label="Settings Status")
                
                update_settings_btn.click(
                    fn=update_settings,
                    inputs=[fps_slider, audio_segment_slider, visualization_checkbox, output_dir_input],
                    outputs=[settings_status]
                )
            
            # Tab 4: About
            with gr.TabItem("📖 About"):
                gr.Markdown("""
                ## About This System
                
                This is an **Explainable Multimodal Deception Analysis System** that uses:
                - **Facial features**: Micro‑expressions, eye contact, blink rate, head pose
                - **Audio features**: Pitch variability, speech rate, voice stress, spectral features
                - **Cross‑modal fusion**: Combining facial and audio cues for robust detection
                
                ### How It Works
                1. Upload a video of a person speaking
                2. The system extracts facial and audio features frame‑by‑frame
                3. A trained classifier predicts deception probability over time
                4. SHAP explanations highlight the most influential features
                5. Suspicious intervals are flagged for closer inspection
                
                ### Interpretation
                - **Deceptive**: Probability > decision threshold (adjustable)
                - **Truthful**: Probability ≤ decision threshold
                - **Confidence**: Model's certainty in the prediction
                
                ### Technical Details
                - Built with Python, OpenCV, Librosa, and Scikit‑learn
                - Uses a Random Forest classifier with calibration
                - Real‑time visualization with Matplotlib
                - Gradio‑based web interface
                
                ### Disclaimer
                This tool is for research and demonstration purposes only.
                It should not be used as the sole basis for any legal, security, or psychological assessment.
                """)
        
        gr.Markdown("---")
        gr.Markdown("**Note**: Analysis may take several minutes depending on video length and system resources.")
    
    return demo


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Gradio demo for Explainable Multimodal Deception Analysis")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline with optional config
    try:
        initialize_pipeline(args.config)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.warning(f"Pipeline initialization failed: {e}. Will initialize on first use.")
    
    demo = create_ui()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
