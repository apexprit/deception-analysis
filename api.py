import os
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import DeceptionPipeline
from src.utils.config import get_default_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deception Analysis API", 
    description="API for Explainable Multimodal Deception Analysis for Mobile Apps"
)

# Enable CORS so the Flutter app can connect easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        logger.info("Initializing Deception ML Pipeline...")
        config = get_default_config()
        pipeline = DeceptionPipeline(config)
        
        # Load the pre-trained model if it exists
        model_path = os.path.join("models", "deception_model.pkl")
        if os.path.exists(model_path):
            pipeline.load_model(model_path)
            logger.info("Pre-trained model loaded successfully.")
        else:
            logger.warning(f"No pre-trained model found at {model_path}. You must generate it first!")
            
        logger.info("Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")

@app.post("/analyze")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    subject_id: Optional[str] = Form(None)
):
    if not pipeline:
        raise HTTPException(status_code=500, detail="ML Pipeline not initialized yet")

    # Create temporary directory for the uploaded video
    temp_dir = tempfile.mkdtemp(prefix="deception_api_")
    video_path = os.path.join(temp_dir, file.filename)

    try:
        # Save uploaded video to disk
        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Analyzing video: {file.filename} (Subject: {subject_id})")

        # Run pipeline
        result = pipeline.analyze_video(
            video_path=video_path,
            subject_id=subject_id,
            generate_visualizations=False, # We don't need Matplotlib images, just JSON data
            output_dir=temp_dir
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Format clean JSON response for the Flutter app
        response_data = {
            "status": "success",
            "prediction": result.get("prediction", "unknown"),
            "verdict": result.get("prediction", "unknown"), # added for Flutter
            "probability": result.get("deception_probability", 0.5),
            "confidence": result.get("confidence", 0.0),
            "is_deceptive": result.get("prediction") == "deceptive",
        }

        # Add temporal analysis (to draw the timeline graph in Flutter)
        if "temporal_analysis" in result:
            response_data["temporal"] = {
                "timestamps": result["temporal_analysis"].get("timestamps", []),
                "probabilities": result["temporal_analysis"].get("probability_trajectory", []),
                "suspicious_intervals": result["temporal_analysis"].get("suspicious_intervals", [])
            }

        # Add SHAP explainability (to draw the Feature Importance chart in Flutter)
        if "explanation" in result:
            fi = result["explanation"].get("feature_importance", {})
            top_indicators = [{"feature": k, "impact": v} for k, v in list(fi.items())[:5]]
            
            response_data["explanation"] = {
                "top_indicators": top_indicators,
                "feature_importance": fi,
                "behavioral_explanations": result["explanation"].get("behavioral_explanations", {})
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up the heavy video file to save server storage
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            os.rmdir(temp_dir)
        except:
            pass

@app.get("/health")
async def health_check():
    return {"status": "online", "pipeline_ready": pipeline is not None}
