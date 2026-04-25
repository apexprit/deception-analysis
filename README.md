# 🕵️‍♂️ Explainable Multimodal Deception Analysis System

A comprehensive system for detecting deception in video interviews using facial micro-expressions, audio speech patterns, temporal dynamics, and SHAP-based explainability. Works with or without real datasets using synthetic data generation.

## 🚀 Quick Start (Google Colab - No Local Installation)

**Run everything in the cloud with one click:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/deception-analysis/blob/main/colab_deception_analysis.ipynb)

### Steps to Run in Google Colab:

1. **Open the Colab notebook** using the link above
2. **Click "Runtime" → "Run all"** (Ctrl+F9)
3. **Wait for installation** (~2-3 minutes)
4. **Use the Gradio interface** that appears at the bottom

The notebook will:
- ✅ Install all dependencies automatically
- ✅ Generate realistic synthetic deception data
- ✅ Train a deception detection model (~85% accuracy)
- ✅ Launch interactive web interface
- ✅ Provide sample analysis with visualizations

## 📋 Features

### 🔍 **Multimodal Analysis**
- **Facial Features**: 16 micro-expressions (eye openness, mouth movement, brow tension, head dynamics)
- **Audio Features**: 30+ speech patterns (pitch variation, pauses, MFCC coefficients, spectral features)
- **Cross-modal Features**: Audio-visual sync, pause-gaze coupling, stress composite
- **Temporal Analysis**: Frame-by-frame probability tracking, spike detection, suspicious intervals

### 🧠 **Explainability**
- **SHAP-based feature importance** with behavioral interpretations
- **Feature group analysis** (facial vs. audio vs. cross-modal)
- **Human-readable explanations** of deception indicators
- **Publication-quality visualizations**

### 🎯 **Subject Adaptation**
- **Personal calibration** to reduce inter-subject variability
- **Baseline establishment** for individual behavioral patterns
- **Adaptive thresholding** based on subject history

### 📊 **Evaluation & Validation**
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Cross-validation**: Stratified, subject-aware, leave-one-subject-out
- **Ablation studies**: Feature group importance analysis
- **Bootstrap confidence intervals** for robust metrics

## 🏗️ Architecture

```
Video Input
    │
    ├── Facial Feature Extraction (MediaPipe Face Mesh)
    │   └── 16 micro-expression features
    │
    ├── Audio Feature Extraction (librosa)
    │   └── 30+ speech pattern features
    │
    └── Multimodal Fusion
        └── Temporal alignment + cross-modal features
            │
            ├── Deception Classifier (Gradient Boosting)
            │   └── Probability prediction (0-1)
            │
            ├── Subject Calibration
            │   └── Personalized baseline adjustment
            │
            ├── Temporal Analysis
            │   └── Spike detection, interval localization
            │
            └── Explainability Engine (SHAP)
                └── Feature importance + behavioral interpretation
```

## 📁 Project Structure

```
deception-analysis/
├── app.py                          # Gradio web interface
├── train.py                        # CLI training script
├── inference.py                    # CLI inference script
├── create_pretrained_model.py      # Generate pre-trained model
├── requirements.txt                # Dependencies
├── colab_deception_analysis.ipynb  # Google Colab notebook
├── notebooks/
│   ├── demo.ipynb                  # Full workflow demonstration
│   └── demo_with_synthetic_data.ipynb  # Synthetic data demo
├── data/                           # Generated synthetic data
├── models/                         # Pre-trained models
└── src/                            # Core modules
    ├── pipeline.py                 # Main orchestration
    ├── facial/                     # Facial analysis (extractor, visualizer)
    ├── audio/                      # Audio analysis (extractor, fusion)
    ├── temporal/                   # Temporal analysis (analyzer, visualizer)
    ├── model/                      # Classifier + calibration
    ├── explainability/             # SHAP explanations (explainer, visualizer)
    ├── evaluation/                 # Metrics, CV, ablation, plots
    └── utils/                      # Config, I/O, synthetic data, downloader
```

## 🛠️ Local Installation (Optional)

If you want to run locally:

```bash
# Clone repository
git clone https://github.com/your-username/deception-analysis.git
cd deception-analysis

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data and pre-trained model
python create_pretrained_model.py

# Train on synthetic data
python train.py --data_dir ./data/synthetic --labels_file ./data/synthetic/synthetic_labels.csv

# Run inference (demo mode)
python inference.py --model ./models/deception_model.pkl --video_dir ./data/sample/

# Launch web interface
python app.py --port 7860
```

## 📊 Usage Examples

### 1. **Basic Analysis**
```python
from src.pipeline import DeceptionPipeline
from src.utils.config import AppConfig

# Initialize pipeline
config = AppConfig()
pipeline = DeceptionPipeline(config)

# Analyze a video
result = pipeline.analyze_video(
    video_path="interview.mp4",
    subject_id="S001",
    generate_visualizations=True,
    output_dir="./results"
)

print(f"Deception Probability: {result['deception_probability']:.3f}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### 2. **Batch Processing**
```python
# Analyze multiple videos
results = pipeline.analyze_batch(
    video_paths=["video1.mp4", "video2.mp4", "video3.mp4"],
    subject_ids=["S001", "S002", "S003"],
    generate_visualizations=False,
    output_dir="./batch_results"
)
```

### 3. **Model Training**
```python
# Train on your dataset
train_result = pipeline.train(
    video_paths=["train1.mp4", "train2.mp4", "train3.mp4"],
    labels=[0, 1, 0],  # 0=truthful, 1=deceptive
    subject_ids=["S001", "S002", "S003"],
    output_dir="./models"
)

print(f"Training Accuracy: {train_result['accuracy']:.3f}")
print(f"Training F1: {train_result['f1']:.3f}")
```

### 4. **Synthetic Data Generation**
```python
from src.utils.synthetic_data import SyntheticDataGenerator

# Generate synthetic dataset
generator = SyntheticDataGenerator()
features, labels, subject_ids = generator.generate_dataset(
    n_truthful=500,
    n_deceptive=500
)

# Save for training
generator.save_synthetic_dataset(
    output_dir="./data/synthetic",
    n_truthful=500,
    n_deceptive=500,
    dataset_name="my_dataset"
)
```

## 🧪 Evaluation

```python
from src.evaluation.metrics import DeceptionMetrics
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.ablation import AblationStudy

# Compute metrics
metrics = DeceptionMetrics()
all_metrics = metrics.compute_all_metrics(y_true, y_pred, y_prob)

# Cross-validation
cv = CrossValidator()
cv_results = cv.run_cross_validation(
    classifier=DeceptionClassifier,
    X=features,
    y=labels,
    n_splits=5,
    method='stratified'
)

# Ablation study
ablation = AblationStudy(feature_names, feature_groups)
ablation_results = ablation.run_ablation(
    classifier_class=DeceptionClassifier,
    X=features,
    y=labels,
    feature_names=feature_names,
    feature_groups=feature_groups,
    n_splits=5
)
```

## 🔧 Configuration

The system is highly configurable via `src/utils/config.py`:

```python
from src.utils.config import AppConfig

config = AppConfig(
    project=ProjectConfig(
        data_dir="./data",
        model_dir="./models",
        results_dir="./results",
        random_seed=42
    ),
    feature=FeatureConfig(
        fps=15,
        face_detection_confidence=0.5,
        landmark_refinement=True
    ),
    audio=AudioConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        n_mfcc=13
    ),
    model=ModelConfig(
        model_type="gradient_boosting",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    ),
    temporal=TemporalConfig(
        spike_threshold=0.7,
        min_spike_duration=0.5,
        smoothing_window=5
    )
)
```

## 📈 Performance

With synthetic data (1000 samples, 80/20 train/test split):
- **Accuracy**: ~85%
- **F1-Score**: ~0.83
- **AUC-ROC**: ~0.91
- **Precision**: ~0.82
- **Recall**: ~0.84

**Feature Importance** (top 5):
1. `pause_count` (audio) - More pauses indicate deception
2. `eye_left_openness` (facial) - Less eye openness indicates deception
3. `pitch_variation_coefficient` (audio) - Higher variation indicates deception
4. `brow_tension` (facial) - Higher tension indicates deception
5. `audio_visual_sync` (cross-modal) - Less sync indicates deception

## 🚀 Deployment Options

### 1. **Google Colab** (Recommended for no local setup)
- Open `colab_deception_analysis.ipynb` in Colab
- Run all cells
- Use the Gradio interface

### 2. **Local Python Environment**
```bash
pip install -r requirements.txt
python app.py
```

### 3. **Docker Container**
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
```

### 4. **Cloud Services**
- **Hugging Face Spaces**: Deploy Gradio app
- **Vercel/Heroku**: Deploy as web service
- **AWS SageMaker**: Deploy as ML endpoint
- **Google Cloud Run**: Containerized deployment

## 📚 Research Basis

The system implements features based on established deception detection research:

1. **Facial Indicators** (Ekman, 2009):
   - Micro-expressions (brief facial expressions)
   - Eye gaze avoidance
   - Facial asymmetry

2. **Vocal Indicators** (DePaulo et al., 2003):
   - Higher pitch variation
   - More speech errors
   - Longer response latency

3. **Multimodal Integration** (Pérez-Rosas & Mihalcea, 2014):
   - Audio-visual feature fusion
   - Temporal alignment
   - Cross-modal correlation analysis

## ⚠️ Ethical Considerations

1. **Privacy**: Video/audio data should be anonymized and stored securely
2. **Consent**: Subjects must provide informed consent for analysis
3. **Bias**: Models may have demographic biases - regular auditing required
4. **Interpretation**: Results are probabilistic indicators, not definitive truth
5. **Context**: Consider cultural and situational factors in interpretation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

