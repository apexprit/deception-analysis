"""Create and save a pre-trained deception detection model using synthetic data.

This script generates a realistic synthetic dataset, trains a deception classifier,
and saves it as a pre-trained model for demonstration purposes.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import AppConfig, get_default_config
from src.utils.synthetic_data import SyntheticDataGenerator
from src.model.classifier import DeceptionClassifier
from src.model.calibration import SubjectCalibrator
from src.explainability.explainer import DeceptionExplainer


def create_pretrained_model(output_dir: str = "./models", 
                           n_samples: int = 1000,
                           test_size: float = 0.2,
                           random_state: int = 42):
    """Create and save a pre-trained deception detection model.
    
    Args:
        output_dir: Directory to save model and related files
        n_samples: Total number of synthetic samples to generate
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    print("=" * 60)
    print("Creating Pre-trained Deception Detection Model")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    config = AppConfig()
    generator = SyntheticDataGenerator(config, seed=random_state)
    
    print(f"\n1. Generating synthetic dataset ({n_samples} samples)...")
    # Generate balanced dataset
    n_truthful = n_samples // 2
    n_deceptive = n_samples // 2
    
    features, labels, subject_ids = generator.generate_dataset(
        n_truthful=n_truthful,
        n_deceptive=n_deceptive
    )
    
    print(f"   Generated {len(features)} samples")
    print(f"   Truthful: {sum(labels == 0)}, Deceptive: {sum(labels == 1)}")
    print(f"   Features: {len(features.columns)}")
    print(f"   Unique subjects: {len(set(subject_ids))}")
    
    # Save dataset
    dataset_path = os.path.join(output_dir, "pretraining_dataset.csv")
    dataset_with_labels = features.copy()
    dataset_with_labels['label'] = labels
    dataset_with_labels['subject_id'] = subject_ids
    dataset_with_labels.to_csv(dataset_path, index=False)
    print(f"   Dataset saved to: {dataset_path}")
    
    print("\n2. Training deception classifier...")
    # Initialize and train classifier
    classifier = DeceptionClassifier(config.model)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train
    train_result = classifier.train(X_train, y_train, list(features.columns))
    
    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)
    
    print(f"   Training accuracy: {train_result['accuracy']:.3f}")
    print(f"   Test accuracy: {test_accuracy:.3f}")
    print(f"   Test F1-score: {test_f1:.3f}")
    print(f"   Test AUC-ROC: {test_auc:.3f}")
    
    print("\n3. Training subject calibrator...")
    # Train calibrator on all data
    calibrator = SubjectCalibrator(config.model)
    
    # Create subject profiles (in real scenario, would use baseline videos)
    subject_profiles = {}
    for subject_id in set(subject_ids):
        subject_mask = [sid == subject_id for sid in subject_ids]
        if sum(subject_mask) >= 5:  # Need enough samples per subject
            subject_features = features[subject_mask]
            calibrator.update_baseline(subject_id, subject_features)
            subject_profiles[subject_id] = {
                'n_samples': sum(subject_mask),
                'baseline_updated': True
            }
    
    print(f"   Created profiles for {len(subject_profiles)} subjects")
    
    print("\n4. Training explainability model...")
    # Train explainer on subset for efficiency
    explainer = DeceptionExplainer(config.model)
    
    # Use smaller subset for SHAP training (computationally expensive)
    if len(X_train) > 200:
        X_explain = X_train.iloc[:200]
    else:
        X_explain = X_train
    
    print(f"   Training SHAP explainer on {len(X_explain)} samples...")
    explainer.fit(classifier, X_explain)
    
    print("\n5. Saving model and components...")
    # Save classifier
    model_path = os.path.join(output_dir, "deception_model.pkl")
    classifier.save(model_path)
    print(f"   Classifier saved to: {model_path}")
    
    # Save calibrator
    calibrator_path = os.path.join(output_dir, "subject_calibrator.pkl")
    calibrator.save(calibrator_path)
    print(f"   Calibrator saved to: {calibrator_path}")
    
    # Save explainer
    explainer_path = os.path.join(output_dir, "deception_explainer.pkl")
    explainer.save(explainer_path)
    print(f"   Explainer saved to: {explainer_path}")
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(list(features.columns), f, indent=2)
    print(f"   Feature names saved to: {feature_names_path}")
    
    # Save model metadata
    metadata = {
        'model_name': 'DeceptionDetectionModel',
        'version': '1.0.0',
        'description': 'Pre-trained deception detection model using synthetic data',
        'training_date': pd.Timestamp.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(features.columns),
        'performance_metrics': {
            'training_accuracy': float(train_result['accuracy']),
            'training_f1': float(train_result['f1']),
            'test_accuracy': float(test_accuracy),
            'test_f1': float(test_f1),
            'test_auc_roc': float(test_auc)
        },
        'feature_groups': {
            'facial': generator.FACIAL_FEATURES,
            'audio': generator.AUDIO_FEATURES,
            'cross_modal': generator.CROSS_MODAL_FEATURES
        },
        'model_parameters': {
            'model_type': config.model.model_type,
            'random_state': random_state,
            'n_estimators': config.model.n_estimators,
            'learning_rate': config.model.learning_rate,
            'max_depth': config.model.max_depth
        }
    }
    
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved to: {metadata_path}")
    
    # Create a sample inference script
    inference_script = '''"""Sample inference with pre-trained deception model."""

import json
import pandas as pd
from src.model.classifier import DeceptionClassifier
from src.model.calibration import SubjectCalibrator
from src.explainability.explainer import DeceptionExplainer

# Load model and components
model = DeceptionClassifier.load("./models/deception_model.pkl")
calibrator = SubjectCalibrator.load("./models/subject_calibrator.pkl")
explainer = DeceptionExplainer.load("./models/deception_explainer.pkl")

# Load feature names
with open("./models/feature_names.json", "r") as f:
    feature_names = json.load(f)

print(f"Model loaded with {len(feature_names)} features")

# Example: Create sample features (replace with real data)
import numpy as np
sample_features = pd.DataFrame({
    feature: np.random.random() for feature in feature_names
}, index=[0])

# Predict
prediction = model.predict(sample_features)
probability = model.predict_proba(sample_features)[0, 1]

print(f"Prediction: {'Deceptive' if prediction[0] == 1 else 'Truthful'}")
print(f"Probability of deception: {probability:.3f}")

# If subject ID is known, calibrate
subject_id = "S001"
if subject_id in calibrator.subject_profiles:
    calibrated_prob = calibrator.calibrate(probability, subject_id)
    print(f"Calibrated probability: {calibrated_prob:.3f}")
'''
    
    inference_path = os.path.join(output_dir, "sample_inference.py")
    with open(inference_path, 'w') as f:
        f.write(inference_script)
    print(f"   Sample inference script: {inference_path}")
    
    print("\n" + "=" * 60)
    print("✅ Pre-trained model creation complete!")
    print("=" * 60)
    print(f"\nModel files saved to: {output_dir}")
    print("\nTo use the model:")
    print(f"  1. Load classifier: DeceptionClassifier.load('{model_path}')")
    print(f"  2. Load calibrator: SubjectCalibrator.load('{calibrator_path}')")
    print(f"  3. Load explainer: DeceptionExplainer.load('{explainer_path}')")
    print(f"\nOr use the pipeline:")
    print(f"  from src.pipeline import DeceptionPipeline")
    print(f"  pipeline = DeceptionPipeline()")
    print(f"  pipeline.load_model('{model_path}')")
    
    return {
        'model_path': model_path,
        'calibrator_path': calibrator_path,
        'explainer_path': explainer_path,
        'metadata_path': metadata_path,
        'dataset_path': dataset_path
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create pre-trained deception detection model")
    parser.add_argument("--output-dir", default="./models", help="Output directory for model files")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    create_pretrained_model(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        test_size=args.test_size,
        random_state=args.seed
    )