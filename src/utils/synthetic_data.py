"""Synthetic data generator for deception analysis system.

This module generates realistic synthetic feature data for training and
demonstrating the deception analysis system when real datasets are unavailable.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import AppConfig, get_default_config


class SyntheticDataGenerator:
    """Generates synthetic facial, audio, and fused features for deception analysis.
    
    The generator creates realistic feature distributions based on research findings
    about deception indicators:
    - Deceptive subjects: higher eye blink rate, less eye contact, more speech pauses,
      higher pitch variation, increased head movement, more facial asymmetry
    - Truthful subjects: more stable gaze, smoother speech, lower pitch variation,
      symmetrical facial expressions, natural head movements
    """
    
    # Feature groups based on actual system
    FACIAL_FEATURES = [
        "eye_left_openness", "eye_right_openness", "mouth_openness", "mouth_width",
        "mouth_asymmetry", "left_brow_height", "right_brow_height", "brow_asymmetry",
        "brow_tension", "nose_wrinkle", "nostril_flare", "nose_asymmetry",
        "head_pitch", "head_yaw", "head_roll", "head_displacement"
    ]
    
    AUDIO_FEATURES = [
        "pitch_mean", "pitch_std", "pitch_range", "pitch_variation_coefficient",
        "voiced_ratio", "pause_count", "avg_pause_duration", "max_pause_duration",
        "pause_rate", "speech_ratio", "energy_mean", "energy_std", "energy_range",
        "spectral_centroid_mean", "spectral_centroid_std", "spectral_rolloff_mean",
        "spectral_rolloff_std", "zero_crossing_rate_mean", "zero_crossing_rate_std",
        "mfcc1_mean", "mfcc1_std", "mfcc2_mean", "mfcc2_std", "mfcc3_mean",
        "mfcc3_std", "mfcc4_mean", "mfcc4_std", "mfcc5_mean", "mfcc5_std",
        "mfcc6_mean", "mfcc6_std", "mfcc7_mean", "mfcc7_std", "mfcc8_mean",
        "mfcc8_std", "mfcc9_mean", "mfcc9_std", "mfcc10_mean", "mfcc10_std",
        "mfcc11_mean", "mfcc11_std", "mfcc12_mean", "mfcc12_std", "mfcc13_mean",
        "mfcc13_std"
    ]
    
    CROSS_MODAL_FEATURES = [
        "audio_visual_sync", "pause_gaze_coupling", "stress_composite",
        "verbal_visual_consistency", "response_latency"
    ]
    
    def __init__(self, config: Optional[AppConfig] = None, seed: int = 42):
        """Initialize synthetic data generator.
        
        Args:
            config: Application configuration for feature ranges
            seed: Random seed for reproducibility
        """
        self.config = config or get_default_config()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Define realistic feature ranges based on research literature
        self._init_feature_ranges()
        
    def _init_feature_ranges(self):
        """Initialize realistic feature value ranges."""
        # Facial features (normalized 0-1 unless otherwise noted)
        self.facial_ranges = {
            "eye_left_openness": (0.3, 0.9),  # eye openness ratio
            "eye_right_openness": (0.3, 0.9),
            "mouth_openness": (0.1, 0.7),  # mouth aperture
            "mouth_width": (0.2, 0.8),  # mouth width ratio
            "mouth_asymmetry": (0.0, 0.3),  # left-right asymmetry
            "left_brow_height": (0.2, 0.8),  # brow position
            "right_brow_height": (0.2, 0.8),
            "brow_asymmetry": (0.0, 0.4),  # brow asymmetry
            "brow_tension": (0.0, 0.6),  # brow muscle tension
            "nose_wrinkle": (0.0, 0.5),  # nose wrinkle intensity
            "nostril_flare": (0.0, 0.4),  # nostril dilation
            "nose_asymmetry": (0.0, 0.3),  # nose asymmetry
            "head_pitch": (-0.2, 0.2),  # head pitch in radians
            "head_yaw": (-0.3, 0.3),  # head yaw in radians
            "head_roll": (-0.15, 0.15),  # head roll in radians
            "head_displacement": (0.0, 0.5),  # head movement magnitude
        }
        
        # Audio features (real-world units)
        self.audio_ranges = {
            "pitch_mean": (100.0, 250.0),  # Hz
            "pitch_std": (5.0, 50.0),  # Hz
            "pitch_range": (20.0, 150.0),  # Hz
            "pitch_variation_coefficient": (0.05, 0.5),  # std/mean
            "voiced_ratio": (0.6, 0.95),  # proportion
            "pause_count": (2.0, 20.0),  # count
            "avg_pause_duration": (0.3, 2.5),  # seconds
            "max_pause_duration": (0.5, 5.0),  # seconds
            "pause_rate": (0.5, 8.0),  # pauses per minute
            "speech_ratio": (0.4, 0.9),  # proportion
            "energy_mean": (0.01, 0.5),  # RMS energy
            "energy_std": (0.005, 0.2),  # RMS energy std
            "energy_range": (0.02, 0.8),  # RMS energy range
            "spectral_centroid_mean": (1000.0, 4000.0),  # Hz
            "spectral_centroid_std": (200.0, 1500.0),  # Hz
            "spectral_rolloff_mean": (2000.0, 8000.0),  # Hz
            "spectral_rolloff_std": (500.0, 3000.0),  # Hz
            "zero_crossing_rate_mean": (0.05, 0.3),  # proportion
            "zero_crossing_rate_std": (0.01, 0.1),  # proportion
            # MFCC coefficients (mean and std)
            "mfcc1_mean": (-500.0, 500.0),
            "mfcc1_std": (50.0, 300.0),
            "mfcc2_mean": (-300.0, 300.0),
            "mfcc2_std": (30.0, 200.0),
            "mfcc3_mean": (-200.0, 200.0),
            "mfcc3_std": (20.0, 150.0),
            "mfcc4_mean": (-150.0, 150.0),
            "mfcc4_std": (15.0, 100.0),
            "mfcc5_mean": (-100.0, 100.0),
            "mfcc5_std": (10.0, 80.0),
            "mfcc6_mean": (-80.0, 80.0),
            "mfcc6_std": (8.0, 60.0),
            "mfcc7_mean": (-60.0, 60.0),
            "mfcc7_std": (6.0, 40.0),
            "mfcc8_mean": (-40.0, 40.0),
            "mfcc8_std": (4.0, 30.0),
            "mfcc9_mean": (-30.0, 30.0),
            "mfcc9_std": (3.0, 20.0),
            "mfcc10_mean": (-20.0, 20.0),
            "mfcc10_std": (2.0, 15.0),
            "mfcc11_mean": (-15.0, 15.0),
            "mfcc11_std": (1.5, 10.0),
            "mfcc12_mean": (-10.0, 10.0),
            "mfcc12_std": (1.0, 8.0),
            "mfcc13_mean": (-8.0, 8.0),
            "mfcc13_std": (0.8, 6.0),
        }
        
        # Cross-modal features
        self.cross_modal_ranges = {
            "audio_visual_sync": (0.3, 0.95),  # correlation coefficient
            "pause_gaze_coupling": (0.1, 0.8),  # probability
            "stress_composite": (0.0, 1.0),  # composite score
            "verbal_visual_consistency": (0.2, 0.9),  # consistency score
            "response_latency": (0.5, 3.5),  # seconds
        }
        
    def _apply_deception_shift(self, features: Dict[str, float], label: int) -> Dict[str, float]:
        """Apply deception-specific shifts to features based on label.
        
        Research-based deception indicators:
        - Deceptive: higher pitch variation, more pauses, less eye openness,
          more facial asymmetry, increased head movement, higher brow tension
        - Truthful: more stable features, symmetrical expressions, natural patterns
        """
        if label == 0:  # Truthful - minimal shifts
            return features
            
        # Deceptive (label=1) - apply systematic shifts
        shifted = features.copy()
        
        # Facial feature shifts for deception
        if "eye_left_openness" in shifted:
            shifted["eye_left_openness"] *= 0.8  # less eye openness
            shifted["eye_right_openness"] *= 0.8
        if "mouth_asymmetry" in shifted:
            shifted["mouth_asymmetry"] *= 1.5  # more asymmetry
        if "brow_asymmetry" in shifted:
            shifted["brow_asymmetry"] *= 1.6
        if "brow_tension" in shifted:
            shifted["brow_tension"] *= 1.8  # more brow tension
        if "nose_wrinkle" in shifted:
            shifted["nose_wrinkle"] *= 1.4  # more nose wrinkling
        if "head_displacement" in shifted:
            shifted["head_displacement"] *= 1.7  # more head movement
            
        # Audio feature shifts for deception
        if "pitch_std" in shifted:
            shifted["pitch_std"] *= 1.5  # more pitch variation
        if "pitch_variation_coefficient" in shifted:
            shifted["pitch_variation_coefficient"] *= 1.6
        if "pause_count" in shifted:
            shifted["pause_count"] *= 1.8  # more pauses
        if "avg_pause_duration" in shifted:
            shifted["avg_pause_duration"] *= 1.3  # longer pauses
        if "pause_rate" in shifted:
            shifted["pause_rate"] *= 1.7  # higher pause rate
        if "speech_ratio" in shifted:
            shifted["speech_ratio"] *= 0.8  # less speech
        
        # Cross-modal feature shifts
        if "audio_visual_sync" in shifted:
            shifted["audio_visual_sync"] *= 0.7  # less sync
        if "verbal_visual_consistency" in shifted:
            shifted["verbal_visual_consistency"] *= 0.6  # less consistency
        if "stress_composite" in shifted:
            shifted["stress_composite"] *= 1.8  # higher stress
            
        return shifted
    
    def generate_facial_features(self, n_samples: int = 100, label: int = 0) -> pd.DataFrame:
        """Generate synthetic facial features.
        
        Args:
            n_samples: Number of samples to generate
            label: 0 for truthful, 1 for deceptive
            
        Returns:
            DataFrame with facial features
        """
        data = {}
        for feature, (low, high) in self.facial_ranges.items():
            # Generate base distribution
            if "asymmetry" in feature or "tension" in feature or "wrinkle" in feature:
                # Skewed distribution for asymmetry/tension features
                base = self.rng.beta(2, 5, n_samples) * (high - low) + low
            else:
                # Normal distribution for most features
                mean = (low + high) / 2
                std = (high - low) / 6  # 99.7% within range
                base = self.rng.normal(mean, std, n_samples)
                # Clip to range
                base = np.clip(base, low, high)
            
            data[feature] = base
            
        df = pd.DataFrame(data)
        
        # Apply deception shifts sample-wise
        if label == 1:
            for i in range(n_samples):
                row = df.iloc[i].to_dict()
                shifted = self._apply_deception_shift(row, label)
                df.iloc[i] = pd.Series(shifted)
                
        return df
    
    def generate_audio_features(self, n_samples: int = 100, label: int = 0) -> pd.DataFrame:
        """Generate synthetic audio features.
        
        Args:
            n_samples: Number of samples to generate
            label: 0 for truthful, 1 for deceptive
            
        Returns:
            DataFrame with audio features
        """
        data = {}
        for feature, (low, high) in self.audio_ranges.items():
            # Different distributions for different feature types
            if "std" in feature or "range" in feature or "variation" in feature:
                # Positive skew for variability measures
                base = self.rng.gamma(shape=2, scale=(high-low)/4, size=n_samples) + low
            elif "ratio" in feature or "rate" in feature:
                # Beta distribution for ratios/rates
                base = self.rng.beta(3, 3, n_samples) * (high - low) + low
            else:
                # Normal distribution for most features
                mean = (low + high) / 2
                std = (high - low) / 6
                base = self.rng.normal(mean, std, n_samples)
            
            # Clip to reasonable ranges
            base = np.clip(base, low * 0.9, high * 1.1)
            data[feature] = base
            
        df = pd.DataFrame(data)
        
        # Apply deception shifts
        if label == 1:
            for i in range(n_samples):
                row = df.iloc[i].to_dict()
                shifted = self._apply_deception_shift(row, label)
                df.iloc[i] = pd.Series(shifted)
                
        return df
    
    def generate_cross_modal_features(self, n_samples: int = 100, label: int = 0) -> pd.DataFrame:
        """Generate synthetic cross-modal features.
        
        Args:
            n_samples: Number of samples to generate
            label: 0 for truthful, 1 for deceptive
            
        Returns:
            DataFrame with cross-modal features
        """
        data = {}
        for feature, (low, high) in self.cross_modal_ranges.items():
            if "sync" in feature or "consistency" in feature:
                # Beta distribution for correlation-like measures
                base = self.rng.beta(4, 2, n_samples) * (high - low) + low
            elif "stress" in feature:
                # Gamma distribution for stress (positive skew)
                base = self.rng.gamma(shape=2, scale=(high-low)/4, size=n_samples) + low
            else:
                # Normal distribution for others
                mean = (low + high) / 2
                std = (high - low) / 6
                base = self.rng.normal(mean, std, n_samples)
            
            base = np.clip(base, low, high)
            data[feature] = base
            
        df = pd.DataFrame(data)
        
        # Apply deception shifts
        if label == 1:
            for i in range(n_samples):
                row = df.iloc[i].to_dict()
                shifted = self._apply_deception_shift(row, label)
                df.iloc[i] = pd.Series(shifted)
                
        return df
    
    def generate_fused_features(self, n_samples: int = 100, label: int = 0) -> pd.DataFrame:
        """Generate complete fused feature set (facial + audio + cross-modal).
        
        Args:
            n_samples: Number of samples to generate
            label: 0 for truthful, 1 for deceptive
            
        Returns:
            DataFrame with all features
        """
        facial = self.generate_facial_features(n_samples, label)
        audio = self.generate_audio_features(n_samples, label)
        cross_modal = self.generate_cross_modal_features(n_samples, label)
        
        # Combine all features
        fused = pd.concat([facial, audio, cross_modal], axis=1)
        return fused
    
    def generate_dataset(self, n_truthful: int = 500, n_deceptive: int = 500,
                        subject_ids: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Generate a complete labeled dataset for training/evaluation.
        
        Args:
            n_truthful: Number of truthful samples
            n_deceptive: Number of deceptive samples
            subject_ids: Optional list of subject IDs. If None, generates IDs.
            
        Returns:
            Tuple of (features DataFrame, labels array, subject_ids list)
        """
        # Generate truthful samples
        truthful_features = self.generate_fused_features(n_truthful, label=0)
        truthful_labels = np.zeros(n_truthful, dtype=int)
        
        # Generate deceptive samples
        deceptive_features = self.generate_fused_features(n_deceptive, label=1)
        deceptive_labels = np.ones(n_deceptive, dtype=int)
        
        # Combine
        features = pd.concat([truthful_features, deceptive_features], axis=0, ignore_index=True)
        labels = np.concatenate([truthful_labels, deceptive_labels])
        
        # Generate subject IDs if not provided
        if subject_ids is None:
            n_subjects = min(20, n_truthful + n_deceptive)  # Up to 20 subjects
            subject_pool = [f"S{i:03d}" for i in range(1, n_subjects + 1)]
            # Assign subjects randomly with some subjects having multiple samples
            subject_ids_list = [self.rng.choice(subject_pool) for _ in range(len(features))]
        else:
            if len(subject_ids) != len(features):
                raise ValueError(f"subject_ids length ({len(subject_ids)}) must match total samples ({len(features)})")
            subject_ids_list = subject_ids
            
        # Shuffle the dataset
        indices = np.arange(len(features))
        self.rng.shuffle(indices)
        features = features.iloc[indices].reset_index(drop=True)
        labels = labels[indices]
        subject_ids_list = [subject_ids_list[i] for i in indices]
        
        return features, labels, subject_ids_list
    
    def generate_temporal_features(self, n_videos: int = 50,
                                  frames_per_video: int = 100,
                                  label: int = 0) -> List[pd.DataFrame]:
        """Generate temporal (frame-by-frame) features for video analysis.
        
        Args:
            n_videos: Number of videos to generate
            frames_per_video: Frames per video
            label: 0 for truthful, 1 for deceptive
            
        Returns:
            List of DataFrames, one per video
        """
        video_features = []
        
        for vid_idx in range(n_videos):
            # Base facial features with temporal correlation
            base_features = {}
            for feature, (low, high) in self.facial_ranges.items():
                # Create correlated time series (AR(1) process)
                ar_param = 0.7  # autocorrelation
                noise_std = (high - low) / 10
                
                # Generate base series
                series = np.zeros(frames_per_video)
                series[0] = self.rng.uniform(low, high)
                for t in range(1, frames_per_video):
                    series[t] = ar_param * series[t-1] + (1 - ar_param) * self.rng.normal((low+high)/2, noise_std)
                
                # Add deception-specific patterns
                if label == 1:
                    # Add spikes for deceptive moments
                    n_spikes = self.rng.integers(1, 4)
                    for _ in range(n_spikes):
                        spike_start = self.rng.integers(20, frames_per_video - 20)
                        spike_duration = self.rng.integers(5, 15)
                        spike_magnitude = self.rng.uniform(0.3, 0.8) * (high - low)
                        
                        if "asymmetry" in feature or "tension" in feature:
                            # Increase asymmetry/tension during spikes
                            for t in range(spike_start, min(spike_start + spike_duration, frames_per_video)):
                                series[t] += spike_magnitude
                
                series = np.clip(series, low, high)
                base_features[feature] = series
            
            df = pd.DataFrame(base_features)
            df['frame'] = np.arange(frames_per_video)
            df['timestamp'] = df['frame'] / self.config.feature.fps  # Assuming 15 FPS
            df['video_id'] = f"video_{vid_idx:03d}"
            df['label'] = label
            
            video_features.append(df)
            
        return video_features
    
    def save_synthetic_dataset(self, output_dir: str,
                              n_truthful: int = 500,
                              n_deceptive: int = 500,
                              dataset_name: str = "synthetic_deception"):
        """Generate and save a synthetic dataset to files.
        
        Args:
            output_dir: Directory to save dataset files
            n_truthful: Number of truthful samples
            n_deceptive: Number of deceptive samples
            dataset_name: Base name for dataset files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate dataset
        features, labels, subject_ids = self.generate_dataset(n_truthful, n_deceptive)
        
        # Save features CSV
        features_path = os.path.join(output_dir, f"{dataset_name}_features.csv")
        features.to_csv(features_path, index=False)
        
        # Save labels CSV
        labels_df = pd.DataFrame({
            'sample_id': range(len(labels)),
            'label': labels,
            'subject_id': subject_ids
        })
        labels_path = os.path.join(output_dir, f"{dataset_name}_labels.csv")
        labels_df.to_csv(labels_path, index=False)
        
        # Save metadata JSON
        metadata = {
            'dataset_name': dataset_name,
            'n_samples': len(features),
            'n_truthful': n_truthful,
            'n_deceptive': n_deceptive,
            'n_features': len(features.columns),
            'feature_groups': {
                'facial': self.FACIAL_FEATURES,
                'audio': self.AUDIO_FEATURES,
                'cross_modal': self.CROSS_MODAL_FEATURES
            },
            'generation_seed': self.seed,
            'generation_date': pd.Timestamp.now().isoformat()
        }
        metadata_path = os.path.join(output_dir, f"{dataset_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Synthetic dataset saved to {output_dir}")
        print(f"   Features: {features_path}")
        print(f"   Labels: {labels_path}")
        print(f"   Metadata: {metadata_path}")
        
        return features_path, labels_path, metadata_path