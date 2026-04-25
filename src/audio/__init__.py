"""Audio analysis module for speech pattern detection."""

from .extractor import AudioFeatureExtractor
from .fusion import MultimodalFusion

__all__ = ['AudioFeatureExtractor', 'MultimodalFusion']
