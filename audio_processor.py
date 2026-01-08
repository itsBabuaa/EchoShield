"""
Audio Processor Component for Deepfake Audio Detection

This module handles audio loading, format conversion, and MFCC feature extraction
for the prediction pipeline.

Requirements: 2.2, 2.4, 8.1, 8.3
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import config


class AudioProcessor:
    """
    Handles audio file loading, validation, and feature extraction.
    
    Supports WAV, MP3, and FLAC formats.
    """
    
    def __init__(
        self,
        sample_rate: int = None,
        n_mfcc: int = None,
        max_length: int = None
    ):
        """
        Initialize AudioProcessor with configuration.
        
        Args:
            sample_rate: Target sample rate for audio (default from config)
            n_mfcc: Number of MFCC coefficients (default from config)
            max_length: Maximum sequence length in frames (default from config)
        """
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.n_mfcc = n_mfcc or config.N_MFCC
        self.max_length = max_length or config.MAX_AUDIO_LENGTH
        self.allowed_extensions = config.ALLOWED_EXTENSIONS
    
    def validate_file_format(self, file_path: str) -> bool:
        """
        Validate if the file format is supported.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if format is supported, False otherwise
        """
        path = Path(file_path)
        extension = path.suffix.lower().lstrip('.')
        return extension in self.allowed_extensions
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return signal with sample rate.
        
        Supports WAV, MP3, and FLAC formats.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_signal, sample_rate)
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
            RuntimeError: If audio loading fails
        """
        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Validate file format
        if not self.validate_file_format(file_path):
            raise ValueError(
                f"Unsupported audio format. Please upload WAV, MP3, or FLAC files."
            )
        
        try:
            # Load audio using librosa
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Validate audio is not empty
            if audio is None or len(audio) == 0:
                raise RuntimeError("Audio file appears to be empty or too short.")
            
            return audio, sr
            
        except Exception as e:
            raise RuntimeError(f"Unable to process audio file: {str(e)}")
    
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC features from audio signal.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate of the audio
            
        Returns:
            MFCC features as 2D numpy array of shape (max_length, n_mfcc)
            
        Raises:
            RuntimeError: If feature extraction fails
        """
        try:
            # Handle empty or invalid audio
            if audio is None or len(audio) == 0:
                return np.zeros((self.max_length, self.n_mfcc))
            
            # Extract MFCC features
            # Shape: (n_mfcc, time_frames)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            
            # Transpose to (time_frames, n_mfcc)
            mfcc = mfcc.T
            
            # Pad or truncate to fixed length
            mfcc = self._pad_or_truncate(mfcc, self.max_length)
            
            # Normalize features
            mfcc = self._normalize_features(mfcc)
            
            return mfcc
            
        except Exception as e:
            raise RuntimeError(f"Unable to extract audio features: {str(e)}")
    
    def _pad_or_truncate(self, features: np.ndarray, max_length: int) -> np.ndarray:
        """
        Pad or truncate feature sequence to fixed length.
        
        Args:
            features: 2D array of shape (time_frames, n_features)
            max_length: Target number of time frames
            
        Returns:
            Padded/truncated array of shape (max_length, n_features)
        """
        current_length = features.shape[0]
        n_features = features.shape[1]
        
        if current_length > max_length:
            # Truncate
            return features[:max_length, :]
        elif current_length < max_length:
            # Pad with zeros
            padding = np.zeros((max_length - current_length, n_features))
            return np.vstack([features, padding])
        else:
            return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using mean and standard deviation.
        
        Args:
            features: Feature array
            
        Returns:
            Normalized feature array
        """
        mean = np.mean(features)
        std = np.std(features)
        
        if std > 0:
            return (features - mean) / std
        else:
            return features
    
    def preprocess_for_prediction(self, file_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline for prediction.
        
        Loads audio file, extracts MFCC features, and prepares for model input.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Preprocessed features ready for model prediction
            Shape: (1, max_length, n_mfcc) - batch dimension added
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
            RuntimeError: If preprocessing fails
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Add batch dimension for model input
        features = np.expand_dims(features, axis=0)
        
        return features
