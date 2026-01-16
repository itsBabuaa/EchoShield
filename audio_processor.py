"""
Audio Processor Component for Deepfake Audio Detection

This module handles audio loading, format conversion, and MFCC feature extraction
for the prediction pipeline.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict
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
    
    def get_audio_metrics(self, file_path: str) -> Dict:
        """
        Extract comprehensive audio metrics for analysis.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing audio metrics:
            - duration: Audio duration in seconds
            - sample_rate: Sample rate in Hz
            - file_size: File size in bytes
            - peak_amplitude: Maximum absolute amplitude (0-1)
            - rms_level: Root mean square level in dB
            - dynamic_range: Dynamic range in dB
            - zero_crossings: Number of zero crossings per second
            - spectral_centroid: Average spectral centroid in Hz
            - noise_floor: Estimated noise floor in dB
        """
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)
            
            # Get file size
            file_size = int(Path(file_path).stat().st_size)
            
            # Calculate duration
            duration = float(len(audio) / sr)
            
            # Peak amplitude (normalized 0-1)
            peak_amplitude = float(np.max(np.abs(audio)))
            
            # RMS level in dB
            rms = float(np.sqrt(np.mean(audio ** 2)))
            rms_db = float(20 * np.log10(rms + 1e-10))  # Add small value to avoid log(0)
            
            # Dynamic range (difference between peak and RMS in dB)
            peak_db = float(20 * np.log10(peak_amplitude + 1e-10))
            dynamic_range = float(peak_db - rms_db)
            
            # Zero crossing rate (per second)
            zero_crossings = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_per_second = float(np.mean(zero_crossings) * sr)
            
            # Spectral centroid (average frequency in Hz)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            avg_spectral_centroid = float(np.mean(spectral_centroid))
            
            # Noise floor estimation (using lowest 10% of RMS values)
            frame_length = 2048
            hop_length = 512
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
            sorted_rms = np.sort(frame_rms)
            noise_floor_rms = float(np.mean(sorted_rms[:max(1, len(sorted_rms) // 10)]))
            noise_floor_db = float(20 * np.log10(noise_floor_rms + 1e-10))
            
            return {
                'duration': round(duration, 2),
                'sample_rate': int(sr),
                'file_size': file_size,
                'peak_amplitude': round(peak_amplitude, 4),
                'rms_level': round(rms_db, 1),
                'dynamic_range': round(dynamic_range, 1),
                'zero_crossings': int(zcr_per_second),
                'spectral_centroid': int(avg_spectral_centroid),
                'noise_floor': round(noise_floor_db, 1)
            }
            
        except Exception as e:
            # Return default values if metrics extraction fails
            return {
                'duration': 0.0,
                'sample_rate': int(self.sample_rate),
                'file_size': 0,
                'peak_amplitude': 0.0,
                'rms_level': -60.0,
                'dynamic_range': 0.0,
                'zero_crossings': 0,
                'spectral_centroid': 0,
                'noise_floor': -60.0
            }
