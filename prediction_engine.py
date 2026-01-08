"""
Prediction Engine Component for Deepfake Audio Detection

This module handles loading the trained BiLSTM model and running predictions
on preprocessed audio features.

Requirements: 2.5, 3.5
"""

import numpy as np
from keras.models import load_model
from typing import Dict
import config


class PredictionEngine:
    """
    Handles model loading and prediction for deepfake audio detection.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize PredictionEngine and load the trained model.
        
        Args:
            model_path: Path to the trained model file (default from config)
            
        Raises:
            FileNotFoundError: If model file does not exist
            RuntimeError: If model loading fails
        """
        self.model_path = model_path or config.MODEL_PATH
        self.model = self._load_model(self.model_path)
        self.class_labels = {0: 'real', 1: 'fake'}
    
    def _load_model(self, model_path: str):
        """
        Load the trained Keras model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded Keras model
            
        Raises:
            FileNotFoundError: If model file does not exist
            RuntimeError: If model loading fails
        """
        import os
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train the model first using train_model.py"
            )
        
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Run prediction on preprocessed audio features.
        
        Args:
            features: Preprocessed MFCC features
                     Shape: (1, max_length, n_mfcc) or (max_length, n_mfcc)
                     
        Returns:
            Dictionary containing:
                - 'label': Predicted class label ('real' or 'fake')
                - 'confidence': Confidence score (0.0 to 1.0)
                - 'probabilities': Dictionary with probabilities for each class
                  {'real': float, 'fake': float}
                  
        Raises:
            ValueError: If input features have incorrect shape
            RuntimeError: If prediction fails
        """
        try:
            # Ensure features have batch dimension
            if features.ndim == 2:
                features = np.expand_dims(features, axis=0)
            
            # Validate input shape
            expected_shape = (1, config.MAX_AUDIO_LENGTH, config.N_MFCC)
            if features.shape != expected_shape:
                raise ValueError(
                    f"Invalid input shape. Expected {expected_shape}, "
                    f"got {features.shape}"
                )
            
            # Run prediction
            predictions = self.model.predict(features, verbose=0)
            
            # Extract probabilities
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_class = np.argmax(probabilities)
            predicted_label = self.class_labels[predicted_class]
            confidence = float(probabilities[predicted_class])
            
            # Build result dictionary
            result = {
                'label': predicted_label,
                'confidence': confidence,
                'probabilities': {
                    'real': float(probabilities[0]),
                    'fake': float(probabilities[1])
                }
            }
            
            return result
            
        except ValueError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_from_file(self, file_path: str) -> Dict:
        """
        Convenience method to predict directly from an audio file.
        
        This method combines audio processing and prediction in one call.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing prediction results (same as predict())
            
        Raises:
            FileNotFoundError: If audio file does not exist
            ValueError: If file format is not supported
            RuntimeError: If processing or prediction fails
        """
        from audio_processor import AudioProcessor
        
        # Initialize audio processor
        processor = AudioProcessor()
        
        # Preprocess audio
        features = processor.preprocess_for_prediction(file_path)
        
        # Run prediction
        result = self.predict(features)
        
        return result
