"""
BiLSTM Model Training Script for Deepfake Audio Detection

This script trains a BiLSTM model on the SceneFake dataset for detecting
deepfake audio. It includes data loading, MFCC feature extraction, model
building, training, and evaluation.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Optional
import config

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# =============================================================================
# Data Loading Functions (Task 2.1 - Completed)
# =============================================================================

def load_audio_files(directory: str) -> List[Tuple[str, int]]:
    """
    Load audio file paths and labels from directory.
    
    Expects directory structure:
    - directory/real/*.wav (or mp3, flac)
    - directory/fake/*.wav (or mp3, flac)
    
    Or files with naming convention containing 'real' or 'fake' in filename.
    
    Args:
        directory: Path to the data directory
        
    Returns:
        List of tuples (file_path, label) where label is 0 for real, 1 for fake
    """
    data = []
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist")
        return data
    
    # Check for subdirectory structure (real/fake folders)
    real_dir = directory / 'real'
    fake_dir = directory / 'fake'
    
    real_count = 0
    fake_count = 0
    
    if real_dir.exists() or fake_dir.exists():
        # Load from subdirectories
        if real_dir.exists():
            for audio_file in real_dir.glob('*'):
                if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac']:
                    data.append((str(audio_file), 0))  # 0 = real
                    real_count += 1
        
        if fake_dir.exists():
            for audio_file in fake_dir.glob('*'):
                if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac']:
                    data.append((str(audio_file), 1))  # 1 = fake
                    fake_count += 1
        
        print(f"  Found {real_count} real and {fake_count} fake samples in {directory}")
    else:
        # Load from flat directory, parse labels from filenames
        for audio_file in directory.glob('*'):
            if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac']:
                filename = audio_file.stem.lower()
                if 'real' in filename or 'bonafide' in filename:
                    data.append((str(audio_file), 0))
                    real_count += 1
                elif 'fake' in filename or 'spoof' in filename:
                    data.append((str(audio_file), 1))
                    fake_count += 1
        
        print(f"  Found {real_count} real and {fake_count} fake samples in {directory}")
    
    return data


def load_audio_signal(file_path: str, sr: int = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return signal with sample rate.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (None to use original)
        
    Returns:
        Tuple of (audio_signal, sample_rate)
    """
    if sr is None:
        sr = config.SAMPLE_RATE
    
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate


# =============================================================================
# MFCC Feature Extraction (Task 2.2)
# =============================================================================

def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = None,
    max_length: int = None
) -> np.ndarray:
    """
    Extract MFCC features from audio signal.
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients to extract (default from config)
        max_length: Maximum number of time frames (default from config)
        
    Returns:
        MFCC features as 2D numpy array of shape (max_length, n_mfcc)
    """
    if n_mfcc is None:
        n_mfcc = config.N_MFCC
    if max_length is None:
        max_length = config.MAX_AUDIO_LENGTH
    
    # Handle empty or invalid audio
    if audio is None or len(audio) == 0:
        return np.zeros((max_length, n_mfcc))
    
    # Extract MFCC features
    # Shape: (n_mfcc, time_frames)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Transpose to (time_frames, n_mfcc)
    mfcc = mfcc.T
    
    # Pad or truncate to fixed length
    mfcc = pad_or_truncate(mfcc, max_length)
    
    return mfcc


def pad_or_truncate(features: np.ndarray, max_length: int) -> np.ndarray:
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


def extract_features_from_file(
    file_path: str,
    sr: int = None,
    n_mfcc: int = None,
    max_length: int = None
) -> np.ndarray:
    """
    Extract MFCC features from an audio file.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate (default from config)
        n_mfcc: Number of MFCC coefficients (default from config)
        max_length: Maximum sequence length (default from config)
        
    Returns:
        MFCC features as 2D numpy array
    """
    audio, sample_rate = load_audio_signal(file_path, sr)
    return extract_mfcc(audio, sample_rate, n_mfcc, max_length)


def prepare_dataset(
    data: List[Tuple[str, int]],
    sr: int = None,
    n_mfcc: int = None,
    max_length: int = None,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset by extracting features from all audio files.
    
    Args:
        data: List of (file_path, label) tuples
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        max_length: Maximum sequence length
        verbose: Print progress
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    features = []
    labels = []
    
    total = len(data)
    for i, (file_path, label) in enumerate(data):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{total} files...")
        
        try:
            mfcc = extract_features_from_file(file_path, sr, n_mfcc, max_length)
            features.append(mfcc)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Normalize features to prevent gradient explosion
    # Use mean and std normalization
    mean = np.mean(features)
    std = np.std(features)
    if std > 0:
        features = (features - mean) / std
    
    return features, labels


# =============================================================================
# BiLSTM Model Building and Training (Task 2.4)
# =============================================================================

def build_bilstm_model(
    input_shape: Tuple[int, int],
    num_classes: int = 2,
    lstm_units: int = 64,
    dropout_rate: float = 0.4
) -> Sequential:
    """
    Build BiLSTM model architecture for audio classification.
    
    Args:
        input_shape: Shape of input features (time_steps, n_features)
        num_classes: Number of output classes (2 for real/fake)
        lstm_units: Number of LSTM units per layer
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # First Bidirectional LSTM layer (use default tanh activation for stability)
        Bidirectional(LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.2)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Second Bidirectional LSTM layer
        Bidirectional(LSTM(lstm_units // 2, recurrent_dropout=0.2)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Dense layers with ReLU activation
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    model_save_path: str = None
) -> keras.callbacks.History:
    """
    Train the BiLSTM model with given data.
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        model_save_path: Path to save the best model
        
    Returns:
        Training history object
    """
    if model_save_path is None:
        model_save_path = config.MODEL_PATH
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (loss, accuracy)
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy


def load_trained_model(model_path: str = None) -> Sequential:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    return keras.models.load_model(model_path)


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Deepfake Audio Detection - BiLSTM Model Training")
    print("=" * 60)
    
    # Configuration
    print("\nConfiguration:")
    print(f"  Sample Rate: {config.SAMPLE_RATE}")
    print(f"  MFCC Coefficients: {config.N_MFCC}")
    print(f"  Max Audio Length: {config.MAX_AUDIO_LENGTH}")
    print(f"  Model Save Path: {config.MODEL_PATH}")
    
    # Load data
    print("\n" + "=" * 60)
    print("Loading Data...")
    print("=" * 60)
    
    train_data = load_audio_files(config.TRAIN_PATH)
    dev_data = load_audio_files(config.DEV_PATH)
    test_data = load_audio_files(config.TEST_PATH)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(dev_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    if len(train_data) == 0:
        print("\nError: No training data found!")
        print("Please ensure the SceneFake dataset is in ./scenefake/train/")
        return
    
    # Extract features
    print("\n" + "=" * 60)
    print("Extracting MFCC Features...")
    print("=" * 60)
    
    print("\nProcessing training data...")
    X_train, y_train = prepare_dataset(train_data)
    
    print("\nProcessing validation data...")
    X_val, y_val = prepare_dataset(dev_data)
    
    print("\nProcessing test data...")
    X_test, y_test = prepare_dataset(test_data)
    
    print(f"\nFeature shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Build model
    print("\n" + "=" * 60)
    print("Building BiLSTM Model...")
    print("=" * 60)
    
    input_shape = (config.MAX_AUDIO_LENGTH, config.N_MFCC)
    model = build_bilstm_model(input_shape)
    model.summary()
    
    # Train model
    print("\n" + "=" * 60)
    print("Training Model...")
    print("=" * 60)
    
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    
    test_loss, test_accuracy = evaluate_model(model, X_test, y_test)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {config.MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
