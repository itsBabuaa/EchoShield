"""
Transcription Component for Deepfake Audio Detection

This module handles audio-to-text transcription using speech recognition.

Requirements: 5.1, 5.3
"""

import speech_recognition as sr
from pathlib import Path
from typing import Optional


class Transcriber:
    """
    Handles audio transcription using speech recognition.
    """
    
    def __init__(self):
        """
        Initialize Transcriber with speech recognition.
        """
        self.recognizer = sr.Recognizer()
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text string. Returns empty string if transcription fails.
            
        Raises:
            FileNotFoundError: If audio file does not exist
        """
        # Validate file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio data
                audio_data = self.recognizer.record(source)
            
            # Attempt transcription using Google Speech Recognition
            try:
                transcript = self.recognizer.recognize_google(audio_data)
                return transcript
            
            except sr.UnknownValueError:
                # Speech was unintelligible
                return "No speech detected in audio."
            
            except sr.RequestError as e:
                # API request failed
                return f"Transcription service error: {str(e)}"
        
        except Exception as e:
            # Handle other errors gracefully
            return f"Transcription unavailable: {str(e)}"
    
    def transcribe_with_alternative(self, audio_path: str, use_sphinx: bool = False) -> str:
        """
        Transcribe audio with fallback to offline recognition.
        
        Args:
            audio_path: Path to audio file
            use_sphinx: If True, use offline Sphinx recognition as fallback
            
        Returns:
            Transcribed text string
            
        Raises:
            FileNotFoundError: If audio file does not exist
        """
        # Try online transcription first
        transcript = self.transcribe(audio_path)
        
        # If online failed and sphinx is available, try offline
        if use_sphinx and ("error" in transcript.lower() or "unavailable" in transcript.lower()):
            try:
                with sr.AudioFile(audio_path) as source:
                    audio_data = self.recognizer.record(source)
                
                # Try offline Sphinx recognition
                transcript = self.recognizer.recognize_sphinx(audio_data)
                return transcript
            
            except Exception:
                # Return original transcript if Sphinx also fails
                pass
        
        return transcript
