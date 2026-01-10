import speech_recognition as sr
from pathlib import Path
from typing import Optional
import time


class Transcriber:
    """
    Handles audio transcription using speech recognition with retry logic.
    """
    
    def __init__(self):
        """
        Initialize Transcriber with speech recognition.
        """
        self.recognizer = sr.Recognizer()
        # Adjust recognizer settings for better accuracy
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
    
    def transcribe(self, audio_path: str, max_retries: int = 3) -> str:
        """
        Transcribe audio file to text with retry logic.
        
        Args:
            audio_path: Path to audio file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Transcribed text string. Returns fallback message if transcription fails.
            
        Raises:
            FileNotFoundError: If audio file does not exist
        """
        # Validate file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Try transcription with retries
        for attempt in range(max_retries):
            try:
                # Load audio file
                with sr.AudioFile(audio_path) as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    # Record audio data
                    audio_data = self.recognizer.record(source)
                
                # Attempt transcription using Google Speech Recognition
                try:
                    transcript = self.recognizer.recognize_google(
                        audio_data,
                        language='en-US',
                        show_all=False
                    )
                    
                    if transcript and transcript.strip():
                        return transcript
                    else:
                        # Empty transcript, try again
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                            continue
                        return "[No speech detected]"
                
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    return "[Audio contains no clear speech]"
                
                except sr.RequestError as e:
                    # API request failed
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return "[Transcription service temporarily unavailable]"
            
            except Exception as e:
                # Handle other errors
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return "[Transcription failed - audio may be corrupted]"
        
        # If all retries failed
        return "[Transcription unavailable]"
    
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
        if use_sphinx and transcript.startswith("["):
            try:
                with sr.AudioFile(audio_path) as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self.recognizer.record(source)
                
                # Try offline Sphinx recognition
                offline_transcript = self.recognizer.recognize_sphinx(audio_data)
                if offline_transcript and offline_transcript.strip():
                    return offline_transcript
            
            except Exception:
                # Return original transcript if Sphinx also fails
                pass
        
        return transcript
    
    def quick_transcribe(self, audio_path: str) -> str:
        """
        Quick transcription without retries (for faster processing).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text string or fallback message
        """
        try:
            if not Path(audio_path).exists():
                return "[Audio file not found]"
            
            with sr.AudioFile(audio_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = self.recognizer.record(source)
            
            transcript = self.recognizer.recognize_google(audio_data, language='en-US')
            return transcript if transcript else "[No speech detected]"
        
        except sr.UnknownValueError:
            return "[No clear speech detected]"
        except sr.RequestError:
            return "[Transcription service unavailable]"
        except Exception:
            return "[Transcription failed]"

