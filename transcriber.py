"""
Transcriber Component using Deepgram API

This module handles audio transcription using Deepgram's speech-to-text API.
"""

from pathlib import Path
from typing import Optional
import httpx
import config


class Transcriber:
    """
    Handles audio transcription using Deepgram API via REST endpoint.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Transcriber with Deepgram API.
        
        Args:
            api_key: Deepgram API key (defaults to config.DEEPGRAM_API_KEY)
            
        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or config.DEEPGRAM_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Deepgram API key not configured. "
                "Please set DEEPGRAM_API_KEY in .env file."
            )
        
        self.base_url = "https://api.deepgram.com/v1/listen"
    
    def transcribe(self, audio_path: str, language: str = "multi") -> str:
        """
        Transcribe audio file to text using Deepgram with multilingual support.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "multi" for automatic detection)
                     Supports: "multi" (auto-detect), "en", "es", "fr", "de", "pt", 
                     "zh", "ja", "ko", "hi", "ar", and many more
            
        Returns:
            Transcribed text string. Returns fallback message if transcription fails.
            
        Raises:
            FileNotFoundError: If audio file does not exist
        """
        # Validate file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Read audio file
            with open(audio_path, "rb") as file:
                audio_data = file.read()
            
            # Prepare headers
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav"
            }
            
            # Prepare query parameters with multilingual support
            params = {
                "model": "nova-2",
                "detect_language": "true",  # Enable automatic language detection
                "smart_format": "true",
                "punctuate": "true",
            }
            
            # If specific language is requested (not multi), use it
            if language and language != "multi":
                params["language"] = language
                params["detect_language"] = "false"
            
            # Make API request
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.base_url,
                    headers=headers,
                    params=params,
                    content=audio_data
                )
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                
                # Extract transcript
                if "results" in result:
                    channels = result["results"].get("channels", [])
                    if channels and len(channels) > 0:
                        alternatives = channels[0].get("alternatives", [])
                        if alternatives and len(alternatives) > 0:
                            transcript = alternatives[0].get("transcript", "")
                            
                            if transcript and transcript.strip():
                                return transcript.strip()
                
                # No speech detected
                return "[No speech detected]"
            
            elif response.status_code == 401:
                return "[Transcription service authentication failed]"
            elif response.status_code == 429:
                return "[Transcription service rate limit exceeded]"
            else:
                print(f"Deepgram API error: {response.status_code} - {response.text}")
                return "[Transcription failed]"
        
        except httpx.TimeoutException:
            return "[Transcription service timeout]"
        except Exception as e:
            print(f"Deepgram transcription error: {e}")
            return "[Transcription failed]"
    
    def transcribe_with_metadata(self, audio_path: str, language: str = "multi") -> dict:
        """
        Transcribe audio with additional metadata and multilingual support.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "multi" for automatic detection)
            
        Returns:
            Dictionary containing:
            - transcript: Transcribed text
            - confidence: Average confidence score (0-1)
            - duration: Audio duration in seconds
            - detected_language: Detected language code (if auto-detection used)
            - words: List of word-level details (if available)
            
        Raises:
            FileNotFoundError: If audio file does not exist
        """
        # Validate file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Read audio file
            with open(audio_path, "rb") as file:
                audio_data = file.read()
            
            # Prepare headers
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav"
            }
            
            # Prepare query parameters with multilingual support
            params = {
                "model": "nova-2",
                "detect_language": "true",
                "smart_format": "true",
                "punctuate": "true",
            }
            
            # If specific language is requested (not multi), use it
            if language and language != "multi":
                params["language"] = language
                params["detect_language"] = "false"
            
            # Make API request
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.base_url,
                    headers=headers,
                    params=params,
                    content=audio_data
                )
            
            # Initialize result
            result = {
                'transcript': '[No speech detected]',
                'confidence': 0.0,
                'duration': 0.0,
                'detected_language': None,
                'words': []
            }
            
            # Check response status
            if response.status_code == 200:
                data = response.json()
                
                # Get metadata
                if "metadata" in data:
                    result['duration'] = data["metadata"].get("duration", 0.0)
                
                # Get transcript and details
                if "results" in data:
                    channels = data["results"].get("channels", [])
                    if channels and len(channels) > 0:
                        channel = channels[0]
                        
                        # Get detected language
                        if "detected_language" in channel:
                            result['detected_language'] = channel.get("detected_language")
                        
                        alternatives = channel.get("alternatives", [])
                        if alternatives and len(alternatives) > 0:
                            alternative = alternatives[0]
                            
                            # Get transcript
                            transcript = alternative.get("transcript", "")
                            if transcript:
                                result['transcript'] = transcript.strip()
                            
                            # Get confidence
                            result['confidence'] = alternative.get("confidence", 0.0)
                            
                            # Get detected language from alternative if not in channel
                            if not result['detected_language'] and "detected_language" in alternative:
                                result['detected_language'] = alternative.get("detected_language")
                            
                            # Get word-level details
                            words = alternative.get("words", [])
                            if words:
                                result['words'] = [
                                    {
                                        'word': word.get('word', ''),
                                        'start': word.get('start', 0.0),
                                        'end': word.get('end', 0.0),
                                        'confidence': word.get('confidence', 0.0)
                                    }
                                    for word in words
                                ]
            else:
                result['transcript'] = '[Transcription failed]'
            
            return result
        
        except Exception as e:
            print(f"Deepgram transcription error: {e}")
            return {
                'transcript': '[Transcription failed]',
                'confidence': 0.0,
                'duration': 0.0,
                'words': []
            }
    
    def quick_transcribe(self, audio_path: str) -> str:
        """
        Quick transcription (alias for transcribe method).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text string or fallback message
        """
        return self.transcribe(audio_path)
