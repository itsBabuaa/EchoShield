"""
Chatbot Component for Deepfake Audio Detection

This module provides an AI-powered chatbot using Groq API for interactive
analysis and explanations of audio detection results.

Requirements: 6.2, 6.3, 6.4, 6.6
"""

from groq import Groq
from typing import List, Dict, Optional
import config


class Chatbot:
    """
    AI-powered chatbot for interactive audio analysis using Groq API.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Chatbot with Groq API.
        
        Args:
            api_key: Groq API key (default from config)
            
        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or config.GROQ_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY in .env file."
            )
        
        self.client = Groq(api_key=self.api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.context: Optional[Dict] = None
        self.model =  "llama-3.1-8b-instant"  #"openai/gpt-oss-120b"
    
    def set_context(self, transcript: str, prediction: Dict):
        """
        Set audio analysis context for the conversation.
        
        Args:
            transcript: Audio transcription text
            prediction: Prediction results dictionary containing:
                       - 'label': 'real' or 'fake'
                       - 'confidence': confidence score
                       - 'probabilities': probability dict
        """
        self.context = {
            'transcript': transcript,
            'prediction': prediction
        }
        
        # Create system message with context
        system_message = self._create_system_message()
        
        # Initialize conversation with system message
        self.conversation_history = [
            {
                "role": "system",
                "content": system_message
            }
        ]
    
    def _create_system_message(self) -> str:
        """
        Create system message with audio analysis context.
        
        Returns:
            System message string
        """
        if not self.context:
            return (
                "You are an AI assistant helping users understand deepfake audio detection results. "
                "Provide clear, helpful explanations about audio authenticity and detection methods."
                "Nerver reply for other queires."
                "Do not use internet for the info just helpp for the uploaded audio."
            )
        
        label = self.context['prediction']['label']
        confidence = self.context['prediction']['confidence']
        transcript = self.context['transcript']
        real_prob = self.context['prediction']['probabilities']['real']
        fake_prob = self.context['prediction']['probabilities']['fake']
        
        system_message = f"""You are an AI assistant helping users understand deepfake audio detection results.

Audio Analysis Results:
- Classification: {label.upper()}
- Confidence: {confidence:.2%}
- Real probability: {real_prob:.2%}
- Fake probability: {fake_prob:.2%}
- Transcript: "{transcript}"

Your role is to:
1. Explain the detection results in clear, understandable terms
2. Answer questions.
3. Provide insights about the audio analysis
4. Help users understand the confidence scores.
5. Discuss potential implications of the results
6. Always keep your answers short and Strictly stick to analysis scope do not over use the internet.

Be helpful, accurate, and educational. If asked about technical details, explain them in accessible language. Do not answer any other query."""
        
        return system_message
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response from the chatbot.
        
        Args:
            user_message: User's message text
            
        Returns:
            Chatbot's response text
            
        Raises:
            RuntimeError: If API request fails
        """
        if not user_message or not user_message.strip():
            return "Please provide a message."
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.6,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            
            # Extract assistant's response
            assistant_message = response.choices[0].message.content
            
            # Add assistant's response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        
        except Exception as e:
            error_message = f"Chatbot error: {str(e)}"
            
            # Check for specific error types
            if "rate_limit" in str(e).lower():
                return "Too many requests. Please wait a moment and try again."
            elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                return "Chatbot service not configured properly. Please check API key."
            else:
                return "Chatbot temporarily unavailable. Please try again."
    
    def clear_history(self):
        """
        Clear conversation history and reset context.
        
        This starts a fresh conversation while maintaining the audio analysis context.
        """
        if self.context:
            # Recreate system message with context
            system_message = self._create_system_message()
            self.conversation_history = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]
        else:
            # Clear everything
            self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        # Return history without system message
        return [msg for msg in self.conversation_history if msg['role'] != 'system']
