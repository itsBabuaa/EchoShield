"""
Chatbot Component for Deepfake Audio Detection

This module provides an AI-powered chatbot using Groq API for interactive
analysis and explanations of audio detection results.
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
    
    def set_context(self, transcript: str, prediction: Dict, audio_metrics: Dict = None):
        """
        Set audio analysis context for the conversation.
        
        Args:
            transcript: Audio transcription text
            prediction: Prediction results dictionary containing:
                       - 'label': 'real' or 'fake'
                       - 'confidence': confidence score
                       - 'probabilities': probability dict
            audio_metrics: Optional audio metrics dictionary containing:
                          - 'duration': Audio duration in seconds
                          - 'sample_rate': Sample rate in Hz
                          - 'file_size': File size in bytes
                          - 'peak_amplitude': Maximum absolute amplitude
                          - 'rms_level': RMS level in dB
                          - 'dynamic_range': Dynamic range in dB
                          - 'zero_crossings': Zero crossings per second
                          - 'spectral_centroid': Average spectral centroid in Hz
                          - 'noise_floor': Noise floor in dB
        """
        self.context = {
            'transcript': transcript,
            'prediction': prediction,
            'audio_metrics': audio_metrics or {}
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
        # Concise EchoShield system information
        echoshield_info = """ECHOSHIELD: Deepfake audio detection system
MODEL: BiLSTM Neural Network | Accuracy: 93.8% | Precision: 93.7% | Recall: 99.0% | F1: 96.3% | AUC-ROC: 88.6%
ARCHITECTURE: 2 BiLSTM layers (64→32 units), Dropout 0.4, Dense 64, Softmax output
FEATURES: 40 MFCC coefficients, 300 time frames
DATASET: SceneFake (13,185 train, 12,842 val samples)

TEAM:
- Dr. Sonali Mathur (Supervisor) - sonali.mathur@imsec.ac.in
- Atharv Singh (Lead Dev) - Model training, Flask API - @itsBabuaa
- Ayush Pratap Singh (Frontend) - Web interface, design - @itsBabuaa
- Ansh Srivastava (Backend) - API, audio processing - @itsBabuaa
- Anushka Singh (Data Analyst) - Dataset analysis - @itsBabuaa"""
        
        if not self.context:
            return f"""You are EchoBot, a friendly and helpful AI assistant for EchoShield deepfake audio detection system.

{echoshield_info}

YOUR PERSONALITY:
- Friendly, conversational, and approachable
- Knowledgeable about EchoShield, deepfake detection, and audio analysis
- Patient and helpful when explaining technical concepts
- Can engage in casual conversation while staying focused on your expertise

YOUR EXPERTISE:
- EchoShield system features and capabilities
- Deepfake audio detection technology and concepts
- Audio analysis and signal processing basics
- Machine learning models (especially BiLSTM)
- Team information and project details
- General audio/speech technology questions

GUIDELINES:
1. Answer questions about EchoShield, deepfakes, audio detection, and related topics thoroughly
2. Be conversational - greet users, use friendly language, acknowledge their questions
3. For general questions (weather, news, etc.), politely redirect: "That's outside my expertise, but I'd love to help with questions about EchoShield or audio detection!"
4. Keep technical answers clear and accessible (under 150 words)
5. If unsure, be honest: "I'm not certain about that. For more help, contact: Atharvsingh.edu@gmail.com"
6. You can discuss audio technology in general, not just EchoShield

TONE: Friendly expert - like a knowledgeable colleague who's happy to help"""
        
        label = self.context['prediction']['label']
        confidence = self.context['prediction']['confidence']
        transcript = self.context['transcript']
        real_prob = self.context['prediction']['probabilities']['real']
        fake_prob = self.context['prediction']['probabilities']['fake']
        
        # Build audio metrics section
        audio_metrics = self.context.get('audio_metrics', {})
        metrics_section = ""
        
        if audio_metrics:
            duration = audio_metrics.get('duration', 0)
            sample_rate = audio_metrics.get('sample_rate', 0)
            file_size = audio_metrics.get('file_size', 0)
            peak_amplitude = audio_metrics.get('peak_amplitude', 0)
            rms_level = audio_metrics.get('rms_level', 0)
            dynamic_range = audio_metrics.get('dynamic_range', 0)
            zero_crossings = audio_metrics.get('zero_crossings', 0)
            spectral_centroid = audio_metrics.get('spectral_centroid', 0)
            noise_floor = audio_metrics.get('noise_floor', 0)
            
            # Format file size
            if file_size >= 1024 * 1024:
                file_size_str = f"{file_size / (1024 * 1024):.2f} MB"
            elif file_size >= 1024:
                file_size_str = f"{file_size / 1024:.2f} KB"
            else:
                file_size_str = f"{file_size} bytes"
            
            # Format sample rate
            if sample_rate >= 1000:
                sample_rate_str = f"{sample_rate / 1000:.1f} kHz"
            else:
                sample_rate_str = f"{sample_rate} Hz"
            
            metrics_section = f"""
AUDIO FILE METRICS:
- Duration: {duration:.2f} seconds
- Sample Rate: {sample_rate_str}
- File Size: {file_size_str}

ADVANCED AUDIO ANALYSIS:
- Peak Amplitude: {peak_amplitude:.4f} (normalized 0-1)
- RMS Level: {rms_level:.1f} dB
- Dynamic Range: {dynamic_range:.1f} dB
- Zero Crossings: {zero_crossings:,} per second
- Spectral Centroid: {spectral_centroid:,} Hz (average frequency)
- Noise Floor: {noise_floor:.1f} dB

METRIC EXPLANATIONS:
- Peak Amplitude: Maximum signal strength (closer to 1 = louder)
- RMS Level: Average loudness in decibels (higher = louder)
- Dynamic Range: Difference between loudest and average (higher = more variation)
- Zero Crossings: How often signal crosses zero (higher = more high-frequency content)
- Spectral Centroid: "Center of mass" of frequencies (higher = brighter sound)
- Noise Floor: Background noise level (lower = cleaner audio)
"""
        
        return f"""You are EchoBot, a friendly and helpful AI assistant for EchoShield deepfake audio detection system.

{echoshield_info}

CURRENT AUDIO ANALYSIS:
- Detection Result: {label.upper()}
- Confidence: {confidence:.1%}
- Real Probability: {real_prob:.1%}
- Fake Probability: {fake_prob:.1%}
{metrics_section}
TRANSCRIPT: "{transcript}"

YOUR PERSONALITY:
- Friendly, conversational, and approachable
- Expert in audio analysis and deepfake detection
- Patient when explaining technical concepts
- Enthusiastic about helping users understand their results

YOUR EXPERTISE:
- This specific audio analysis and what the results mean
- All audio metrics and their significance
- How our model makes predictions
- EchoShield features and capabilities
- Deepfake detection technology
- Audio signal processing concepts
- Team information

GUIDELINES:
1. Help users understand their audio analysis results thoroughly
2. Explain audio metrics clearly when asked (you have full access to all metrics above)
3. Be conversational - acknowledge questions, use friendly language
4. For unrelated topics, politely redirect: "That's outside my expertise, but I'd love to help you understand your audio analysis results!"
5. Keep answers clear and accessible (under 200 words)
6. Connect metrics to real-world meaning (e.g., "High dynamic range means your audio has good variation between loud and quiet parts")
7. If unsure, be honest: "I'm not certain about that. Contact: Atharvsingh.edu@gmail.com"

TONE: Friendly expert - like a helpful colleague explaining your results"""
    
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
        
        # Check if query is related to EchoShield/deepfake detection
        user_message_lower = user_message.lower()
        
        # Keywords that indicate the query is about audio content/transcript (VALID)
        audio_content_keywords = [
            'audio', 'recording', 'transcript', 'speech', 'voice', 'sound',
            'said', 'saying', 'talk', 'talking', 'speak', 'speaking',
            'hear', 'heard', 'listen', 'content', 'file', 'clip'
        ]
        
        # Keywords that indicate EchoShield-related queries (VALID)
        echoshield_keywords = [
            'echoshield', 'echo shield', 'echobot', 'deepfake', 
            'detection', 'fake', 'real', 'authentic', 'model', 'accuracy',
            'confidence', 'analysis', 'result', 'prediction', 'team',
            'creator', 'developer', 'bilstm', 'mfcc', 'neural network',
            'upload', 'detect', 'analyze', 'classification'
        ]
        
        # Keywords that indicate off-topic queries (INVALID - but only if not about audio content)
        off_topic_keywords = [
            'code', 'program', 'function', 'script', 'algorithm',
            'weather', 'news', 'recipe', 'movie',
            'math', 'calculate', 'sum', 'multiply', 'divide',
            'translate', 'write', 'create', 'build', 'develop',
            'joke', 'story', 'poem', 'song',
            'travel', 'hotel', 'stay', 'trip', 'vacation', 'tour',
            'restaurant', 'food', 'cook', 'eat',
            'stock', 'market', 'price', 'rate', 'rbi', 'repo', 'interest',
            'bank', 'loan', 'credit', 'investment',
            'health', 'medicine', 'doctor', 'disease',
            'politics', 'election', 'government', 'law',
            'shopping', 'buy', 'purchase', 'product',
            'book', 'read', 'novel', 'author',
            'music', 'concert', 'band', 'singer',
            'car', 'bike', 'vehicle', 'drive',
            'phone', 'mobile', 'laptop', 'computer',
            'job', 'career', 'salary', 'interview'
        ]
        
        # Check if message is about audio content or EchoShield
        has_audio_content_keyword = any(keyword in user_message_lower for keyword in audio_content_keywords)
        has_echoshield_keyword = any(keyword in user_message_lower for keyword in echoshield_keywords)
        has_off_topic_keyword = any(keyword in user_message_lower for keyword in off_topic_keywords)
        
        # Additional check: if message is a general question NOT about audio
        general_question_patterns = [
            'where should i', 'where can i', 'what is the',
            'tell me about', 'tell me the', 'what are the best',
            'recommend', 'suggest', 'advice', 'tips for',
            'current', 'latest', 'today', 'now'
        ]
        
        has_general_question = any(pattern in user_message_lower for pattern in general_question_patterns)
        
        # VALID if: has audio content keywords OR has EchoShield keywords
        is_valid_query = has_audio_content_keyword or has_echoshield_keyword
        
        # INVALID if: has off-topic keywords OR general questions WITHOUT valid context
        is_invalid_query = (has_off_topic_keyword or has_general_question) and not is_valid_query
        
        # Block invalid queries
        if is_invalid_query:
            return (
                "I'm EchoBot, specialized in EchoShield and deepfake audio detection. "
                "I can only help with questions about:\n"
                "• EchoShield system and features\n"
                "• Audio analysis results and content\n"
                "• Model performance and architecture\n"
                "• Team members and creators\n"
                "• Deepfake detection concepts\n\n"
                "Please ask me something related to EchoShield or the analyzed audio."
            )
        
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
                temperature=0.5,
                max_tokens=256,
                top_p=0.9,
                stream=False
            )
            
            # Extract assistant's response
            assistant_message = response.choices[0].message.content
            
            # Check if the response seems off-topic (contains code blocks or unrelated content)
            if '```' in assistant_message or 'def ' in assistant_message or 'function ' in assistant_message:
                # Remove the off-topic response from history
                self.conversation_history.pop()
                return (
                    "I'm EchoBot, specialized in EchoShield and deepfake audio detection. "
                    "I cannot help with coding, programming, or unrelated tasks. "
                    "Please ask me about EchoShield, audio analysis, or deepfake detection."
                )
            
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
                return "Too many requests. Please wait a moment and try again. For assistance, contact: Atharvsingh.edu@gmail.com"
            elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                return "Chatbot service not configured properly. Please contact our team at Atharvsingh.edu@gmail.com"
            else:
                return "Chatbot temporarily unavailable. Please try again or contact us at Atharvsingh.edu@gmail.com"
    
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
