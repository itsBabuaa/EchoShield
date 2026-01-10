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
            return f"""You are EchoBot for EchoShield deepfake audio detection.

{echoshield_info}

STRICT RULES - NO EXCEPTIONS:
1. ONLY answer EchoShield/deepfake/audio detection questions
2. REFUSE ALL other topics including: travel, finance, RBI rates, hotels, restaurants, shopping, health, politics, general knowledge, current events, recommendations
3. If asked ANYTHING unrelated, respond: "I'm EchoBot for EchoShield. I only answer questions about deepfake audio detection, our model, and team. Please ask about EchoShield."
4. Keep answers under 100 words
5. No internet/external info
6. If unsure, say: "I'm not sure about that. For assistance, contact: Atharvsingh.edu@gmail.com"

ONLY THESE TOPICS: EchoShield features, model specs, team info, deepfake concepts, audio detection

REFUSE: Travel, finance, rates, hotels, food, shopping, health, politics, news, general questions"""
        
        label = self.context['prediction']['label']
        confidence = self.context['prediction']['confidence']
        transcript = self.context['transcript']
        real_prob = self.context['prediction']['probabilities']['real']
        fake_prob = self.context['prediction']['probabilities']['fake']
        
        return f"""You are EchoBot for EchoShield deepfake audio detection.

{echoshield_info}

ANALYSIS: {label.upper()} | Confidence: {confidence:.1%} | Real: {real_prob:.1%} | Fake: {fake_prob:.1%}
Transcript: "{transcript}"

STRICT RULES - NO EXCEPTIONS:
1. ONLY answer about this analysis, EchoShield, model, or team
2. REFUSE ALL other topics: travel, finance, rates, hotels, food, shopping, health, politics, news, general questions
3. If asked unrelated, respond: "I'm EchoBot for EchoShield. I only discuss this audio analysis and our detection system."
4. Keep answers under 100 words
5. Explain results clearly
6. If unsure, say: "I'm not sure. Contact: Atharvsingh.edu@gmail.com"

ONLY THESE: Current analysis, confidence meaning, model specs, team info

REFUSE: Everything else"""
    
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
