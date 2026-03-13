"""
Context-Aware Chatbot Component for Deepfake Audio Detection

This module provides an AI-powered chatbot using Groq API that maintains
full awareness of the current and previous audio analysis results,
conversation history with a sliding window, and audio metrics context.
"""

from groq import Groq
from typing import List, Dict, Optional
import config


class Chatbot:
    """
    Context-aware chatbot for interactive audio analysis using Groq API.
    Maintains a sliding conversation window and tracks analysis history
    so the bot can reference current and previous results naturally.
    """

    MAX_HISTORY_TURNS = 20  # keep last N user+assistant pairs

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY in .env file."
            )

        self.client = Groq(api_key=self.api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.context: Optional[Dict] = None
        self.analysis_history: List[Dict] = []
        self.model = "llama-3.1-8b-instant"

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def set_context(self, transcript: str, prediction: Dict, audio_metrics: Dict = None):
        """
        Set or update audio analysis context. Archives the previous context
        so the bot can reference earlier analyses when the user asks.
        """
        if self.context is not None:
            self.analysis_history.append(self.context)

        self.context = {
            'transcript': transcript,
            'prediction': prediction,
            'audio_metrics': audio_metrics or {}
        }
        self._rebuild_system_message()

    def _rebuild_system_message(self):
        """Rebuild system message and inject it at the front of history."""
        system_message = self._create_system_message()
        self.conversation_history = [
            msg for msg in self.conversation_history if msg['role'] != 'system'
        ]
        self.conversation_history.insert(0, {
            "role": "system",
            "content": system_message
        })

    def _create_system_message(self) -> str:
        echoshield_info = (
            "ECHOSHIELD: Deepfake audio detection system\n"
            "MODEL: BiLSTM Neural Network | Accuracy: 93.8% | Precision: 93.7% | "
            "Recall: 99.0% | F1: 96.3% | AUC-ROC: 88.6%\n"
            "ARCHITECTURE: 2 BiLSTM layers (64->32 units), Dropout 0.4, Dense 64, Softmax\n"
            "FEATURES: 40 MFCC coefficients, 300 time frames\n"
            "DATASET: SceneFake (13,185 train, 12,842 val samples)\n\n"
            "TEAM:\n"
            "- Dr. Sonali Mathur (Supervisor) - sonali.mathur@imsec.ac.in\n"
            "- Atharv Singh (Lead Dev) - Model training, Flask API - @itsBabuaa\n"
            "- Ayush Pratap Singh (Frontend) - Web interface, design - @itsBabuaa\n"
            "- Ansh Srivastava (Backend) - API, audio processing - @itsBabuaa\n"
            "- Anushka Singh (Data Analyst) - Dataset analysis - @itsBabuaa"
        )

        if not self.context:
            return self._base_prompt(echoshield_info, has_analysis=False)

        p = self.context['prediction']
        label = p['label']
        confidence = p['confidence']
        real_prob = p['probabilities']['real']
        fake_prob = p['probabilities']['fake']
        transcript = self.context['transcript']
        metrics_section = self._format_audio_metrics(self.context.get('audio_metrics', {}))
        history_section = self._format_analysis_history()

        return (
            f"{self._base_prompt(echoshield_info, has_analysis=True)}\n\n"
            f"CURRENT AUDIO ANALYSIS (latest):\n"
            f"- Detection Result: {label.upper()}\n"
            f"- Confidence: {confidence:.1%}\n"
            f"- Real Probability: {real_prob:.1%}\n"
            f"- Fake Probability: {fake_prob:.1%}\n"
            f"{metrics_section}\n"
            f'TRANSCRIPT: "{transcript}"\n'
            f"{history_section}"
        )

    def _base_prompt(self, echoshield_info: str, has_analysis: bool) -> str:
        context_note = (
            "You have access to the user's latest audio analysis results below. "
            "Reference them naturally when answering. If the user has analyzed "
            "multiple files, you can compare results from the PREVIOUS ANALYSES section."
            if has_analysis
            else "No audio has been analyzed yet in this session. Encourage the user "
                 "to upload or record audio so you can help them understand the results."
        )

        return (
            f"You are EchoBot, a friendly and helpful AI assistant for EchoShield "
            f"deepfake audio detection system.\n\n"
            f"{echoshield_info}\n\n"
            f"{context_note}\n\n"
            "YOUR PERSONALITY:\n"
            "- Friendly, conversational, and approachable\n"
            "- Expert in audio analysis and deepfake detection\n"
            "- Patient when explaining technical concepts\n\n"
            "YOUR EXPERTISE:\n"
            "- Audio analysis results and what they mean\n"
            "- All audio metrics and their significance\n"
            "- How the BiLSTM model makes predictions\n"
            "- EchoShield features and capabilities\n"
            "- Deepfake detection technology\n"
            "- Audio signal processing concepts\n"
            "- Team information\n\n"
            "GUIDELINES:\n"
            "1. Help users understand their audio analysis results thoroughly\n"
            "2. Explain audio metrics clearly when asked\n"
            "3. Be conversational - acknowledge questions, use friendly language\n"
            "4. STRICTLY REFUSE any off-topic questions. For ANY question not related "
            "to EchoShield, deepfake detection, audio analysis, or the team, respond "
            "ONLY with: \"I'm EchoBot - I only help with EchoShield and deepfake audio "
            "detection. Try asking about your analysis results or how our model works!\"\n"
            "5. NEVER answer questions about people, places, sports, politics, coding, "
            "math, general knowledge, or anything outside your expertise\n"
            "6. Keep answers clear and accessible (under 200 words)\n"
            "7. Connect metrics to real-world meaning\n"
            "8. When the user asks about a previous analysis, refer to the PREVIOUS "
            "ANALYSES section\n"
            "9. If unsure, be honest: \"I'm not certain about that. Contact: "
            "Atharvsingh.edu@gmail.com\"\n\n"
            "TONE: Friendly expert - like a helpful colleague explaining your results"
        )

    def _format_audio_metrics(self, audio_metrics: Dict) -> str:
        if not audio_metrics:
            return ""

        duration = audio_metrics.get('duration', 0)
        sample_rate = audio_metrics.get('sample_rate', 0)
        file_size = audio_metrics.get('file_size', 0)
        peak_amplitude = audio_metrics.get('peak_amplitude', 0)
        rms_level = audio_metrics.get('rms_level', 0)
        dynamic_range = audio_metrics.get('dynamic_range', 0)
        zero_crossings = audio_metrics.get('zero_crossings', 0)
        spectral_centroid = audio_metrics.get('spectral_centroid', 0)
        noise_floor = audio_metrics.get('noise_floor', 0)

        if file_size >= 1024 * 1024:
            file_size_str = f"{file_size / (1024 * 1024):.2f} MB"
        elif file_size >= 1024:
            file_size_str = f"{file_size / 1024:.2f} KB"
        else:
            file_size_str = f"{file_size} bytes"

        sample_rate_str = (
            f"{sample_rate / 1000:.1f} kHz" if sample_rate >= 1000
            else f"{sample_rate} Hz"
        )

        return (
            f"\nAUDIO FILE METRICS:\n"
            f"- Duration: {duration:.2f}s | Sample Rate: {sample_rate_str} | "
            f"File Size: {file_size_str}\n\n"
            f"ADVANCED AUDIO ANALYSIS:\n"
            f"- Peak Amplitude: {peak_amplitude:.4f} | RMS Level: {rms_level:.1f} dB\n"
            f"- Dynamic Range: {dynamic_range:.1f} dB | "
            f"Zero Crossings: {zero_crossings:,}/s\n"
            f"- Spectral Centroid: {spectral_centroid:,} Hz | "
            f"Noise Floor: {noise_floor:.1f} dB\n"
        )

    def _format_analysis_history(self) -> str:
        if not self.analysis_history:
            return ""

        lines = ["\nPREVIOUS ANALYSES (oldest first):"]
        for i, ctx in enumerate(self.analysis_history[-5:], 1):
            p = ctx['prediction']
            t = ctx['transcript']
            snippet = f"{t[:80]}..." if len(t) > 80 else t
            lines.append(
                f"  #{i}: {p['label'].upper()} ({p['confidence']:.1%}) "
                f'- "{snippet}"'
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def _trim_history(self):
        """Keep conversation history within the sliding window."""
        non_system = [m for m in self.conversation_history if m['role'] != 'system']
        if len(non_system) > self.MAX_HISTORY_TURNS * 2:
            trimmed = non_system[-(self.MAX_HISTORY_TURNS * 2):]
            system_msgs = [m for m in self.conversation_history if m['role'] == 'system']
            self.conversation_history = system_msgs + trimmed

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a context-aware response from the chatbot.
        """
        if not user_message or not user_message.strip():
            return "Please provide a message."

        # Ensure system message exists
        if not self.conversation_history or self.conversation_history[0]['role'] != 'system':
            self._rebuild_system_message()

        # On-topic check
        if not self._is_on_topic(user_message):
            return (
                "I'm EchoBot, and I only handle questions about EchoShield "
                "and deepfake audio detection. I can help with:\n"
                "- Your audio analysis results\n"
                "- How our detection model works\n"
                "- Deepfake audio concepts\n"
                "- EchoShield team and features\n\n"
                "Try asking something like: \"Is my audio fake?\" or "
                "\"How does the model work?\""
            )

        try:
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self._trim_history()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.5,
                max_tokens=256,
                top_p=0.9,
                stream=False
            )

            assistant_message = response.choices[0].message.content

            # Block code-generation responses
            if '```' in assistant_message or 'def ' in assistant_message:
                self.conversation_history.pop()
                return (
                    "I'm EchoBot, specialized in EchoShield and deepfake audio "
                    "detection. I cannot help with coding or unrelated tasks. "
                    "Please ask me about EchoShield, audio analysis, or deepfake detection."
                )

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            return assistant_message

        except Exception as e:
            if "rate_limit" in str(e).lower():
                return "Too many requests. Please wait a moment and try again."
            elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                return (
                    "Chatbot service not configured properly. "
                    "Please contact our team at Atharvsingh.edu@gmail.com"
                )
            else:
                return (
                    "Chatbot temporarily unavailable. "
                    "Please try again or contact us at Atharvsingh.edu@gmail.com"
                )

    @staticmethod
    def _is_on_topic(user_message: str) -> bool:
        """Check whether the user message is related to EchoShield / audio / deepfakes."""
        msg = user_message.lower()

        on_topic_keywords = [
            'audio', 'recording', 'transcript', 'speech', 'voice', 'sound',
            'said', 'saying', 'talk', 'talking', 'speak', 'speaking',
            'hear', 'heard', 'listen', 'content', 'file', 'clip',
            'echoshield', 'echo shield', 'echobot', 'deepfake', 'deep fake',
            'detection', 'fake', 'real', 'authentic', 'spoof', 'cloned',
            'model', 'accuracy', 'precision', 'recall', 'f1', 'auc',
            'confidence', 'analysis', 'result', 'prediction', 'probability',
            'team', 'creator', 'developer', 'who made', 'who built',
            'bilstm', 'lstm', 'mfcc', 'neural network', 'machine learning',
            'upload', 'detect', 'analyze', 'classification',
            'sample rate', 'frequency', 'spectrogram', 'waveform', 'noise',
            'amplitude', 'decibel', 'spectral', 'signal', 'feature',
            'rms', 'dynamic range', 'zero crossing',
            'hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'help',
            'previous', 'last', 'history', 'compare', 'earlier',
            'report', 'forensic', 'download',
        ]

        if any(kw in msg for kw in on_topic_keywords):
            return True
        return len(user_message.strip()) <= 10

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def clear_history(self):
        """Clear conversation history while preserving analysis context."""
        self.conversation_history = []
        self._rebuild_system_message()

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history without system message."""
        return [msg for msg in self.conversation_history if msg['role'] != 'system']
