"""
Flask Application for Deepfake Audio Detection
with routes for audio upload, prediction, transcription, and chatbot.
"""

import os
import gc

# Memory optimization settings - must be set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

# Limit TensorFlow memory usage
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "0"

# Set memory allocator for better memory management
os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "100000"

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid

# Import backend components
from audio_processor import AudioProcessor
from prediction_engine import PredictionEngine
from transcriber import Transcriber
from chatbot import Chatbot
import config

# Reduce TensorFlow overhead
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Preload all components at startup (before first request)
print("Loading EchoShield components...")
audio_processor = AudioProcessor()
prediction_engine = PredictionEngine()
transcriber = Transcriber()
print("✓ All components loaded and ready")

# Store chatbot instances per session (with limit to prevent memory leaks)
chatbots = {}
MAX_CHATBOT_SESSIONS = 100  # Limit concurrent chatbot sessions


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def cleanup_old_chatbots():
    """Remove oldest chatbot sessions if limit is exceeded."""
    if len(chatbots) > MAX_CHATBOT_SESSIONS:
        # Remove oldest 20% of sessions
        sessions_to_remove = list(chatbots.keys())[:MAX_CHATBOT_SESSIONS // 5]
        for session_id in sessions_to_remove:
            del chatbots[session_id]
        gc.collect()


def get_chatbot():
    """Get or create chatbot for current session."""
    session_id = session.get('session_id')
    
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    if session_id not in chatbots:
        try:
            # Clean up old sessions before creating new one
            cleanup_old_chatbots()
            chatbots[session_id] = Chatbot()
        except ValueError:
            # API key not configured
            return None
    
    return chatbots[session_id]


@app.route('/')
def home():
    """
    Serve home page with information about deepfakes.
    """
    return render_template('home.html')


@app.route('/detect')
def detect():
    """
    Serve detection page with upload and recording interface.
    """
    return render_template('detect.html')


@app.route('/learn-more')
def learn_more():
    """
    Serve detailed information page about deepfakes.
    """
    return render_template('learn_more.html')


@app.route('/model-info')
def model_info():
    """
    Serve model information page with performance metrics.
    """
    model_metrics = {
        # Actual metrics from model evaluation on 32,746 test samples
        'accuracy': 93.8,
        'precision': 93.7,
        'recall': 99.0,
        'f1_score': 96.3,
        'auc_roc': 88.6,
        
        # Actual dataset sizes
        'training_samples': 13185,
        'validation_samples': 12842,
        'test_samples': 32746,
        
        # Training configuration from train_model.py
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.0001,
        
        # Actual model parameters (trainable)
        'parameters': 102306,
        'architecture': {
            'input_shape': '(300, 40)',
            'layers': [
                'Input Layer (300, 40)',
                'Bidirectional LSTM (64 units, return_sequences=True, recurrent_dropout=0.2)',
                'Batch Normalization',
                'Dropout (0.4)',
                'Bidirectional LSTM (32 units, recurrent_dropout=0.2)',
                'Batch Normalization',
                'Dropout (0.4)',
                'Dense (64 units, ReLU activation)',
                'Batch Normalization',
                'Dropout (0.4)',
                'Dense (32 units, ReLU activation)',
                'Dropout (0.4)',
                'Dense (2 units, Softmax activation)'
            ]
        },
        # Feature extraction from config.py and train_model.py
        'features': {
            'type': 'MFCC',
            'n_coefficients': 40,
            'sample_rate': 16000,
            'max_length': 300
        },
        # Optimizer configuration from train_model.py
        'optimizer': {
            'type': 'Adam',
            'learning_rate': 0.0001,
            'clipnorm': 1.0
        },
        'loss_function': 'SCC',
        'callbacks': [
            'EarlyStopping (patience=10)',
            'ModelCheckpoint (save_best_only=True)',
            'ReduceLROnPlateau (factor=0.5, patience=5)'
        ]
    }
    return render_template('model_info.html', metrics=model_metrics)


@app.route('/about')
def about():
    """
    Serve about us page with team information.
    """
    team_members = [
        {
            'name': 'Dr. Sonali Mathur',
            'role': 'Project Supervisor',
            'contribution': 'Supervised the team throughout the development of EchoShield, providing guidance on AI ethics, model architecture, and research methodology',
            'email': 'sonali.mathur@imsec.ac.in',
            'github': None,
            'photo': 'Sonali-Mathur.jpg'
        },
        {
            'name': 'Atharv Singh',
            'role': 'Lead Developer',
            'contribution': 'Trained the BiLSTM model, developed Flask backend with RESTful APIs, integrated the AI chatbot system, and coordinated overall development of the application',
            'email': 'atharvsingh.edu@gmail.com',
            'github': 'itsBabuaa',
            'photo': 'Atharv.jpeg'
        },
        {
            'name': 'Ayush Pratap Singh',
            'role': 'Back-End Developer',
            'contribution': 'Developed the Flask backend architecture, API endpoints, and integrated audio processing pipeline with the prediction engine',
            'email': 'ayushrajawat2005@gmail.com',
            'github': 'AyushPratap05',
            'photo': 'Ayush.jpeg'
        },
        {
            'name': 'Ansh Srivastava',
            'role': 'Front-End Developer',
            'contribution': 'Designed and built the complete user interface with responsive layouts, interactive components, and modern styling',
            'email': 'srivastavaansh1408@gmail.com',
            'github': 'Ansh-1401',
            'photo': 'Ansh.jpeg'
        },
        {
            'name': 'Anushka Singh',
            'role': 'Data Analyst',
            'contribution': 'Analyzed the SceneFake dataset, performed data preprocessing, and assisted in model training and validation processes',
            'email': 'singhanu8404@gmail.com',
            'github': None,
            'photo': 'team5.jpg'
        }
    ]
    return render_template('about.html', team_members=team_members)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle audio upload and prediction.
    
    Accepts audio file upload, processes it, and returns prediction results.

    
    Returns:
        JSON response with:
        - success: boolean
        - prediction: dict with label, confidence, probabilities
        - transcript: string
        - error: string (if failed)
    """
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        file = request.files['audio']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file format
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Unsupported audio format. Please upload WAV, MP3, or FLAC files.'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        
        # Handle recorded audio (browser sends as 'blob' or empty filename)
        if not filename or filename == 'blob':
            filename = 'recording.wav'
        
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Process audio and get prediction
            features = audio_processor.preprocess_for_prediction(filepath)
            prediction = prediction_engine.predict(features)
            
            # Get audio metrics
            audio_metrics = audio_processor.get_audio_metrics(filepath)
            
            # Get transcription (with error handling)
            try:
                transcript = transcriber.transcribe(filepath)
            except Exception as trans_error:
                print(f"Transcription error: {trans_error}")
                transcript = "[Transcription unavailable]"
            
            # Store results in session for chatbot
            session['last_prediction'] = prediction
            session['last_transcript'] = transcript
            session['last_audio_metrics'] = audio_metrics
            
            # Set chatbot context if available
            chatbot = get_chatbot()
            if chatbot:
                chatbot.set_context(transcript, prediction, audio_metrics)
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'transcript': transcript,
                'audio_metrics': audio_metrics
            })
        
        finally:
            # Clean up uploaded file and free memory
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Free memory
            del features, prediction, audio_metrics
            if 'transcript' in locals():
                del transcript
            gc.collect()
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Handle transcription requests.
    
    Requirements: 3.6
    
    Returns:
        JSON response with:
        - success: boolean
        - transcript: string
        - error: string (if failed)
    """
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Get transcription (with error handling)
            try:
                transcript = transcriber.transcribe(filepath)
            except Exception as trans_error:
                print(f"Transcription error: {trans_error}")
                transcript = "[Transcription unavailable]"
            
            return jsonify({
                'success': True,
                'transcript': transcript
            })
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chatbot messages.
    
    Requirements: 6.1
    
    Returns:
        JSON response with:
        - success: boolean
        - response: string (chatbot response)
        - error: string (if failed)
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        user_message = data['message']
        
        # Get chatbot for this session
        chatbot = get_chatbot()
        
        if not chatbot:
            return jsonify({
                'success': False,
                'error': 'Chatbot service not configured. Please set GROQ_API_KEY in .env file.'
            }), 503
        
        # Get response
        response = chatbot.chat(user_message)
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    """
    Clear chatbot conversation history.
    
    Returns:
        JSON response with success status
    """
    try:
        chatbot = get_chatbot()
        
        if chatbot:
            chatbot.clear_history()
            
            # Re-set context if available
            if 'last_prediction' in session and 'last_transcript' in session:
                audio_metrics = session.get('last_audio_metrics', None)
                chatbot.set_context(session['last_transcript'], session['last_prediction'], audio_metrics)
        
        return jsonify({
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': 'File size exceeds maximum limit (50MB).'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again.'
    }), 500

# For Local Host
# if __name__ == '__main__':
#     app.run(
#         debug=config.FLASK_DEBUG,
#         host='0.0.0.0',
#         port=5000
#     )

# For Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",   # THIS IS THE KEY FIX
        port=port,
        debug=False
    )







