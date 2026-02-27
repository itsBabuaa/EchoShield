# EchoShield

A Flask-based web application that detects deepfake audio using a BiLSTM (Bidirectional Long Short-Term Memory) neural network model. The system allows users to upload audio files or record audio directly through the web interface and provides real-time predictions on whether the audio is authentic or synthetically generated.

## Features

- **Audio Upload**: Support for WAV, MP3, and FLAC formats
- **Live Recording**: Record audio directly in the browser
- **BiLSTM Model**: Deep learning model for accurate detection
- **Visual Results**: Animated confidence gauge and color-coded results
- **Transcription**: Automatic speech-to-text conversion
- **AI Chatbot**: Interactive assistant powered by Groq API
- **Futuristic UI**: Dark theme with gradients and animations
- **Responsive Design**: Works on desktop and mobile devices

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Safari, Edge) for recording feature

### Setup

1. **Clone or download the repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   
   Create a `.env` file in the project root:
   ```bash
   # Model Configuration
   MODEL_PATH=models/bilstm_model.keras
   UPLOAD_FOLDER=uploads

   # Audio Processing
   SAMPLE_RATE=16000
   N_MFCC=40
   MAX_AUDIO_LENGTH=300

   # API Keys
   GROQ_API_KEY=your_groq_api_key_here
   DEEPGRAM_API_KEY=your_deepgram_api_key_here

   # Flask Configuration
   FLASK_DEBUG=False
   SECRET_KEY=your-secret-key-here

   # Dataset Paths (for training)
   TRAIN_PATH=./scenefake/train
   DEV_PATH=./scenefake/dev
   TEST_PATH=./scenefake/test
   ```

4. **Train the model** (if not already trained):
   ```bash
   python train_model.py
   ```
   
   Note: You'll need the SceneFake dataset in the `scenefake/` directory with `train/`, `dev/`, and `test/` subdirectories, each containing `real/` and `fake/` folders.

## Usage

### Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Using the Application

#### Upload Audio
1. Drag and drop an audio file or click to browse
2. Click "Analyze Audio"
3. View the results with confidence scores and transcript

#### Record Audio
1. Click the record button
2. Allow microphone access when prompted
3. Click "Stop Recording" when done
4. Click "Analyze Recording"

#### Chat with AI Assistant
- Ask questions about the detection results
- Get explanations about confidence scores
- Learn about deepfake detection technology

## Project Structure

```
.
├── app.py                      # Flask application
├── audio_processor.py          # Audio loading and feature extraction
├── prediction_engine.py        # Model loading and prediction
├── transcriber.py              # Speech-to-text transcription
├── chatbot.py                  # AI chatbot using Groq API
├── train_model.py              # Model training script
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── models/                     # Trained models directory
│   └── bilstm_model.keras
├── templates/                  # HTML templates
│   ├── base.html
│   └── index.html
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
└── uploads/                    # Temporary upload directory
```

## Model Architecture

The BiLSTM model consists of:
- 2 Bidirectional LSTM layers with batch normalization
- Dropout layers for regularization
- Dense layers with ReLU activation
- Softmax output layer for binary classification

**Input**: MFCC features (40 coefficients, 300 time frames)
**Output**: Binary classification (Real/Fake) with confidence scores

## API Endpoints

### `GET /`
Serve the main application page

### `POST /predict`
Analyze uploaded audio file
- **Input**: Audio file (multipart/form-data)
- **Output**: JSON with prediction and transcript

### `POST /transcribe`
Transcribe audio to text
- **Input**: Audio file (multipart/form-data)
- **Output**: JSON with transcript

### `POST /chat`
Send message to AI chatbot
- **Input**: JSON with message
- **Output**: JSON with chatbot response

### `POST /chat/clear`
Clear chatbot conversation history

## Configuration

All configuration is managed through environment variables in the `.env` file:

- `MODEL_PATH`: Path to the trained model
- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `N_MFCC`: Number of MFCC coefficients (default: 40)
- `MAX_AUDIO_LENGTH`: Maximum audio length in frames (default: 300)
- `GROQ_API_KEY`: API key for Groq chatbot (optional)

## Troubleshooting

### Model not found
- Ensure you've trained the model using `python train_model.py`
- Check that `MODEL_PATH` in `.env` points to the correct location

### Chatbot not working
- Set `GROQ_API_KEY` in your `.env` file
- Get an API key from [Groq](https://console.groq.com/)

### Transcription not working
- Set `DEEPGRAM_API_KEY` in your `.env` file
- Get an API key from [Deepgram](https://console.deepgram.com/)

### Microphone access denied
- Check browser permissions for microphone access
- Use HTTPS in production (required for microphone API)

### File upload fails
- Check file size (max 16MB)
- Ensure file format is WAV, MP3, or FLAC

## Technologies Used

- **Backend**: Flask, TensorFlow/Keras, Librosa
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML Model**: BiLSTM with MFCC features
- **Speech Recognition**: Deepgram API
- **AI Chatbot**: Groq API

## Acknowledgments

- SceneFake dataset for training data
- TensorFlow/Keras for deep learning framework
- Librosa for audio processing
- Groq for AI chatbot capabilities
