"""Configuration module - loads settings from environment variables."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'models/bilstm_model.keras')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')

# Audio Processing
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 16000))
N_MFCC = int(os.getenv('N_MFCC', 40))
MAX_AUDIO_LENGTH = int(os.getenv('MAX_AUDIO_LENGTH', 300))

# Groq API
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

# Flask Configuration
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

# Dataset Paths
TRAIN_PATH = os.getenv('TRAIN_PATH', './scenefake/train')
DEV_PATH = os.getenv('DEV_PATH', './scenefake/dev')
TEST_PATH = os.getenv('TEST_PATH', './scenefake/eval')

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

# Max file size (50MB)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024
