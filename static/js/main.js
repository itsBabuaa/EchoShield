// Deepfake Audio Detector - Frontend JavaScript

// Global variables
let selectedFile = null;
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const audioFileInput = document.getElementById('audioFile');
const uploadBtn = document.getElementById('uploadBtn');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const recordingStatus = document.getElementById('recordingStatus');
const analyzeRecordingContainer = document.getElementById('analyzeRecordingContainer');
const analyzeRecordingBtn = document.getElementById('analyzeRecordingBtn');
const resultsContainer = document.getElementById('resultsContainer');
const loadingOverlay = document.getElementById('loadingOverlay');
const chatInput = document.getElementById('chatInput');
const sendChatBtn = document.getElementById('sendChatBtn');
const chatMessages = document.getElementById('chatMessages');
const clearChatBtn = document.getElementById('clearChatBtn');
const alertContainer = document.getElementById('alertContainer');

// Audio playback elements
const audioPlaybackCard = document.getElementById('audioPlaybackCard');
const audioPlayer = document.getElementById('audioPlayer');
const audioFilename = document.getElementById('audioFilename');
const audioDuration = document.getElementById('audioDuration');

// ============================================================================
// File Upload Functionality
// ============================================================================

// Click to upload
uploadArea.addEventListener('click', () => {
    audioFileInput.click();
});

// File selection
audioFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac'];
    const validExtensions = ['.wav', '.mp3', '.flac'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        showAlert('Please select a valid audio file (WAV, MP3, or FLAC)', 'error');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size exceeds 16MB limit', 'error');
        return;
    }
    
    selectedFile = file;
    uploadBtn.disabled = false;
    
    // Update upload area text
    const uploadText = uploadArea.querySelector('.upload-text');
    uploadText.textContent = `Selected: ${file.name}`;
    uploadText.style.color = 'var(--primary-color)';
}

// Upload and analyze
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    await analyzeAudio(selectedFile);
});

// ============================================================================
// Audio Recording Functionality
// ============================================================================

recordBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            // Create blob from recorded chunks
            const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
            
            // Convert to WAV format in browser
            try {
                recordedBlob = await convertToWav(webmBlob);
                recordingStatus.textContent = 'Recording complete! Click "Analyze Recording" to proceed.';
                recordingStatus.style.color = 'var(--success-color)';
            } catch (error) {
                console.error('Conversion error:', error);
                // Fallback: use original blob
                recordedBlob = webmBlob;
                recordingStatus.textContent = 'Recording complete (may need FFmpeg on server)';
                recordingStatus.style.color = 'var(--warning-color)';
            }
            
            analyzeRecordingContainer.style.display = 'block';
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        
        // Update UI
        recordBtn.classList.add('recording');
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        recordingStatus.textContent = 'Recording... Click "Stop Recording" when done.';
        recordingStatus.style.color = 'var(--danger-color)';
        analyzeRecordingContainer.style.display = 'none';
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showAlert('Could not access microphone. Please check permissions.', 'error');
    }
});

stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        
        // Update UI
        recordBtn.classList.remove('recording');
        recordBtn.disabled = false;
        stopBtn.disabled = true;
    }
});

analyzeRecordingBtn.addEventListener('click', async () => {
    if (!recordedBlob) return;
    
    // Convert blob to file
    const file = new File([recordedBlob], 'recording.wav', { type: 'audio/wav' });
    await analyzeAudio(file);
});

// ============================================================================
// Audio Analysis
// ============================================================================

async function analyzeAudio(file) {
    // Show loading
    loadingOverlay.style.display = 'flex';
    resultsContainer.classList.remove('show');
    
    try {
        const formData = new FormData();
        formData.append('audio', file);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Setup audio playback
            setupAudioPlayback(file, file.name || 'audio.wav');
            
            // Display results
            displayResults(data.prediction, data.transcript);
        } else {
            showAlert(data.error || 'Analysis failed', 'error');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('An error occurred during analysis', 'error');
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

// ============================================================================
// Results Display
// ============================================================================

function displayResults(prediction, transcript) {
    // Update result badge
    const resultBadge = document.getElementById('resultBadge');
    const label = prediction.label.toUpperCase();
    resultBadge.textContent = label;
    resultBadge.className = 'result-badge ' + prediction.label;
    
    // Update confidence gauge
    const confidence = Math.round(prediction.confidence * 100);
    updateGauge(confidence);
    
    // Update probabilities
    document.getElementById('realProb').textContent = 
        Math.round(prediction.probabilities.real * 100) + '%';
    document.getElementById('fakeProb').textContent = 
        Math.round(prediction.probabilities.fake * 100) + '%';
    
    // Update transcript
    document.getElementById('transcriptBox').textContent = transcript;
    
    // Show results
    resultsContainer.classList.add('show');
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setupAudioPlayback(audioSource, filename = 'audio.wav') {
    // Create object URL from blob or file
    let audioUrl;
    if (audioSource instanceof Blob) {
        audioUrl = URL.createObjectURL(audioSource);
    } else if (audioSource instanceof File) {
        audioUrl = URL.createObjectURL(audioSource);
    } else {
        return;
    }
    
    // Set audio source
    audioPlayer.src = audioUrl;
    audioFilename.textContent = filename;
    
    // Update duration when metadata loads
    audioPlayer.addEventListener('loadedmetadata', function() {
        const duration = audioPlayer.duration;
        const minutes = Math.floor(duration / 60);
        const seconds = Math.floor(duration % 60);
        audioDuration.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    });
    
    // Show audio playback card
    audioPlaybackCard.style.display = 'block';
    
    // Clean up old URL when new audio is loaded
    audioPlayer.addEventListener('ended', function() {
        URL.revokeObjectURL(audioUrl);
    });
}

function updateGauge(percentage) {
    const gaugeValue = document.getElementById('gaugeValue');
    const gaugeCircle = document.querySelector('.gauge-circle');
    
    gaugeValue.textContent = percentage + '%';
    
    // Animate gauge
    const angle = (percentage / 100) * 360;
    gaugeCircle.style.setProperty('--gauge-angle', angle + 'deg');
    
    // Change color based on confidence
    let color;
    if (percentage >= 80) {
        color = 'var(--success-color)';
    } else if (percentage >= 60) {
        color = 'var(--primary-color)';
    } else {
        color = 'var(--danger-color)';
    }
    
    gaugeValue.style.color = color;
}

// ============================================================================
// Chatbot Functionality
// ============================================================================

sendChatBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addChatMessage(message, 'user');
    chatInput.value = '';
    
    // Show loading
    const loadingMsg = addChatMessage('...', 'assistant', true);
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        const data = await response.json();
        
        // Remove loading message
        loadingMsg.remove();
        
        if (data.success) {
            addChatMessage(data.response, 'assistant');
        } else {
            if (data.error.includes('not configured')) {
                document.getElementById('chatError').style.display = 'block';
            }
            addChatMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        loadingMsg.remove();
        addChatMessage('Sorry, I encountered an error. Please try again.', 'assistant');
    }
}

function addChatMessage(text, role, isLoading = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    
    if (isLoading) {
        messageDiv.innerHTML = '<span class="loading"></span>';
    } else {
        messageDiv.textContent = text;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

clearChatBtn.addEventListener('click', async () => {
    try {
        await fetch('/chat/clear', { method: 'POST' });
        
        // Clear messages except welcome message
        chatMessages.innerHTML = `
            <div class="chat-message assistant">
                Hello! I'm EchoBot, your AI assistant. I'm here to help you understand the audio analysis results. Feel free to ask me any questions about the detection, confidence scores, or deepfake technology in general.
            </div>
        `;
        
        showAlert('Chat history cleared', 'success');
    } catch (error) {
        console.error('Error clearing chat:', error);
    }
});

// ============================================================================
// Alert System
// ============================================================================

function showAlert(message, type = 'info') {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    
    alertContainer.appendChild(alert);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 300);
    }, 5000);
}

// ============================================================================
// Audio Format Conversion
// ============================================================================

async function convertToWav(blob) {
    return new Promise((resolve, reject) => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const fileReader = new FileReader();
        
        fileReader.onload = async (e) => {
            try {
                // Decode audio data
                const audioBuffer = await audioContext.decodeAudioData(e.target.result);
                
                // Convert to WAV
                const wavBlob = audioBufferToWav(audioBuffer);
                resolve(wavBlob);
            } catch (error) {
                reject(error);
            }
        };
        
        fileReader.onerror = () => reject(new Error('Failed to read audio file'));
        fileReader.readAsArrayBuffer(blob);
    });
}

function audioBufferToWav(audioBuffer) {
    const numberOfChannels = 1; // Mono
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    // Get audio data (convert to mono if needed)
    let audioData;
    if (audioBuffer.numberOfChannels === 1) {
        audioData = audioBuffer.getChannelData(0);
    } else {
        // Mix down to mono
        const left = audioBuffer.getChannelData(0);
        const right = audioBuffer.getChannelData(1);
        audioData = new Float32Array(left.length);
        for (let i = 0; i < left.length; i++) {
            audioData[i] = (left[i] + right[i]) / 2;
        }
    }
    
    // Convert float samples to 16-bit PCM
    const samples = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        const s = Math.max(-1, Math.min(1, audioData[i]));
        samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    
    // Create WAV file
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, format, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numberOfChannels * bitDepth / 8, true);
    view.setUint16(32, numberOfChannels * bitDepth / 8, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    // Write PCM samples
    const offset = 44;
    for (let i = 0; i < samples.length; i++) {
        view.setInt16(offset + i * 2, samples[i], true);
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// ============================================================================
// Initialize
// ============================================================================

console.log('Deepfake Audio Detector initialized');
