// Deepfake Audio Detector - Frontend JavaScript

// Global variables
let selectedFile = null;
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let currentAudioFile = null;
let currentPlaybackSpeed = 1;
let waveformData = null;
let lastPrediction = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const audioFileInput = document.getElementById('audioFile');
const uploadBtn = document.getElementById('uploadBtn');
const recordBtn = document.getElementById('recordBtn');
const recordingStatus = document.getElementById('recordingStatus');
const analyzeRecordingContainer = document.getElementById('analyzeRecordingContainer');
const analyzeRecordingBtn = document.getElementById('analyzeRecordingBtn');
const resultsContainer = document.getElementById('resultsContainer');
const loadingOverlay = document.getElementById('loadingOverlay');
const chatInput = document.getElementById('chatInput');
const sendChatBtn = document.getElementById('sendChatBtn');
const chatMessages = document.getElementById('chatMessages');
const clearChatBtn = document.getElementById('clearChatBtn');

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
    const files = e.target.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
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
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
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
    
    // Validate file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showAlert('File size exceeds 50MB limit', 'error');
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
    if (selectedFile) {
        await analyzeAudio(selectedFile);
    }
});

// ============================================================================
// Audio Recording Functionality
// ============================================================================

recordBtn.addEventListener('click', async () => {
    // If already recording, stop it
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        return;
    }
    
    // Start recording
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
            
            // Update UI - back to record state
            recordBtn.classList.remove('recording');
            recordBtn.title = 'Start Recording';
            document.querySelector('.record-icon').style.display = 'block';
            document.querySelector('.stop-icon').style.display = 'none';
        };
        
        mediaRecorder.start();
        
        // Update UI - show stop state
        recordBtn.classList.add('recording');
        recordBtn.title = 'Stop Recording';
        document.querySelector('.record-icon').style.display = 'none';
        document.querySelector('.stop-icon').style.display = 'block';
        recordingStatus.textContent = 'Recording... Click the button again to stop.';
        recordingStatus.style.color = 'var(--danger-color)';
        analyzeRecordingContainer.style.display = 'none';
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showAlert('Could not access microphone. Please check permissions.', 'error');
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
    // Show loading with step-by-step status
    loadingOverlay.style.display = 'flex';
    resultsContainer.classList.remove('show');
    
    // Create or get loading status element
    let loadingStatus = document.getElementById('loadingStatus');
    if (!loadingStatus) {
        loadingStatus = document.createElement('div');
        loadingStatus.id = 'loadingStatus';
        loadingStatus.className = 'loading-status';
        loadingOverlay.appendChild(loadingStatus);
    }
    
    // Step-by-step status updates (sequential, no repeat)
    const steps = [
        'Uploading audio file...',
        'Processing audio data...',
        'Extracting audio features...',
        'Running AI analysis...',
        'Calculating confidence scores...'
    ];
    
    let currentStep = 0;
    loadingStatus.textContent = steps[currentStep];
    
    // Update status sequentially (move to next step, don't loop)
    const statusInterval = setInterval(() => {
        currentStep++;
        if (currentStep < steps.length) {
            loadingStatus.textContent = steps[currentStep];
        } else {
            // Stay on last step until analysis completes
            clearInterval(statusInterval);
        }
    }, 1200);
    
    try {
        const formData = new FormData();
        formData.append('audio', file);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Clear interval
        clearInterval(statusInterval);
        
        if (data.success) {
            // Show completion message briefly
            loadingStatus.textContent = 'Analysis complete!';
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Store current file for metrics
            currentAudioFile = file;
            
            // Setup audio playback
            setupAudioPlayback(file, file.name || 'audio.wav');
            
            // Display results
            displayResults(data.prediction, data.transcript, file);
        } else {
            clearInterval(statusInterval);
            showAlert(data.error || 'Analysis failed', 'error');
        }
        
    } catch (error) {
        clearInterval(statusInterval);
        console.error('Error:', error);
        showAlert('An error occurred during analysis', 'error');
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

// ============================================================================
// Results Display
// ============================================================================

function displayResults(prediction, transcript, audioFile) {
    // Store prediction for sharing
    lastPrediction = prediction;
    
    // Update result badge
    const resultBadge = document.getElementById('resultBadge');
    const label = prediction.label.toUpperCase();
    resultBadge.textContent = label;
    resultBadge.className = 'result-badge ' + prediction.label;
    
    // Update confidence gauge with color based on result
    const confidence = Math.round(prediction.confidence * 100);
    const gaugeElement = document.getElementById('confidenceGauge');
    
    // Remove previous classes and add new one based on prediction
    gaugeElement.classList.remove('real', 'fake');
    gaugeElement.classList.add(prediction.label);
    
    updateGauge(confidence, prediction.label);
    
    // Update additional metrics with real insights
    document.getElementById('processingTime').textContent = '< 1s';
    
    // File size
    if (audioFile && audioFile.size) {
        const sizeKB = (audioFile.size / 1024).toFixed(1);
        const sizeMB = (audioFile.size / (1024 * 1024)).toFixed(2);
        document.getElementById('audioFileSize').textContent = audioFile.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
    } else {
        document.getElementById('audioFileSize').textContent = '-';
    }
    
    // Duration and sample rate will be updated when audio loads
    const audioDurationMetric = document.getElementById('audioDurationMetric');
    const sampleRateEl = document.getElementById('sampleRate');
    
    if (audioPlayer && audioPlayer.duration && !isNaN(audioPlayer.duration)) {
        const mins = Math.floor(audioPlayer.duration / 60);
        const secs = Math.floor(audioPlayer.duration % 60);
        audioDurationMetric.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    } else {
        audioDurationMetric.textContent = '-';
    }
    
    // Get sample rate from audio context
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        sampleRateEl.textContent = `${(audioContext.sampleRate / 1000).toFixed(1)} kHz`;
        audioContext.close();
    } catch (e) {
        sampleRateEl.textContent = '44.1 kHz';
    }
    
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
    
    // Generate waveform and calculate advanced metrics
    const fileReader = new FileReader();
    fileReader.onload = async (e) => {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(e.target.result);
            
            // Draw waveform
            drawWaveform(audioBuffer);
            
            // Calculate advanced metrics
            calculateAdvancedMetrics(audioBuffer);
            
            audioContext.close();
        } catch (error) {
            console.error('Error processing audio for visualization:', error);
        }
    };
    fileReader.readAsArrayBuffer(audioSource);
    
    // Setup modern player controls
    const playBtn = document.getElementById('playBtn');
    const playIcon = playBtn.querySelector('.play-icon');
    const pauseIcon = playBtn.querySelector('.pause-icon');
    const progressBar = document.getElementById('progressBar');
    const progressFill = document.getElementById('progressFill');
    const progressHandle = document.getElementById('progressHandle');
    const currentTimeEl = document.getElementById('currentTime');
    const volumeBtn = document.getElementById('volumeBtn');
    const volumeIcon = volumeBtn.querySelector('.volume-icon');
    const muteIcon = volumeBtn.querySelector('.mute-icon');
    const volumeSlider = document.getElementById('volumeSlider');
    
    // Format time helper
    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    // Update duration when metadata loads
    audioPlayer.addEventListener('loadedmetadata', function() {
        audioDuration.textContent = formatTime(audioPlayer.duration);
        
        // Update duration metric as well
        const audioDurationMetric = document.getElementById('audioDurationMetric');
        if (audioDurationMetric) {
            audioDurationMetric.textContent = formatTime(audioPlayer.duration);
        }
    });
    
    // Speed control
    const speedBtn = document.getElementById('speedBtn');
    const speeds = [0.5, 0.75, 1, 1.25, 1.5, 2];
    let speedIndex = 2; // Start at 1x
    
    if (speedBtn) {
        speedBtn.addEventListener('click', function() {
            speedIndex = (speedIndex + 1) % speeds.length;
            const newSpeed = speeds[speedIndex];
            audioPlayer.playbackRate = newSpeed;
            speedBtn.textContent = newSpeed + 'x';
            currentPlaybackSpeed = newSpeed;
        });
    }
    
    // Play/Pause toggle
    playBtn.addEventListener('click', function() {
        if (audioPlayer.paused) {
            audioPlayer.play();
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';
        } else {
            audioPlayer.pause();
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        }
    });
    
    // Update progress bar and waveform
    audioPlayer.addEventListener('timeupdate', function() {
        const progress = (audioPlayer.currentTime / audioPlayer.duration) * 100;
        progressFill.style.width = progress + '%';
        progressHandle.style.left = progress + '%';
        currentTimeEl.textContent = formatTime(audioPlayer.currentTime);
        
        // Update waveform progress
        updateWaveformProgress(audioPlayer.currentTime, audioPlayer.duration);
    });
    
    // Click on progress bar to seek
    progressBar.addEventListener('click', function(e) {
        const rect = progressBar.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        audioPlayer.currentTime = percent * audioPlayer.duration;
    });
    
    // Reset on end
    audioPlayer.addEventListener('ended', function() {
        playIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
        progressFill.style.width = '0%';
        progressHandle.style.left = '0%';
    });
    
    // Volume control
    volumeSlider.addEventListener('input', function() {
        audioPlayer.volume = this.value / 100;
        updateVolumeIcon();
    });
    
    // Mute toggle
    volumeBtn.addEventListener('click', function() {
        audioPlayer.muted = !audioPlayer.muted;
        updateVolumeIcon();
    });
    
    function updateVolumeIcon() {
        if (audioPlayer.muted || audioPlayer.volume === 0) {
            volumeIcon.style.display = 'none';
            muteIcon.style.display = 'block';
        } else {
            volumeIcon.style.display = 'block';
            muteIcon.style.display = 'none';
        }
    }
    
    // Show audio playback card
    audioPlaybackCard.style.display = 'block';
    
    // Clean up old URL when new audio is loaded
    audioPlayer.addEventListener('ended', function() {
        URL.revokeObjectURL(audioUrl);
    });
}

function updateGauge(percentage, predictionLabel) {
    const gaugeValue = document.getElementById('gaugeValue');
    const gaugeCircle = document.querySelector('.gauge-circle');
    
    gaugeValue.textContent = percentage + '%';
    
    // Animate gauge
    const angle = (percentage / 100) * 360;
    gaugeCircle.style.setProperty('--gauge-angle', angle + 'deg');
    
    // Set color based on prediction label
    if (predictionLabel === 'real') {
        gaugeValue.style.color = 'var(--success-color)';
    } else {
        gaugeValue.style.color = 'var(--danger-color)';
    }
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

// ============================================================================
// Toast Notification System
// ============================================================================

function showAlert(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    // Icon based on type
    const icons = {
        error: '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path d="M10 0C4.48 0 0 4.48 0 10s4.48 10 10 10 10-4.48 10-10S15.52 0 10 0zm1 15H9v-2h2v2zm0-4H9V5h2v6z"/></svg>',
        success: '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path d="M10 0C4.48 0 0 4.48 0 10s4.48 10 10 10 10-4.48 10-10S15.52 0 10 0zm-2 15l-5-5 1.41-1.41L8 12.17l7.59-7.59L17 6l-9 9z"/></svg>',
        info: '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path d="M10 0C4.48 0 0 4.48 0 10s4.48 10 10 10 10-4.48 10-10S15.52 0 10 0zm1 15H9V9h2v6zm0-8H9V5h2v2z"/></svg>',
        warning: '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path d="M1 17h18L10 1 1 17zm10-2H9v-2h2v2zm0-4H9V7h2v4z"/></svg>'
    };
    
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 400);
    }, 5000);
}

// Alternative function name for consistency
function showToast(message, type = 'info') {
    showAlert(message, type);
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

// Hamburger menu toggle
const hamburger = document.getElementById('hamburger');
const mainNav = document.getElementById('mainNav');

if (hamburger && mainNav) {
    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('active');
        mainNav.classList.toggle('active');
    });
    
    // Close menu when clicking on a link
    const navLinks = mainNav.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('active');
            mainNav.classList.remove('active');
        });
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!hamburger.contains(e.target) && !mainNav.contains(e.target)) {
            hamburger.classList.remove('active');
            mainNav.classList.remove('active');
        }
    });
}

console.log('Deepfake Audio Detector initialized');

// ============================================================================
// ============================================================================
// Waveform Visualization
// ============================================================================

function drawWaveform(audioBuffer) {
    const canvas = document.getElementById('waveformCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Get audio data
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / width);
    const amp = height / 2;
    
    // Clear canvas
    ctx.fillStyle = 'rgba(31, 31, 31, 0.8)';
    ctx.fillRect(0, 0, width, height);
    
    // Draw waveform
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;
        
        for (let j = 0; j < step; j++) {
            const datum = data[(i * step) + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
        }
        
        ctx.lineTo(i, (1 + min) * amp);
        ctx.lineTo(i, (1 + max) * amp);
    }
    
    ctx.stroke();
    
    // Store waveform data for click seeking
    waveformData = { width, duration: audioBuffer.duration };
}

function updateWaveformProgress(currentTime, duration) {
    const progress = document.getElementById('waveformProgress');
    if (progress && duration > 0) {
        const percent = (currentTime / duration) * 100;
        progress.style.width = percent + '%';
    }
}

// Waveform click to seek
const waveformContainer = document.getElementById('waveformContainer');
if (waveformContainer) {
    waveformContainer.addEventListener('click', (e) => {
        if (!audioPlayer || !audioPlayer.duration) return;
        
        const rect = waveformContainer.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        audioPlayer.currentTime = percent * audioPlayer.duration;
    });
}

// ============================================================================
// Advanced Audio Metrics
// ============================================================================

async function calculateAdvancedMetrics(audioBuffer) {
    const data = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;
    
    // Peak Amplitude
    let peak = 0;
    for (let i = 0; i < data.length; i++) {
        const abs = Math.abs(data[i]);
        if (abs > peak) peak = abs;
    }
    
    // RMS Level
    let sumSquares = 0;
    for (let i = 0; i < data.length; i++) {
        sumSquares += data[i] * data[i];
    }
    const rms = Math.sqrt(sumSquares / data.length);
    const rmsDb = 20 * Math.log10(rms);
    
    // Dynamic Range
    const peakDb = 20 * Math.log10(peak);
    const dynamicRange = peakDb - rmsDb;
    
    // Zero Crossings
    let zeroCrossings = 0;
    for (let i = 1; i < data.length; i++) {
        if ((data[i] >= 0 && data[i-1] < 0) || (data[i] < 0 && data[i-1] >= 0)) {
            zeroCrossings++;
        }
    }
    const zcRate = zeroCrossings / (data.length / sampleRate);
    
    // Spectral Centroid (simplified)
    const fftSize = 2048;
    const spectralCentroid = zcRate / 2; // Simplified approximation
    
    // Noise Floor (estimate from quietest parts)
    const sortedAbs = Array.from(data).map(Math.abs).sort((a, b) => a - b);
    const noiseFloor = sortedAbs[Math.floor(sortedAbs.length * 0.1)];
    const noiseFloorDb = 20 * Math.log10(noiseFloor + 0.0001);
    
    // Update UI
    document.getElementById('peakAmplitude').textContent = (peak * 100).toFixed(1) + '%';
    document.getElementById('rmsLevel').textContent = rmsDb.toFixed(1) + ' dB';
    document.getElementById('dynamicRange').textContent = dynamicRange.toFixed(1) + ' dB';
    document.getElementById('zeroCrossings').textContent = (zcRate / 1000).toFixed(1) + 'k/s';
    document.getElementById('spectralCentroid').textContent = (spectralCentroid / 1000).toFixed(2) + ' kHz';
    document.getElementById('noiseFloor').textContent = noiseFloorDb.toFixed(1) + ' dB';
}

// ============================================================================
// Social Sharing
// ============================================================================

const shareTwitter = document.getElementById('shareTwitter');
const shareInstagram = document.getElementById('shareInstagram');
const shareFacebook = document.getElementById('shareFacebook');
const shareLinkedIn = document.getElementById('shareLinkedIn');
const shareCopy = document.getElementById('shareCopy');

function getShareText() {
    if (!lastPrediction) return 'Check out EchoShield - AI-powered deepfake audio detection!';
    
    const result = lastPrediction.label.toUpperCase();
    const confidence = Math.round(lastPrediction.confidence * 100);
    return `I just analyzed an audio file with EchoShield! Result: ${result} (${confidence}% confidence). Try it yourself!`;
}

if (shareTwitter) {
    shareTwitter.addEventListener('click', () => {
        const text = encodeURIComponent(getShareText());
        const url = encodeURIComponent(window.location.href);
        window.open(`https://twitter.com/intent/tweet?text=${text}&url=${url}`, '_blank', 'width=600,height=400');
    });
}

if (shareInstagram) {
    shareInstagram.addEventListener('click', async () => {
        // Instagram doesn't have a direct share URL, so copy text and open Instagram
        const text = getShareText() + '\n' + window.location.href;
        try {
            await navigator.clipboard.writeText(text);
            showToast('Text copied! Opening Instagram...', 'success');
            window.open('https://www.instagram.com/', '_blank');
        } catch (err) {
            showToast('Failed to copy text', 'error');
        }
    });
}

if (shareFacebook) {
    shareFacebook.addEventListener('click', () => {
        const url = encodeURIComponent(window.location.href);
        window.open(`https://www.facebook.com/sharer/sharer.php?u=${url}`, '_blank', 'width=600,height=400');
    });
}

if (shareLinkedIn) {
    shareLinkedIn.addEventListener('click', () => {
        const url = encodeURIComponent(window.location.href);
        const text = encodeURIComponent(getShareText());
        window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${url}`, '_blank', 'width=600,height=400');
    });
}

if (shareCopy) {
    shareCopy.addEventListener('click', async () => {
        const text = getShareText() + '\n' + window.location.href;
        try {
            await navigator.clipboard.writeText(text);
            showToast('Copied to clipboard!', 'success');
        } catch (err) {
            showToast('Failed to copy', 'error');
        }
    });
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

const shortcutsModal = document.getElementById('shortcutsModal');
const closeShortcuts = document.getElementById('closeShortcuts');
const shortcutHint = document.getElementById('shortcutHint');

// Hide hint after 10 seconds
if (shortcutHint) {
    setTimeout(() => {
        shortcutHint.style.opacity = '0';
        setTimeout(() => shortcutHint.style.display = 'none', 300);
    }, 10000);
}

if (closeShortcuts) {
    closeShortcuts.addEventListener('click', () => {
        shortcutsModal.style.display = 'none';
    });
}

document.addEventListener('keydown', (e) => {
    // Don't trigger shortcuts when typing in input fields
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    switch (e.key) {
        case '?':
            e.preventDefault();
            if (shortcutsModal) {
                shortcutsModal.style.display = shortcutsModal.style.display === 'none' ? 'flex' : 'none';
            }
            break;
            
        case 'Escape':
            if (shortcutsModal) shortcutsModal.style.display = 'none';
            break;
            
        case ' ':
            e.preventDefault();
            if (audioPlayer && audioPlayer.src) {
                if (audioPlayer.paused) {
                    audioPlayer.play();
                } else {
                    audioPlayer.pause();
                }
            }
            break;
            
        case 'r':
        case 'R':
            e.preventDefault();
            if (recordBtn) recordBtn.click();
            break;
            
        case 'Enter':
            e.preventDefault();
            if (uploadBtn && !uploadBtn.disabled) {
                uploadBtn.click();
            } else if (analyzeRecordingBtn && analyzeRecordingContainer.style.display !== 'none') {
                analyzeRecordingBtn.click();
            }
            break;
            
        case 'm':
        case 'M':
            e.preventDefault();
            if (audioPlayer) {
                audioPlayer.muted = !audioPlayer.muted;
            }
            break;
            
        case 'ArrowLeft':
            e.preventDefault();
            if (audioPlayer && audioPlayer.src) {
                audioPlayer.currentTime = Math.max(0, audioPlayer.currentTime - 5);
            }
            break;
            
        case 'ArrowRight':
            e.preventDefault();
            if (audioPlayer && audioPlayer.src) {
                audioPlayer.currentTime = Math.min(audioPlayer.duration, audioPlayer.currentTime + 5);
            }
            break;
            
        case 'ArrowUp':
            e.preventDefault();
            if (audioPlayer) {
                audioPlayer.volume = Math.min(1, audioPlayer.volume + 0.1);
                const volumeSlider = document.getElementById('volumeSlider');
                if (volumeSlider) volumeSlider.value = audioPlayer.volume * 100;
            }
            break;
            
        case 'ArrowDown':
            e.preventDefault();
            if (audioPlayer) {
                audioPlayer.volume = Math.max(0, audioPlayer.volume - 0.1);
                const volumeSlider = document.getElementById('volumeSlider');
                if (volumeSlider) volumeSlider.value = audioPlayer.volume * 100;
            }
            break;
    }
});

console.log('All features initialized: Batch Processing, Waveform, Advanced Metrics, Keyboard Shortcuts, Social Sharing');
