let currentStream = null;
let currentMode = 'upload';
let compareImages = { img1: null, img2: null };

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Face database will be loaded from backend
let faceDatabase = [];

document.addEventListener('DOMContentLoaded', function() {
    setupUploadArea();
    setupCompareInputs();
    loadKnownFaces();
});

function switchMode(mode) {
    // Update active mode button
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Hide all modes
    document.querySelectorAll('.recognition-mode').forEach(mode => mode.classList.remove('active'));
    
    // Show selected mode
    document.getElementById(`${mode}-mode`).classList.add('active');
    
    currentMode = mode;
    
    // Stop camera if switching away from camera mode
    if (mode !== 'camera' && currentStream) {
        stopCamera();
    }
}

function setupUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
        showMessage('Please select a valid image file', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImg = document.getElementById('previewImg');
        previewImg.src = e.target.result;
        
        document.getElementById('imagePreview').style.display = 'block';
        document.getElementById('uploadArea').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('fileInput').value = '';
    hideResults();
}

async function processImage() {
    const img = document.getElementById('previewImg');
    if (!img.src) return;
    
    showMessage('Processing image...', 'info');
    
    try {
        const base64Image = await getBase64FromImage(img);
        
        const response = await fetch(`${API_BASE_URL}/recognize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.faces);
            showMessage(`Found ${data.count} face(s)!`, 'success');
        } else {
            showMessage(data.error || 'Recognition failed', 'error');
        }
    } catch (error) {
        showMessage('Error connecting to server', 'error');
        console.error('Recognition error:', error);
    }
}

async function startCamera() {
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        const video = document.getElementById('cameraVideo');
        video.srcObject = currentStream;
        
        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('captureBtn').style.display = 'inline-block';
        document.getElementById('stopCamera').style.display = 'inline-block';
        
        // Start real-time face detection simulation
        startRealTimeDetection();
        
    } catch (error) {
        showMessage('Camera access denied or not available', 'error');
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    
    document.getElementById('startCamera').style.display = 'inline-block';
    document.getElementById('captureBtn').style.display = 'none';
    document.getElementById('stopCamera').style.display = 'none';
    
    // Clear overlay
    document.getElementById('cameraOverlay').innerHTML = '';
}

function startRealTimeDetection() {
    const overlay = document.getElementById('cameraOverlay');
    
    // Simulate real-time face detection
    const detectionInterval = setInterval(() => {
        if (!currentStream) {
            clearInterval(detectionInterval);
            return;
        }
        
        // Clear previous detections
        overlay.innerHTML = '';
        
        // Simulate face detection with API call
        simulateRealTimeDetection(overlay);
    }, 1000);
}

async function simulateRealTimeDetection(overlay) {
    try {
        // Capture current frame for analysis
        const video = document.getElementById('cameraVideo');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        
        const base64Image = canvas.toDataURL('image/jpeg', 0.3); // Lower quality for real-time
        
        const response = await fetch(`${API_BASE_URL}/recognize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image })
        });
        
        const data = await response.json();
        
        if (data.success && data.faces) {
            data.faces.forEach(face => {
                createFaceBox(overlay, face);
            });
        }
    } catch (error) {
        // Fallback to mock detection if API fails
        createMockFaceBox(overlay);
    }
}

function createFaceBox(overlay, faceData) {
    const faceBox = document.createElement('div');
    faceBox.className = 'face-box';
    
    // Use actual face location or random position
    const x = faceData.location ? (faceData.location.left / 640 * 100) : (Math.random() * 60 + 10);
    const y = faceData.location ? (faceData.location.top / 480 * 100) : (Math.random() * 60 + 10);
    const width = faceData.location ? ((faceData.location.right - faceData.location.left) / 640 * 100) : (Math.random() * 15 + 15);
    const height = faceData.location ? ((faceData.location.bottom - faceData.location.top) / 480 * 100) : (width * 1.2);
    
    faceBox.style.left = `${x}%`;
    faceBox.style.top = `${y}%`;
    faceBox.style.width = `${width}%`;
    faceBox.style.height = `${height}%`;
    
    // Add label
    const label = document.createElement('div');
    label.className = 'face-label';
    
    const confidence = Math.floor(faceData.confidence * 100);
    label.textContent = `${faceData.name} (${confidence}%)`;
    
    if (faceData.name === 'Unknown') {
        label.style.background = '#ef4444';
    }
    
    faceBox.appendChild(label);
    overlay.appendChild(faceBox);
}

function createMockFaceBox(overlay) {
    // Fallback mock detection
    if (Math.random() > 0.5) {
        const faceBox = document.createElement('div');
        faceBox.className = 'face-box';
        
        const x = Math.random() * 60 + 10;
        const y = Math.random() * 60 + 10;
        const width = Math.random() * 15 + 15;
        const height = width * 1.2;
        
        faceBox.style.left = `${x}%`;
        faceBox.style.top = `${y}%`;
        faceBox.style.width = `${width}%`;
        faceBox.style.height = `${height}%`;
        
        const label = document.createElement('div');
        label.className = 'face-label';
        label.textContent = 'Detecting...';
        label.style.background = '#3b82f6';
        
        faceBox.appendChild(label);
        overlay.appendChild(faceBox);
    }
}

async function captureImage() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    showMessage('Analyzing captured image...', 'info');
    
    try {
        const base64Image = canvas.toDataURL('image/jpeg', 0.8);
        
        const response = await fetch(`${API_BASE_URL}/recognize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.faces);
            showMessage(`Found ${data.count} face(s)!`, 'success');
        } else {
            showMessage(data.error || 'Recognition failed', 'error');
        }
    } catch (error) {
        showMessage('Error connecting to server', 'error');
        console.error('Recognition error:', error);
    }
}

function setupCompareInputs() {
    document.getElementById('compareInput1').addEventListener('change', (e) => {
        if (e.target.files[0]) {
            loadCompareImage(1, e.target.files[0]);
        }
    });
    
    document.getElementById('compareInput2').addEventListener('change', (e) => {
        if (e.target.files[0]) {
            loadCompareImage(2, e.target.files[0]);
        }
    });
}

function selectCompareImage(imageNum) {
    document.getElementById(`compareInput${imageNum}`).click();
}

function loadCompareImage(imageNum, file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById(`compareImg${imageNum}`);
        img.src = e.target.result;
        img.style.display = 'block';
        
        compareImages[`img${imageNum}`] = e.target.result;
        
        // Show compare button if both images are loaded
        if (compareImages.img1 && compareImages.img2) {
            document.getElementById('compareBtn').style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
}

async function compareImages() {
    if (!compareImages.img1 || !compareImages.img2) {
        showMessage('Please upload both images', 'error');
        return;
    }
    
    showMessage('Comparing faces...', 'info');
    
    try {
        const response = await fetch(`${API_BASE_URL}/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                image1: compareImages.img1,
                image2: compareImages.img2
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const result = data.result;
            const matchResult = document.getElementById('matchResult');
            const percentageEl = matchResult.querySelector('.match-percentage');
            const statusEl = matchResult.querySelector('.match-status');
            
            const percentage = Math.floor(result.confidence * 100);
            percentageEl.textContent = `${percentage}%`;
            
            if (result.match) {
                statusEl.textContent = 'Match Found';
                statusEl.style.color = '#10b981';
            } else {
                statusEl.textContent = 'No Match';
                statusEl.style.color = '#ef4444';
            }
            
            matchResult.style.display = 'block';
            showMessage('Face comparison completed!', 'success');
        } else {
            showMessage(data.error || 'Comparison failed', 'error');
        }
    } catch (error) {
        showMessage('Error connecting to server', 'error');
        console.error('Comparison error:', error);
    }
}

function getBase64FromImage(img) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        
        ctx.drawImage(img, 0, 0);
        resolve(canvas.toDataURL('image/jpeg', 0.8));
    });
}

async function loadKnownFaces() {
    try {
        const response = await fetch(`${API_BASE_URL}/known_faces`);
        const data = await response.json();
        
        if (data.success) {
            faceDatabase = data.faces;
        }
    } catch (error) {
        console.error('Error loading known faces:', error);
    }
}

function displayResults(results) {
    const resultsPanel = document.getElementById('resultsPanel');
    const detectionResults = document.getElementById('detectionResults');
    
    detectionResults.innerHTML = '';
    
    results.forEach((result, index) => {
        const item = document.createElement('div');
        item.className = 'detection-item';
        
        const confidenceClass = result.confidence >= 0.8 ? 'high' : 
                               result.confidence >= 0.6 ? 'medium' : 'low';
        
        const isUnknown = result.name === 'Unknown';
        
        item.innerHTML = `
            <div class="detection-info">
                <h4>${result.name}</h4>
                <p>Location: (${result.location.left}, ${result.location.top})</p>
                ${isUnknown ? '<p style="color: #f59e0b;">Face not in database</p>' : ''}
            </div>
            <div class="confidence-badge ${confidenceClass}">
                ${Math.floor(result.confidence * 100)}%
            </div>
        `;
        
        detectionResults.appendChild(item);
    });
    
    resultsPanel.style.display = 'block';
}

function hideResults() {
    document.getElementById('resultsPanel').style.display = 'none';
}

function logout() {
    localStorage.removeItem('currentUser');
    window.location.href = 'index.html';
}

function showMessage(message, type) {
    const existingMessage = document.querySelector('.message');
    if (existingMessage) existingMessage.remove();
    
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}`;
    messageEl.textContent = message;
    
    messageEl.style.cssText = `
        position: fixed;
        top: 100px;
        left: 50%;
        transform: translateX(-50%);
        padding: 1rem 2rem;
        border-radius: 10px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideDown 0.3s ease;
        ${type === 'success' ? 'background: #10b981;' : ''}
        ${type === 'error' ? 'background: #ef4444;' : ''}
        ${type === 'info' ? 'background: #3b82f6;' : ''}
    `;
    
    document.body.appendChild(messageEl);
    
    setTimeout(() => {
        messageEl.style.animation = 'slideUp 0.3s ease';
        setTimeout(() => messageEl.remove(), 300);
    }, 3000);
}