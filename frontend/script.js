// DOM element references with null checks for error handling
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const loading = document.getElementById("loading");
const resultBox = document.getElementById("result");
const textResult = document.getElementById("textResult");
const confidenceBar = document.getElementById("confidenceBar");
const probList = document.getElementById("probabilityList");
const dropZone = document.getElementById("dropZone");

// Track current blob URL to prevent memory leaks
let currentBlobUrl = null;

// Validate that all required DOM elements exist
if (!imageInput || !preview || !loading || !resultBox || !textResult || !confidenceBar || !probList || !dropZone) {
    console.error('Required DOM elements not found');
    showUserMessage('Application initialization failed. Please refresh the page.');
}

/**
 * Opens file selection dialog
 */
function openFile() {
    if (imageInput) {
        imageInput.click();
    }
}

// Handle file input change event
if (imageInput) {
    imageInput.addEventListener("change", () => {
        if (imageInput.files && imageInput.files[0]) {
            showPreview(imageInput.files[0]);
        }
    });
}

// Setup drag and drop functionality with proper event handling
if (dropZone) {
    dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        dropZone.style.background = "#eaf4ff";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.background = "white";
    });

    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.style.background = "white";
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
            imageInput.files = e.dataTransfer.files;
            showPreview(e.dataTransfer.files[0]);
        }
    });
}

/**
 * Display image preview and clean up previous blob URLs to prevent memory leaks
 * @param {File} file - The image file to preview
 */
function showPreview(file) {
    if (!file || !preview || !resultBox) return;
    
    // Revoke previous blob URL to prevent memory leak
    if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
    }
    
    // Create new blob URL and store reference
    currentBlobUrl = URL.createObjectURL(file);
    preview.src = currentBlobUrl;
    preview.style.display = "block";
    resultBox.style.display = "none";
}

/**
 * Send image to backend for brain tumor prediction
 * Includes proper error handling and user feedback
 */
async function predict() {
    // Validate input
    if (!imageInput || !imageInput.files || !imageInput.files.length) {
        showUserMessage("Please upload an MRI image");
        return;
    }

    // Validate required DOM elements
    if (!loading || !resultBox || !textResult || !confidenceBar || !probList) {
        showUserMessage("Application error: Missing interface elements");
        return;
    }

    // Show loading state
    loading.style.display = "block";
    resultBox.style.display = "none";

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    try {
        // Connect to Flask backend (ensure backend is running on port 5000)
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData,
            // Add timeout to prevent hanging requests
            signal: AbortSignal.timeout(30000)
        });

        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }

        const data = await res.json();

        // Validate response data
        if (!data || typeof data.prediction !== 'string' || typeof data.confidence !== 'number') {
            throw new Error('Invalid response format');
        }

        // Hide loading and show results
        loading.style.display = "none";
        resultBox.style.display = "block";

        // Display prediction results
        textResult.innerText = `Prediction: ${data.prediction}\nConfidence: ${data.confidence}%`;
        confidenceBar.style.width = Math.min(100, Math.max(0, data.confidence)) + "%";

        // Build probability list efficiently (avoid innerHTML concatenation in loop)
        if (data.probabilities && typeof data.probabilities === 'object') {
            const probEntries = Object.entries(data.probabilities);
            // Limit iterations to prevent DoS
            const maxEntries = Math.min(probEntries.length, 10);
            
            let probHtml = "";
            for (let i = 0; i < maxEntries; i++) {
                const [k, v] = probEntries[i];
                const safeKey = escapeHtml(k);
                const safeValue = Math.min(100, Math.max(0, parseFloat(v) || 0));
                
                probHtml += `
                    <div class="prob-row">
                        ${safeKey} (${safeValue}%)
                        <div class="prob-bar">
                            <div class="prob-fill" style="width:${safeValue}%"></div>
                        </div>
                    </div>
                `;
            }
            probList.innerHTML = probHtml;
        }

        // Add low confidence warning
        if (data.confidence < 70) {
            textResult.innerText += "\n⚠️ Low confidence – manual review advised.";
        }

    } catch (err) {
        loading.style.display = "none";
        console.error('Prediction error:', err);
        
        if (err.name === 'TimeoutError') {
            showUserMessage("Request timeout. Please try again.");
        } else if (err.message.includes('HTTP error')) {
            showUserMessage("Server error. Please try again later.");
        } else {
            showUserMessage("Network error. Please check your connection.");
        }
    }
}

/**
 * Display user-friendly messages instead of alert boxes
 * @param {string} message - Message to display
 */
function showUserMessage(message) {
    // Create or update message element instead of using alert
    let messageEl = document.getElementById('userMessage');
    if (!messageEl) {
        messageEl = document.createElement('div');
        messageEl.id = 'userMessage';
        messageEl.style.cssText = 'position:fixed;top:20px;left:50%;transform:translateX(-50%);background:#f44336;color:white;padding:10px 20px;border-radius:4px;z-index:1000;';
        document.body.appendChild(messageEl);
    }
    messageEl.textContent = message;
    messageEl.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (messageEl) messageEl.style.display = 'none';
    }, 5000);
}

/**
 * Escape HTML to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Clean up blob URLs when page unloads to prevent memory leaks
window.addEventListener('beforeunload', () => {
    if (currentBlobUrl) {
        URL.revokeObjectURL(currentBlobUrl);
    }
});
