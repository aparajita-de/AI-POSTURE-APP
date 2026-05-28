import * as ort from 'onnxruntime-web';

const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const statusText = document.getElementById('status');

// 1. Bulletproof Camera Setup
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false,
        });
        
        // Force browser compliance via JS instead of HTML
        video.srcObject = stream;
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                video.play(); // Explicitly trigger playback
                resolve(video);
            };
        });
    } catch (err) {
        console.error(err);
        statusText.innerText = "Error: Webcam blocked.";
        statusText.style.color = "red";
    }
}

// 2. Load Model (CDN Version to prevent Vite WASM crashes)
async function loadModel() {
    try {
        // Force the app to fetch the WASM math core from a secure CDN
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        return await ort.InferenceSession.create('/yolov8n-pose.onnx', {
            executionProviders: ['wasm'] 
        });
    } catch (err) {
        console.error(err);
        statusText.innerText = "CRITICAL ERROR: Failed to load ONNX model.";
        statusText.style.color = "red";
    }
}

// 3. Tensor Conversion (Formatting the image for the Neural Network)
function preprocess(video) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 640;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(video, 0, 0, 640, 640);
    const imageData = tempCtx.getImageData(0, 0, 640, 640).data;

    // Convert pixel data to Float32 and normalize (0 to 1)
    const float32Data = new Float32Array(1 * 3 * 640 * 640);
    for (let i = 0; i < 640 * 640; i++) {
        float32Data[i] = imageData[i * 4] / 255.0;         // Red
        float32Data[i + 640 * 640] = imageData[i * 4 + 1] / 255.0; // Green
        float32Data[i + 2 * 640 * 640] = imageData[i * 4 + 2] / 255.0; // Blue
    }
    return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
}

// 4. Execution Loop & Mathematical Heuristics
async function detectPose(session) {
    // Clear canvas every frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
        const tensor = preprocess(video);
        const feeds = {};
        feeds[session.inputNames[0]] = tensor;
        
        // Run the AI model
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]].data;

        // Search the 8,400 anchors for the highest confidence person
        let maxScore = 0;
        let bestCol = 0;
        for (let i = 0; i < 8400; i++) {
            const score = output[4 * 8400 + i]; // Row 4 holds the confidence score
            if (score > maxScore) {
                maxScore = score;
                bestCol = i;
            }
        }

        // Proceed only if the AI is confident it sees a person
        if (maxScore > 0.5) {
            
            // YOLOv8 Pose Matrix Math: row = 5 + (keypoint_index * 3)
            // Nose (kp 0) = Row 5, 6
            // Left Shoulder (kp 5) = Row 20, 21
            // Right Shoulder (kp 6) = Row 23, 24
            
            // Extract coordinates and mathematically mirror the X-axis
            const nose = {
                x: canvas.width - (output[5 * 8400 + bestCol] * canvas.width / 640),
                y: output[6 * 8400 + bestCol] * canvas.height / 640
            };
            const lShoulder = {
                x: canvas.width - (output[20 * 8400 + bestCol] * canvas.width / 640),
                y: output[21 * 8400 + bestCol] * canvas.height / 640
            };
            const rShoulder = {
                x: canvas.width - (output[23 * 8400 + bestCol] * canvas.width / 640),
                y: output[24 * 8400 + bestCol] * canvas.height / 640
            };

            // Calculate the Slouch Ratio
            const midShoulder = {
                x: (lShoulder.x + rShoulder.x) / 2,
                y: (lShoulder.y + rShoulder.y) / 2
            };
            const neckLength = Math.hypot(nose.x - midShoulder.x, nose.y - midShoulder.y);
            const shoulderWidth = Math.hypot(lShoulder.x - rShoulder.x, lShoulder.y - rShoulder.y);
            const postureRatio = neckLength / shoulderWidth;

            // Product Logic Gate (Threshold set to 0.40)
            if (postureRatio < 0.40) { 
                statusText.innerText = "WARNING: Slouching Detected!";
                statusText.style.color = "#ff4444"; 
                ctx.strokeStyle = '#ff4444'; 
            } else {
                statusText.innerText = "Posture Tracking Active (Perfect Alignment)";
                statusText.style.color = "#00ff00"; 
                ctx.strokeStyle = '#00ff00';
            }

            // Draw T-Shirt Skeleton
            ctx.lineWidth = 5;
            ctx.beginPath();
            ctx.moveTo(lShoulder.x, lShoulder.y); 
            ctx.lineTo(rShoulder.x, rShoulder.y); // Line between shoulders
            ctx.moveTo(midShoulder.x, midShoulder.y); 
            ctx.lineTo(nose.x, nose.y); // Line from chest to nose
            ctx.stroke();

            // Display Ratio Text
            ctx.fillStyle = ctx.strokeStyle;
            ctx.font = 'bold 30px Arial';
            ctx.fillText("Ratio: " + postureRatio.toFixed(2), midShoulder.x + 20, midShoulder.y);
        }

    } catch (e) {
        console.error(e);
    }

    // Loop indefinitely
    requestAnimationFrame(() => detectPose(session));
}

// 5. Boot Sequence
async function main() {
    statusText.innerText = "Requesting Camera...";
    await setupCamera();
    
    statusText.innerText = "Loading AI Core...";
    const session = await loadModel();

    if (session) {
        detectPose(session); 
    }
}

// Initialize on user click to bypass browser security
document.addEventListener('click', () => {
    if(statusText.innerText === "Click anywhere on the screen to start."){
        main();
    }
}, { once: true });