# Real-Time Edge AI Posture Corrector 🚀

An end-to-end Machine Learning application that performs real-time human pose estimation entirely within the client browser. 

By leveraging **YOLOv8** and **ONNX Runtime WebAssembly**, this application bypasses backend server inference, resulting in zero-latency tracking, zero server costs, and total user privacy (frames never leave the webcam).

## 🧠 System Architecture
1. **Model Pipeline:** Pre-trained PyTorch YOLOv8 Nano model exported to ONNX format.
2. **Inference Engine:** ONNX Runtime Web (`onnxruntime-web`) utilizing local CPU/WASM acceleration.
3. **Frontend:** Vanilla JavaScript and HTML5 Canvas API via Vite.

## ⚙️ Quick Start (Run Locally)

**Prerequisites:** Node.js installed on your machine.


