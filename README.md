# 🧘‍♀️ Real-Time Edge AI Posture Tracker

A high-performance, completely local computer vision application that tracks human posture in real-time using a standard webcam. 

Unlike traditional cloud-based AI, this engine compiles a neural network directly into WebAssembly (WASM) to run locally in the browser. This guarantees **zero network latency** and **100% data privacy**—no video frames are ever sent to a server.

---

## 🏗️ System Architecture (Edge Inference Pipeline)

This application is built on a decentralized edge-computing architecture. The entire data pipeline executes strictly on the client side.

1. **Media Stream Intake:** The `navigator.mediaDevices` API captures raw video frames via the user's local webcam.
2. **Tensor Preprocessing:** Video frames are drawn to an HTML5 Canvas, extracted as raw RGB pixel data, normalized (0.0 to 1.0), and reshaped into a `Float32Array` tensor `[1, 3, 640, 640]`.
3. **WASM Execution:** The PyTorch YOLOv8 model (quantized to `.onnx` format) is executed via `onnxruntime-web` directly on the local CPU utilizing WebAssembly threads.
4. **Matrix Parsing:** The resulting 8,400-column bounding-box matrix is parsed locally to isolate the highest-confidence human detection and map 17 anatomical keypoints.
5. **Heuristic Engine:** The custom client-side algorithmic engine calculates the anatomical "Slouch Ratio" (see below).
6. **UI Rendering:** The Canvas 2D API handles the mirrored skeletal tracking and dynamic UI alerts at 30+ FPS.

---

## ✨ Key Features
* **Edge Inference:** No backend servers, no Python environments, no API keys. Pure local compute.
* **Custom Frontal-Heuristics:** Traditional posture apps require a side-profile view to calculate spinal angles. This app utilizes a custom algorithm designed specifically for front-facing webcams.
* **Zero-Setup Deployment:** Users simply open a URL, grant camera access, and the AI initializes.
* **Hardware Agnostic:** Runs on any device with a modern browser, optimized via WebAssembly.

---

## 🧠 The Mathematical Model (The Slouch Ratio)

When facing a camera directly, 2D joint angles compress, making standard trigonometric posture tracking ineffective. This engine solves that by dynamically tracking three specific keypoints: the **Nose**, **Left Shoulder**, and **Right Shoulder**.

1. It calculates the exact midpoint of the chest (between the shoulders).
2. It measures the `Neck Length` (distance from the nose to the chest midpoint).
3. It measures the `Shoulder Width`.
4. It calculates the dynamic ratio: `Ratio = Neck Length / Shoulder Width`

Because it relies on a relative ratio rather than absolute pixel distances, the tracking remains perfectly accurate regardless of how close or far the user sits from the camera. A ratio below `0.40` triggers the slouching logic gate.

---

## 🛠️ Technology Stack
* **Model:** [YOLOv8n-pose](https://github.com/ultralytics/ultralytics) (Quantized PyTorch Model)
* **Inference Engine:** ONNX Runtime Web (`onnxruntime-web`)
* **Frontend:** Vanilla JavaScript, HTML5 Canvas, CSS3
* **Build Tool:** Vite

---

## 🚀 Local Installation

### Prerequisites
* [Node.js](https://nodejs.org/) (v16 or higher)
* A modern web browser (Chrome, Edge, Safari) with a working webcam.

