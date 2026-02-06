# 🔍 Crack Detection System with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete system for **crack detection and segmentation** on surfaces using Convolutional Neural Networks (CNN). This project was developed as part of a Bachelor's Thesis (TFG).

![Demo](https://img.shields.io/badge/Status-Ready_to_Deploy-brightgreen)

---

## ✨ Features

The system includes two specialized deep learning models:

| Model | Architecture | Purpose |
|-------|-------------|---------|
| **Faster R-CNN** | ResNet50 backbone | Crack localization with bounding boxes and confidence scores |
| **UNet++** | ResNet34 encoder | Pixel-precise segmentation with area and extent metrics |

### Key Capabilities
- 🎯 **Real-time detection** with adjustable confidence threshold
- 📊 **Quantitative metrics**: affected area (px) and estimated crack extent
- 🖼️ **Visual overlays** for easy interpretation of results
- 🚀 **GPU acceleration** when available (CUDA support)

---

## 📁 Project Structure

```
crack-detection/
├── stream2.py          # Main Streamlit app (unified, recommended)
├── model_loader.py     # Model loading utilities (PyTorch)
├── model_utils.py      # Post-processing (filtering, skeletonization)
├── app.py              # REST API with FastAPI
├── frontend.py         # Legacy Streamlit frontend
├── weights/            # Trained model weights (.pth)
│   ├── fasterrcnn_final_SGD.pth
│   └── unetpp_final.pth
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container deployment
└── DEPLOYMENT_GUIDE.md # Deployment instructions
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA for faster inference

### Installation

```bash
# Clone the repository
git clone https://github.com/marcoslopezs/crack-detection.git
cd crack-detection

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

**Option 1: Unified Streamlit App (Recommended)**
```bash
streamlit run stream2.py
```
This opens an interactive web interface where you can upload images and get instant predictions.

**Option 2: API + Frontend (Separate)**
```bash
# Terminal 1: Start the API server
uvicorn app:app --reload

# Terminal 2: Start the frontend
streamlit run frontend.py
```
API documentation available at `http://127.0.0.1:8000/docs`

---

## 🎮 Usage

1. **Upload** an image (JPG, JPEG, or PNG)
2. **Select mode**:
   - *Crack Detection*: Get bounding boxes around detected cracks
   - *Segmentation & Measurement*: Get pixel-precise masks with metrics
3. **Adjust** the confidence threshold as needed
4. **Analyze** and view results with visual overlays

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t crack-detection .

# Run the container
docker run -p 8501:8501 crack-detection
```

Access the app at `http://localhost:8501`

---

## 📦 Model Weights

The trained weights are stored using **Git LFS**. They will be downloaded automatically when cloning the repository. If not:

```bash
git lfs install
git lfs pull
```

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, torchvision, segmentation-models-pytorch
- **Web Interface**: Streamlit
- **API**: FastAPI, Uvicorn
- **Image Processing**: OpenCV, Pillow, scikit-image
- **Containerization**: Docker

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Developed as a Bachelor's Thesis project.

---

<p align="center">
  <i>Built with ❤️ using PyTorch and Streamlit</i>
</p>