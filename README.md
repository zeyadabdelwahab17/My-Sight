# My Sight - Raspberry Pi Assistant

A voice-controlled computer vision assistant with face recognition, object detection, currency identification, and color detection capabilities.

## 📋 Prerequisites

### Hardware
- Raspberry Pi 4/5 (Recommended: Pi 5 with 4GB+ RAM)
- Official Raspberry Pi Camera Module or USB webcam
- IR sensor
- Microphone (Built-in or USB)
- Speaker/headphones (3.5mm jack or HDMI audio)

### Software
- Raspberry Pi OS (64-bit) Bullseye or newer
- Python 3.9+ (Pre-installed on Raspberry Pi OS)

## 🛠️ Installation Guide

### 1. System Setup

```bash
# Update system packages
sudo apt update && sudo apt full-upgrade -y

# Install core dependencies
sudo apt install -y \
    python3-pip python3-venv \
    libatlas-base-dev libopenblas-dev \
    portaudio19-dev libjasper-dev \
    libjpeg-dev libpng-dev \
    libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 \
    libcamera-dev libopencv-dev
```
### 2. Enable Hardware Interfaces
```
sudo raspi-config
```
- Select Interface Options → Camera → Enable

- Select Interface Options → SPI → Enable (if using certain sensors)

- Select Performance Options → GPU Memory → Set to 128MB

- Reboot when prompted

### 3. Project Setup
```bash
# Clone repository (or copy your project files)
git clone https://your-repository-url.git ~/my_sight_project
cd ~/my_sight_project

# Create virtual environment
python3 -m venv ~/my_sight_env
source ~/my_sight_env/bin/activate

# Install Python packages
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 📂 File Structure
```bash
my_sight_project/
├── Models/
│   ├── vosk-model-small-en-us-0.15/  # Speech recognition
│   ├── face_recognition_model.pkl    # Face encodings
│   ├── currency_int8.tflite          # Currency model
│   └── yolov8n.pt                    # Object detection
├── my_sight.py                       # Main application
└── requirements.txt                  # Dependencies
```
## 🚀 Running the Application

```bash
# Activate virtual environment
source ~/my_sight_env/bin/activate

# Run the application
cd ~/my_sight_project
python my_sight.py
```
### Voice Commands
- "face" - Face recognition

- "currency" - Identify currency

- "color" - Detect dominant color

- "object" - Object detection

- "exit" - Quit application



