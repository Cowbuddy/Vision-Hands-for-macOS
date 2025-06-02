# Advanced Hand Tracking System for macOS

A streamlined computer vision system for real-time hand tracking, gesture recognition, and macOS system control using MediaPipe and OpenCV. Features **ultra-fast Apple Vision Pro style controls** for seamless mouse replacement.

## 🚀 Quick Start

```bash
python main.py
```

**Optimized dual-hand control system featuring:**
- **Memory-optimized processing**: Zero flickering, stable 60 FPS
- **Apple Vision Pro style controls**: Natural hand-based cursor control
- **Dual-hand operation**: Left hand for mode switching, right hand for cursor control
- **Native macOS APIs**: Quartz/CoreGraphics for zero-latency interaction

## Features

### ✨ Optimized Dual-Hand Control System
- **Memory-optimized processing**: Zero flickering, stable 60 FPS performance
- **Object pooling**: Reusable objects for minimal memory allocation
- **Native macOS APIs**: Quartz/CoreGraphics for zero-latency cursor control
- **Smart frame processing**: Dynamic optimization for consistent performance

### 🍎 Apple Vision Pro Style Controls
- **Dual-hand operation**: 
  - **Left hand**: Pinch to toggle cursor tracking on/off
  - **Right hand**: Index finger position controls cursor movement
- **Natural gestures**: Pinch for click, hold pinch for drag operations
- **Universal tracking**: Works at any hand angle or orientation
- **Intelligent sensitivity**: Automatic threshold-based movement detection

### 🖐️ Advanced Hand Detection
- **Multi-hand tracking**: Detects up to 2 hands simultaneously
- **High precision**: 21 landmark points per hand with confidence scoring
- **Robust detection**: Handles various hand orientations and lighting conditions
- **Real-time processing**: Optimized for Apple Silicon performance

### 👆 Precise Gesture Recognition
- **Finger state detection**: All 5 fingers with multiple detection algorithms
- **Pinch detection**: Accurate thumb-index distance calculation
- **Gesture classification**: Open hand, fist, pointing, and custom gestures
- **State management**: Smooth transitions between different hand states

### 🖱️ Complete Mouse Replacement
- **Smooth movement**: EMA-smoothed cursor tracking for natural feel
- **Click operations**: Quick pinch for click, hold pinch for drag
- **Smart detection**: Movement threshold to prevent cursor jitter
- **Screen mapping**: Accurate camera-to-screen coordinate transformation

## Architecture

### Streamlined Modular Design
```
src/
├── models.py                    # Core data structures and types
├── hand_analyzer.py            # Hand detection and finger analysis
├── enhanced_gesture_recognition.py  # Advanced gesture pattern matching
└── system_controller.py        # macOS system integration

main.py                         # Optimized main application
```

### Performance Optimizations
- **CPU-optimized**: Uses TensorFlow Lite XNNPACK delegate
- **M4 Pro tuned**: Leverages Apple Silicon performance cores
- **Efficient processing**: Minimal latency for real-time interaction

## Installation

### Requirements
- Python 3.8+
- macOS (tested on macOS with M4 Pro)
- Webcam/camera

### Setup
```bash
# Clone or navigate to project directory
cd HandTracking

# Install dependencies
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

### Dependencies
- `mediapipe>=0.10.21` - Hand landmark detection
- `opencv-python` - Computer vision and camera handling
- `numpy` - Numerical computations
- `pyautogui` - System cursor control
- `Pillow` - Image processing

## Usage

### Basic Usage
```bash
python main.py
```

### Keyboard Controls
- **ESC**: Quit application
- **SPACE**: Toggle cursor tracking on/off
- **1-9**: Adjust cursor sensitivity

## 🎮 Dual-Hand Control System

### Left Hand (Mode Control)
- **🤏 Pinch**: Toggle cursor tracking ON/OFF
- **✋ Open hand**: General navigation and interaction

### Right Hand (Cursor Control - when tracking enabled)
- **👉 Index finger position**: Move cursor (works at any angle)
- **🤏 Quick pinch**: Perform click
- **🤏 Hold pinch (0.8s+)**: Click-and-hold/drag mode

## Performance

- **60 FPS**: Stable performance optimized for Apple Silicon
- **Zero flickering**: Memory-optimized processing
- **Low latency**: Native macOS APIs for instant response
- **Universal tracking**: Works at any hand angle or distance

## Development

### Adding New Gestures
1. Define gesture pattern in `enhanced_gesture_recognition.py`
2. Add gesture detection logic in `hand_analyzer.py`
3. Implement system actions in `system_controller.py`

### Testing
```bash
# Test pinch detection accuracy
python test_pinch_click.py
```

## Troubleshooting

### Common Issues
- **No camera detected**: Check camera permissions in System Preferences
- **Poor performance**: Close other camera applications
- **Cursor not moving**: Ensure right hand is clearly visible
- **Sensitivity issues**: Use keyboard keys 1-9 to adjust

### Performance Tips
- **Lighting**: Use good lighting for better detection
- **Background**: Avoid cluttered backgrounds
- **Distance**: Keep hands 1-2 feet from camera
- **Hand position**: Keep hands in camera frame

## License

This project is open source. Feel free to contribute improvements!
