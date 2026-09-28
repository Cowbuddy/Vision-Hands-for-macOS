# Advanced Hand Tracking System for macOS

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)

A streamlined computer vision system for real-time hand tracking, gesture recognition, and macOS system control using MediaPipe and OpenCV. Features **ultra-fast Apple Vision Pro style controls** for seamless mouse replacement with 60 FPS performance on Apple Silicon.

## 🚀 Quick Start

```bash
# Install dependencies
./setup.sh

# Run the optimized version (recommended)
python main.py

# Or run the ultra-fast version
python main_ultra_fast_optimized.py
```

**First time setup?** See the [Installation](#installation) section below.

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
- **Natural gestures**: 
  - Quick pinch for click
  - Hold pinch (0.8s+) for drag operations
  - 3 fingers for scroll up
  - 4 fingers for scroll down
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

## 📁 Project Structure

```
HandTracking/
├── src/
│   ├── __init__.py
│   ├── models.py                    # Core data structures and types
│   ├── hand_analyzer.py            # Hand detection and finger analysis
│   ├── enhanced_gesture_recognition.py  # Advanced gesture pattern matching
│   └── system_controller.py        # macOS system integration
├── main.py                         # Main application (optimized)
├── main_ultra_fast_optimized.py   # Ultra-fast version with no flickering
├── config.py                       # Configuration management
├── benchmark.py                    # Performance benchmarking tool
├── setup.sh                        # Automated setup script
├── requirements.txt                # Python dependencies
├── requirements-uv.txt            # UV-optimized dependencies
└── hand_landmarker.task           # MediaPipe model (auto-downloaded)
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

#### Option 1: Automated Setup (Recommended)
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script (installs UV, dependencies, and downloads model)
./setup.sh
```

#### Option 2: Manual Setup
```bash
# Install dependencies with pip
pip install -r requirements.txt

# Download MediaPipe model
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

#### Option 3: Using UV (Fastest)
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with UV
uv pip install -r requirements-uv.txt

# Download model (if not already done)
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
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
# Standard optimized version
python main.py

# Ultra-fast optimized version (best for Apple Silicon)
python main_ultra_fast_optimized.py

# Run performance benchmark
python benchmark.py --duration 30

# Generate example configuration
python config.py
```

### ⌨️ Keyboard Controls
- **ESC**: Quit application
- **SPACE**: Toggle cursor tracking on/off
- **1-9**: Adjust cursor sensitivity (1=slow, 9=fast)
- **R**: Reset system state

## 🎮 Dual-Hand Control System

### 👈 Left Hand (Mode Control)
| Gesture | Action | Description |
|---------|--------|-------------|
| 🤏 **Pinch** | Toggle Tracking | Turn cursor tracking ON/OFF |
| ✋ **Open hand** | Navigation | General hand tracking |

### 👉 Right Hand (Cursor Control - when tracking enabled)
| Gesture | Action | Description |
|---------|--------|-------------|
| 👆 **Index finger** | Move Cursor | Position controls cursor (any angle) |
| 🤏 **Quick pinch** | Click | Tap-like click action |
| 🤏 **Hold pinch** (0.8s+) | Drag | Click-and-hold/drag mode |
| 🖖 **3 fingers** | Scroll Up | Scroll content upward |
| 🖐️ **4 fingers** | Scroll Down | Scroll content downward |

## Performance

- **60 FPS**: Stable performance optimized for Apple Silicon
- **Zero flickering**: Memory-optimized processing
- **Low latency**: Native macOS APIs for instant response
- **Universal tracking**: Works at any hand angle or distance

## 🛠️ Configuration

The system supports extensive configuration through `config.py`:

```bash
# Create example configuration file
python config.py

# Edit config.json to customize:
# - Camera settings (resolution, FPS)
# - Hand detection thresholds
# - Cursor sensitivity and smoothing
# - Gesture thresholds
# - Performance settings
# - UI preferences
```

### Example Configuration
```json
{
  "camera": {
    "width": 640,
    "height": 480,
    "fps": 60
  },
  "cursor": {
    "default_sensitivity": 0.4,
    "movement_threshold": 1.0
  },
  "gesture": {
    "pinch_threshold": 50.0,
    "scroll_sensitivity": 3
  }
}
```

## 🔧 Development

### Adding New Gestures
1. Define gesture pattern in `src/enhanced_gesture_recognition.py`
2. Add gesture detection logic in `src/hand_analyzer.py`
3. Implement system actions in `src/system_controller.py`
4. Update gesture handling in main application

### Performance Benchmarking
```bash
# Run 30-second benchmark
python benchmark.py

# Custom duration
python benchmark.py --duration 60
```

The benchmark reports:
- Average/min/max FPS
- Frame processing time distribution
- Memory usage statistics
- Performance rating and recommendations

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check camera permissions
System Preferences > Security & Privacy > Camera

# Try different camera device ID in config
"camera": { "device_id": 1 }  # Try 0, 1, 2, etc.
```

#### Poor Performance
- Close other applications using the camera
- Reduce resolution in config: `"width": 320, "height": 240`
- Disable UI elements: `"show_hand_landmarks": false`
- Run benchmark to identify bottlenecks: `python benchmark.py`

#### Cursor Not Moving
1. Ensure right hand is clearly visible to camera
2. Check if tracking is enabled (look for "Cursor: ON" in window)
3. Try adjusting sensitivity with keys 1-9
4. Verify lighting is adequate
5. Check movement threshold in config

#### Flickering/Unstable Cursor
- Use `main_ultra_fast_optimized.py` (specifically designed to fix flickering)
- Increase smoothing: `"ema_alpha": 0.3` (lower = smoother, higher = more responsive)
- Adjust movement threshold: `"movement_threshold": 2.0`

#### Gestures Not Recognized
- Increase stability: `"gesture_stability_frames": 10`
- Adjust confidence: `"min_detection_confidence": 0.5`
- Ensure fingers are clearly separated
- Check lighting and background contrast

### Performance Tips
| Area | Recommendation |
|------|----------------|
| **Lighting** | Use bright, even lighting for better detection |
| **Background** | Avoid cluttered backgrounds; plain wall works best |
| **Distance** | Keep hands 1-2 feet (30-60cm) from camera |
| **Hand Position** | Keep hands fully visible in camera frame |
| **System Load** | Close unnecessary background applications |
| **Hardware** | Works best on Apple Silicon (M1/M2/M3/M4) |

## 📊 Performance Metrics

Tested on **MacBook Pro M4 Pro**:
- **60 FPS** sustained performance
- **~16ms** average frame processing time
- **< 200MB** memory usage
- **< 10ms** gesture detection latency
- **Zero flickering** with optimized version

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional gesture patterns
- Cross-platform support (Windows, Linux)
- GPU acceleration options
- Custom gesture recording/playback
- Integration with accessibility APIs

## 📝 License

This project is open source under the MIT License. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for excellent hand tracking models
- [OpenCV](https://opencv.org/) for computer vision tools
- Apple's Vision Pro for gesture control inspiration

## 📚 Additional Resources

- [MediaPipe Hand Landmarker Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [macOS Accessibility APIs](https://developer.apple.com/accessibility/)

---

**Made with ❤️ for the macOS community**

*Star ⭐ this repo if you find it useful!*

### Jevis gesture event boundary

`python main.py --events-only` runs hand detection but emits JSONL to stdout instead
of moving the pointer, clicking, dragging, or scrolling. Diagnostic messages go to
stderr. This mode still needs the existing model, camera, and Python dependencies;
it does not connect to Jevis or execute model-generated actions.

```json
{"version":1,"source":"visionhands","type":"pinch.start","hand":"Right","timestamp":1750000000.25}
```

`type` is `pinch.start`, `pinch.end`, or `tracking.lost`; `hand` is `Left` or
`Right`; `timestamp` is Unix time in seconds. Pinch events occur on transitions,
not every frame. Loss is emitted once per previously visible hand. A consumer
must cancel pending pinch intent on loss (there may be no `pinch.end`). The
existing left-hand tracking toggle still gates right-hand pinch recognition.
These events describe observed gestures, not permission to perform an action.

Normal mode also releases an active drag when a hand disappears, tracking is
disabled, the tracker is reset, or the capture loop exits. Release failures retain
state for a retry while the tracker continues running.

Camera-free regression checks (no OS input or third-party dependencies):

```bash
python3 -m unittest discover -s tests -v
```
