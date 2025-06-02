# Advanced Hand Tracking System for macOS

A comprehensive computer vision system for real-time hand tracking, gesture recognition, and macOS system control using MediaPipe and OpenCV. Now featuring **ultra-fast Apple Vision Pro style controls** for seamless mouse replacement.

## 🚀 Quick Start

### OPTIMIZED: No Flickering Version (RECOMMENDED ⭐)
```bash
python main_ultra_fast_optimized.py
```
**Best for:** Mouse replacement, zero flickering, optimized memory management, stable 60 FPS

### Ultra-Fast Apple Vision Pro Style
```bash
python main_ultra_fast.py
```
**Best for:** Mouse replacement, maximum performance, intuitive Apple Vision Pro style gestures

### Threaded Maximum Performance
```bash
python main_threaded.py
```
**Best for:** Ultimate performance with multi-threaded processing, frame queuing, parallel hand detection

### Blazing Fast with Vim-like Controls
```bash
python main_blazing.py
```
**Best for:** Power users, vim-like modal controls, advanced customization

### Standard Hand Tracking
```bash
python main.py
```
**Best for:** Basic hand tracking, learning the system, development

## Features

### ✨ OPTIMIZED Version - No Flickering!
- **Memory-optimized processing**: Fixed UI flickering issues
- **Object pooling**: Reusable objects for minimal memory allocation
- **Garbage collection management**: Periodic cleanup for stable performance
- **Buffer reuse**: Pre-allocated arrays for frame processing
- **Stable 60 FPS**: Consistent performance without frame drops
- **Native macOS APIs**: Quartz/CoreGraphics for zero-latency cursor control

### 🍎 Ultra-Fast Apple Vision Pro Controls
- **Right-hand dominant**: Natural right-hand cursor control
- **Index finger tracking**: Point to move cursor with ultra-smooth EMA smoothing
- **Pinch gestures**: Thumb+index pinch for click and drag operations
- **Peace sign**: Right-click with intuitive two-finger gesture
- **Sensitivity control**: Adjustable sensitivity (1-9) for personalized control
- **Native macOS APIs**: Quartz/CoreGraphics for minimum latency cursor movement
- **640x480 @ 60fps**: Optimized camera resolution for maximum performance
- **Dynamic frame skipping**: Maintains 60fps even under load

### 🎮 Vim-like Modal Controls (Blazing Version)
- **Dual-hand control**: Left hand commands, right hand cursor
- **Modal interface**: Normal, cursor, and precision modes
- **Zero interference**: Vim-style separation of concerns
- **Mission Control**: Advanced system integration

### 🖐️ Hand Detection & Analysis
- **Multi-hand tracking**: Detects up to 2 hands simultaneously
- **Left/right classification**: Correctly identifies hand type (fixes MediaPipe mirroring)
- **High precision**: 21 landmark points per hand with confidence scoring
- **Real-time processing**: Optimized for M4 Pro performance cores

### 👆 Finger State Detection
- **Multiple algorithms**: Angle analysis, distance analysis, Y-coordinate analysis
- **All 5 fingers**: Thumb, index, middle, ring, pinky state detection
- **Robust detection**: Handles various hand orientations and lighting conditions

### 🤲 Gesture Recognition
- **Basic gestures**: Fist, open hand, pointing, peace sign
- **Advanced gestures**: Thumbs up, gun gesture, counting (1-5)
- **Gesture transitions**: Detects fist→open hand for system triggers
- **Extensible system**: Easy to add new gesture patterns

### 🖱️ Complete Mouse Replacement
- **Smooth movement**: EMA-smoothed cursor tracking for natural feel
- **Click and hold**: Easy pinch-and-hold for drag operations
- **Right-click support**: Peace sign gesture for context menus
- **Adjustable sensitivity**: 9-level sensitivity control for different use cases
- **Conditional updates**: Only moves cursor on significant position changes
- **Auto-disable**: Left hand fist gesture to pause cursor control

### 🖥️ System Integration
- **Mission Control**: Fist→open hand gesture triggers macOS Mission Control
- **AppleScript integration**: Native macOS automation with fallback support
- **System actions**: Expandable for additional macOS integrations

## Architecture

### Modular Design
```
src/
├── models.py              # Data structures and types
├── hand_analyzer.py       # Hand detection and analysis
├── gesture_recognition.py # Gesture pattern matching
├── cursor_controller.py   # Mouse control logic
└── system_controller.py   # macOS system integration
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
cd /Users/kaushaldadi/Code/HandTracking

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

### Controls
- **ESC**: Quit application
- **Left hand pinch**: Enable/disable cursor tracking (primary)
- **Left hand double pinch**: Perform left click
- **Left hand index finger**: Control cursor position
- **L key**: Switch cursor control to left hand
- **R key**: Switch cursor control to right hand
- **Any hand: Fist → Open hand**: Trigger Mission Control

## 🎮 Controls Guide

### Ultra-Fast Apple Vision Pro Style (`main_ultra_fast.py`)

#### Right Hand (Primary Cursor Control)
- **👉 Point (index finger extended)**: Move cursor
- **🤏 Pinch (thumb + index)**: Click / Start drag
- **🤏 Hold pinch + move**: Drag and drop
- **✌️ Peace sign (index + middle)**: Right-click
- **🖐️ 3 fingers extended**: Scroll up
- **🖐️ 4 fingers extended**: Scroll down
- **✊ Fist**: Pause cursor movement

#### Left Hand (System Control)
- **✊ Fist**: Disable cursor
- **🖐️ Open hand**: Enable cursor

#### Keyboard Controls
#### Keyboard Controls
- **1-9**: Adjust sensitivity (1=slowest, 9=fastest)
- **Space**: Toggle cursor on/off
- **R**: Reset system state
- **ESC**: Quit application

### Threaded Maximum Performance (`main_threaded.py`)

#### Right Hand (Primary Cursor Control)
- **👉 Point**: Move cursor with threading optimization
- **🤏 Pinch**: Click / Start drag
- **✌️ Peace sign**: Right-click
- **✊ Fist**: Hold position

#### Left Hand (Commands)
- **👉 Point**: Enable cursor mode
- **✌️ Peace**: Precision mode
- **✊ Fist**: Normal mode

#### Keyboard Controls
- **1-2**: Adjust frame skipping (1=process all, 2=skip more)
- **R**: Reset system state
- **ESC**: Quit application

### Blazing Fast Vim-like (`main_blazing.py`)

#### Right Hand (Cursor)
- **👉 Point**: Move cursor
- **🤏 Pinch**: Click
- **🖐️ Open hand**: Free movement
- **✊ Fist**: Hold position

#### Left Hand (Commands)
- **👉 Point**: Enable cursor mode
- **✌️ Peace**: Precision mode
- **✊ Fist**: Normal mode
- **👍 Thumbs up**: Mission Control

#### Keyboard Controls
- **S**: Speed mode
- **B**: Balanced mode
- **Q**: Quality mode
- **V**: Toggle visualization
- **R**: Reset
- **ESC**: Quit

### Standard Hand Tracking (`main.py`)

#### General Controls
- **🤏 Pinch gestures**: Basic cursor control
- **✊➡️🖐️ Fist to open**: Mission Control trigger
- **Multiple gesture recognition**: Learning and development

## Features Comparison

| Feature                          | Ultra-Fast Apple Vision Pro | Threaded Maximum Performance | Blazing Fast Vim-like | Standard Hand Tracking |
|----------------------------------|-----------------------------|-----------------------------|------------------------|------------------------|
| **Best For**                    | Mouse replacement, maximum performance, intuitive Apple Vision Pro style gestures | Ultimate performance with multi-threaded processing | Power users, vim-like modal controls, advanced customization | Basic hand tracking, learning the system, development |
| **Right-hand cursor control**    | Yes                         | Yes                         | Yes                   | Limited                |
| **Dual-hand control**           | No                          | Yes                         | Yes                   | No                     |
| **Modal interface**             | No                          | Yes                         | Yes                   | No                     |
| **Mission Control integration**  | Yes                         | Yes                         | Yes                   | Yes                    |
| **Gesture recognition**         | Basic + Advanced            | Advanced                    | Basic + Advanced      | Basic                  |
| **Mouse replacement**            | Complete with scrolling     | Complete                    | Partial                | No                     |
| **Sensitivity control**         | Yes (1-9)                   | Yes (frame skipping)        | Yes (3 levels)        | No                     |
| **Frame rate**                  | 60fps                       | 60fps+ (threaded)           | 30fps                 | 25-30fps               |
| **Threading**                   | No                          | **Yes (Multi-threaded)**   | No                    | No                     |
| **Scrolling support**           | **Yes (3/4 fingers)**      | Planned                     | No                    | No                     |
| **Latency**                     | Minimum                     | Low                   | Moderate               |
| **Ease of use**                 | High                        | Moderate              | High                   |
| **Customization**               | Limited                     | High                   | Low                    |

## Architecture

The system is organized into modular components for flexibility and performance:

```
src/
├── models.py              # Data structures and types
├── hand_analyzer.py       # Hand detection and analysis
├── gesture_recognition.py # Gesture pattern matching
├── cursor_controller.py   # Mouse control logic
└── system_controller.py   # macOS system integration
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
cd /Users/kaushaldadi/Code/HandTracking

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

### Controls
- **ESC**: Quit application
- **Left hand pinch**: Enable/disable cursor tracking (primary)
- **Left hand double pinch**: Perform left click
- **Left hand index finger**: Control cursor position
- **L key**: Switch cursor control to left hand
- **R key**: Switch cursor control to right hand
- **Any hand: Fist → Open hand**: Trigger Mission Control

## Hand Separation Logic

### LEFT HAND (Primary Cursor Control)
- **Pinch gestures**: Enable/disable cursor tracking
- **Double-pinch**: Left mouse clicks
- **Index finger**: Controls cursor position with smoothing
- **Visual feedback**: Real-time pinch distance visualization

### RIGHT HAND (Free for Other Gestures)  
- **Mission Control**: Fist → open hand gesture
- **Gesture recognition**: All gesture patterns available
- **No interference**: Operates independently of cursor control
- **Automatic fallback**: Can control cursor if left hand unavailable

### Benefits of Hand Separation
- **No Interference**: Both hands work simultaneously without conflicts
- **Natural Workflow**: Left hand for precision cursor, right hand for gestures  
- **Runtime Flexibility**: L/R keys to switch cursor control between hands
- **Robust Operation**: System adapts when preferred hand unavailable

## Technical Details

### Hand Landmark Detection
Uses MediaPipe's hand landmarker model with 21 key points:
- Wrist (0)
- Thumb: 1-4
- Index: 5-8
- Middle: 9-12
- Ring: 13-16
- Pinky: 17-20

### Gesture Recognition Algorithms
1. **Angle Analysis**: Calculates finger bend angles
2. **Distance Analysis**: Measures fingertip distances
3. **Y-Coordinate Analysis**: Compares relative positions

### Performance Notes
- **GPU Acceleration**: Metal delegate tested but unstable on macOS
- **CPU Optimization**: TensorFlow Lite XNNPACK provides excellent performance
- **Frame Rate**: Typically 25-30 FPS on M4 Pro
- **Latency**: <50ms for gesture recognition

## Troubleshooting

### Common Issues
1. **Camera not found**: Check camera permissions in System Preferences
2. **Poor detection**: Ensure good lighting and clear hand visibility
3. **Cursor too sensitive**: Adjust smoothing parameters in cursor_controller.py
4. **Mission Control not working**: Check Accessibility permissions for Terminal/Python

### Performance Optimization
- Close other camera applications
- Ensure adequate lighting
- Position hands 1-3 feet from camera
- Use main_modular.py for best performance

## Development

### Adding New Gestures
1. Define gesture pattern in `gesture_recognition.py`
2. Add detection logic to `recognize_gesture()` method
3. Update gesture transition handling if needed

### Extending System Integration
1. Add new actions to `system_controller.py`
2. Define gesture triggers in main loop
3. Test with appropriate cooldown periods

### Modifying Cursor Behavior
1. Adjust sensitivity in `cursor_controller.py`
2. Modify smoothing algorithms
3. Update enable/disable logic

## Future Enhancements

- [ ] Complex gesture sequences
- [ ] Two-hand gesture combinations
- [ ] Custom gesture training interface
- [ ] Gesture recording and playback
- [ ] Performance profiling dashboard
- [ ] Additional system integrations
- [ ] Voice command integration
- [ ] Machine learning gesture customization

## License

This project is for educational and personal use.

## Credits

- MediaPipe by Google
- OpenCV community
- Apple Silicon optimization techniques

---

**Status**: Production Ready ✅  
**Last Updated**: May 30, 2025  
**Platform**: macOS (M4 Pro optimized)

---

## 🔥 OPTIMIZATION UPDATES (June 2025)

### ⭐ NEW: Optimized Version (RECOMMENDED)
```bash
python main_ultra_fast_optimized.py  # NO FLICKERING!
```

**✅ FIXES APPLIED:**
- **Memory-optimized processing**: Eliminated UI flickering after extended use
- **Object pooling**: Reusable cursor event objects
- **Garbage collection**: Periodic cleanup every 300 frames
- **Buffer reuse**: Pre-allocated arrays for frame processing
- **Stable 60 FPS**: Consistent performance without degradation

### 🧵 NEW: Simple Threaded Version
```bash
python main_simple_threaded.py  # Multi-threaded processing
```

**🚀 THREADING FEATURES:**
- Multi-threaded frame processing
- Frame and result queues
- Non-blocking operations
- Real-time queue monitoring

### 📦 UV Package Manager Support
```bash
./setup.sh  # Auto-setup with UV (10x faster installs)
```

**⚡ INSTALLATION OPTIMIZATIONS:**
- UV package manager integration
- Automated setup script
- Optimized dependency list
- Faster development workflow

### 🎮 Complete Mouse Replacement
**All versions now include:**
- 👉 Point (index finger) = Move cursor  
- 🤏 Pinch (thumb + index) = Click
- ✌️ Peace sign = Right-click
- 🖐️ 3 fingers = Scroll up
- 🖐️ 4 fingers = Scroll down
- ⌨️ 1-9 keys = Sensitivity control

### 💡 Why the Original Flickered
The flickering issue was caused by:
1. Memory allocation every frame for MediaPipe objects
2. Unlimited frame history causing memory growth
3. Complex UI rendering overhead
4. No garbage collection management
5. New cursor event creation instead of reusing objects

**SOLUTION**: The optimized version uses object pooling, limited memory buffers, and periodic garbage collection to maintain stable performance.

---
