# Advanced Hand Tracking System for macOS

A comprehensive computer vision system for real-time hand tracking, gesture recognition, and macOS system control using MediaPipe and OpenCV.

## Features

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

### 🖱️ Cursor Control
- **Left hand primary**: Left hand controls cursor by default (prevents interference)
- **Pinch activation**: Thumb + index finger pinch to enable/disable cursor
- **Smooth movement**: Index finger controls cursor position with smoothing
- **Click gestures**: Double-pinch for left mouse click
- **Hand switching**: Runtime switching between left/right hand control (L/R keys)
- **Automatic fallback**: Uses available hand if preferred hand not detected
- **Visual feedback**: Real-time pinch distance visualization

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

### Visual Interface
- Green landmarks: Hand detection points
- Blue line: Pinch distance visualization
- Status text: Current gesture and system state
- FPS counter: Performance monitoring

## File Structure

```
HandTracking/
├── main.py                # Main application entry point
├── requirements.txt       # Python dependencies
├── hand_landmarker.task   # MediaPipe model file
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── src/                   # Modular components
    ├── __init__.py
    ├── models.py            # Data structures
    ├── hand_analyzer.py     # Hand detection and analysis
    ├── gesture_recognition.py # Gesture pattern matching
    ├── cursor_controller.py   # Mouse control logic
    └── system_controller.py   # macOS system integration
```

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
