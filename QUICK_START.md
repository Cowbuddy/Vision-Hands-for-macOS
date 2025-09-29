# Quick Start Guide

## 🚀 Installation (5 minutes)

```bash
# Clone or download the repository
cd HandTracking

# Run automated setup
chmod +x setup.sh
./setup.sh

# Choose installation method when prompted:
# 1) UV (fastest) - recommended
# 2) pip (standard)
# 3) Skip (if already installed)
```

## 🎮 Running the Application

```bash
# Optimized version (recommended)
python3 main.py

# Ultra-fast version (no flickering, best for Apple Silicon)
python3 main_ultra_fast_optimized.py
```

## 👋 Basic Hand Controls

### Left Hand (Mode Control)
- **🤏 Pinch** → Toggle cursor tracking ON/OFF

### Right Hand (Cursor Control)
- **👆 Point** → Move cursor
- **🤏 Quick pinch** → Click
- **🤏 Hold pinch** (0.8s) → Drag
- **🖖 3 fingers** → Scroll up
- **🖐️ 4 fingers** → Scroll down

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `ESC` | Exit application |
| `SPACE` | Toggle cursor tracking |
| `1-9` | Adjust sensitivity (1=slow, 9=fast) |
| `R` | Reset system state |

## 🎯 Tips for Best Performance

1. **Lighting**: Use bright, even lighting
2. **Distance**: Keep hands 1-2 feet from camera
3. **Background**: Use plain background for better detection
4. **Position**: Keep hands fully visible in frame
5. **Camera**: Close other apps using the camera

## 🐛 Quick Troubleshooting

### Camera not working?
```bash
# Check permissions:
System Preferences > Security & Privacy > Camera
```

### Cursor not moving?
1. Press `SPACE` to ensure tracking is ON
2. Try adjusting sensitivity with keys `1-9`
3. Ensure right hand is visible

### Performance issues?
```bash
# Run benchmark to diagnose
python3 benchmark.py

# Try reducing camera resolution in config.json
```

## 📊 Performance Check

```bash
# Run 30-second performance test
python3 benchmark.py --duration 30
```

Expected results on Apple Silicon:
- 55-60 FPS average
- <20ms frame processing time

## 🔧 Customization

```bash
# Generate configuration file
python3 config.py

# Edit config.json to adjust:
# - Camera settings
# - Cursor sensitivity
# - Gesture thresholds
# - Performance options
```

## 📚 Learn More

- Full documentation: [README.md](README.md)
- Custom gestures: [examples/custom_gestures.py](examples/custom_gestures.py)
- Configuration: [config.py](config.py)

## 🆘 Still Need Help?

Check the [Troubleshooting](README.md#troubleshooting) section in README.md

---

**Ready to start?** Run `python3 main.py` and show your hands to the camera! 👋