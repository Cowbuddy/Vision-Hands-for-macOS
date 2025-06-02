#!/bin/bash
# Hand Tracking Setup Script - Using UV for faster installation

echo "🚀 Hand Tracking System Setup"
echo "=============================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Installing UV first..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    echo "✅ UV installed successfully"
fi

echo "📦 Installing dependencies with UV (super fast)..."
uv pip install -r requirements-uv.txt

echo "🔍 Checking for hand_landmarker.task model..."
if [ ! -f "hand_landmarker.task" ]; then
    echo "📥 Downloading MediaPipe hand model..."
    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    echo "✅ Model downloaded successfully"
else
    echo "✅ Model file already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To run the optimized version (no flickering):"
echo "   python main_ultra_fast_optimized.py"
echo ""
echo "⚡ Other versions available:"
echo "   python main.py                 # Basic version"
echo "   python main_blazing.py         # Blazing fast version"
echo "   python main_ultra_fast.py      # Ultra fast version"
echo ""
echo "🎮 Controls:"
echo "   Point (index finger) = Move cursor"
echo "   Pinch (thumb + index) = Click"
echo "   3 fingers = Scroll up"
echo "   4 fingers = Scroll down"
echo "   ESC = Exit | SPACE = Toggle cursor | 1-9 = Sensitivity"
