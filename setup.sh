#!/bin/bash
# Hand Tracking Setup Script - Enhanced with better checks

set -e  # Exit on error

echo "🚀 Hand Tracking System Setup"
echo "=============================="
echo ""

# Check Python version
echo "🐍 Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        echo "✅ Python $PYTHON_VERSION detected"
    else
        echo "❌ Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check operating system
echo ""
echo "💻 Checking operating system..."
OS="$(uname -s)"
case "${OS}" in
    Darwin*)    
        echo "✅ macOS detected"
        MACOS=true
        ;;
    Linux*)     
        echo "⚠️  Linux detected (experimental support)"
        MACOS=false
        ;;
    *)          
        echo "❌ Unsupported OS: ${OS}"
        echo "This system is optimized for macOS"
        exit 1
        ;;
esac

# Check for camera
echo ""
echo "📷 Checking for camera..."
if [ "$MACOS" = true ]; then
    if system_profiler SPCameraDataType 2>/dev/null | grep -q "Camera"; then
        echo "✅ Camera detected"
    else
        echo "⚠️  No camera detected (may still work with external webcam)"
    fi
fi

# Offer installation method choice
echo ""
echo "📦 Choose installation method:"
echo "  1) UV (fastest, recommended)"
echo "  2) pip (standard)"
echo "  3) Skip dependency installation"
read -p "Enter choice [1-3]: " INSTALL_CHOICE

case $INSTALL_CHOICE in
    1)
        # Check if uv is installed
        if ! command -v uv &> /dev/null; then
            echo ""
            echo "📥 UV not found. Installing UV..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            
            # Try to source the shell config
            if [ -f "$HOME/.cargo/env" ]; then
                source "$HOME/.cargo/env"
            fi
            
            # Verify installation
            if command -v uv &> /dev/null; then
                echo "✅ UV installed successfully"
            else
                echo "⚠️  UV installation may require shell restart"
                echo "Please run: source ~/.bashrc (or ~/.zshrc)"
                exit 1
            fi
        else
            echo "✅ UV already installed"
        fi
        
        echo ""
        echo "📦 Installing dependencies with UV (super fast)..."
        uv pip install -r requirements-uv.txt
        
        # Install benchmark dependencies
        echo "📦 Installing benchmark dependencies..."
        uv pip install psutil
        ;;
    2)
        echo ""
        echo "📦 Installing dependencies with pip..."
        python3 -m pip install --upgrade pip
        python3 -m pip install -r requirements.txt
        
        # Install benchmark dependencies
        echo "📦 Installing benchmark dependencies..."
        python3 -m pip install psutil
        ;;
    3)
        echo ""
        echo "⏭️  Skipping dependency installation"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Download model
echo ""
echo "🔍 Checking for MediaPipe hand model..."
if [ ! -f "hand_landmarker.task" ]; then
    echo "📥 Downloading MediaPipe hand model (~10MB)..."
    if command -v curl &> /dev/null; then
        curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    elif command -v wget &> /dev/null; then
        wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    else
        echo "❌ Neither curl nor wget found. Please install one and rerun."
        exit 1
    fi
    
    if [ -f "hand_landmarker.task" ]; then
        echo "✅ Model downloaded successfully"
    else
        echo "❌ Failed to download model"
        exit 1
    fi
else
    MODEL_SIZE=$(ls -lh hand_landmarker.task | awk '{print $5}')
    echo "✅ Model file already exists ($MODEL_SIZE)"
fi

# Create config file if not exists
echo ""
echo "⚙️  Setting up configuration..."
if [ ! -f "config.json" ]; then
    echo "📝 Creating default configuration..."
    python3 config.py
    echo "✅ Configuration created (config.json)"
else
    echo "✅ Configuration already exists"
fi

# Check camera permissions (macOS only)
if [ "$MACOS" = true ]; then
    echo ""
    echo "🔐 Camera Permissions:"
    echo "   If prompted, please grant camera access to Terminal/Python"
    echo "   You can also check: System Preferences > Security & Privacy > Camera"
fi

# Setup complete
echo ""
echo "="*60
echo "🎉 Setup Complete!"
echo "="*60
echo ""
echo "🚀 Quick Start:"
echo "   python3 main.py                      # Standard optimized version"
echo "   python3 main_ultra_fast_optimized.py # Ultra-fast (no flickering)"
echo ""
echo "🔧 Utilities:"
echo "   python3 benchmark.py                 # Performance benchmarking"
echo "   python3 config.py                    # Generate configuration"
echo "   python3 examples/custom_gestures.py  # View custom gesture examples"
echo ""
echo "🎮 Controls:"
echo "   LEFT HAND:"
echo "     • Pinch = Toggle cursor tracking ON/OFF"
echo "   RIGHT HAND (when tracking enabled):"
echo "     • Index finger = Move cursor"
echo "     • Quick pinch = Click"
echo "     • Hold pinch (0.8s+) = Drag"
echo "     • 3 fingers = Scroll up"
echo "     • 4 fingers = Scroll down"
echo "   KEYBOARD:"
echo "     • ESC = Exit | SPACE = Toggle | 1-9 = Sensitivity"
echo ""
echo "📚 Documentation:"
echo "   See README.md for full documentation and troubleshooting"
echo ""
echo "="*60
