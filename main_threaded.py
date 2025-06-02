#!/usr/bin/env python3
"""
THREADED Hand Tracking System - Maximum Performance Edition

Multi-threaded hand tracking system optimized for ultimate performance using 
parallel processing, frame queuing, and advanced optimization techniques.
"""
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.threaded_hand_tracker import ThreadedHandTracker

def main():
    """Main entry point for maximum performance threaded hand tracking"""
    print("🚀 THREADED Hand Tracking System - Maximum Performance")
    print("=" * 60)
    print("🧵 Multi-threaded processing enabled")
    print("⚡ Frame queuing and parallel processing")
    print("🎮 Vim-like controls with zero interference")
    print("🖐️ Right-hand dominant design")
    
    # Check if model file exists
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the hand_landmarker.task file is in the current directory")
        return
    
    try:
        # Create and run the threaded tracker
        tracker = ThreadedHandTracker(model_path)
        tracker.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
