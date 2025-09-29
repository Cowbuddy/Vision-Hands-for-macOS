#!/usr/bin/env python3
"""
Performance Benchmarking Tool for Hand Tracking System

Measures and reports performance metrics including:
- FPS (frames per second)
- Frame processing time
- Memory usage
- Gesture detection latency
- Cursor movement latency
"""
import os
import time
import psutil
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
import cv2

# Import MediaPipe
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

from src.hand_analyzer import HandAnalyzer


class PerformanceBenchmark:
    """Performance benchmarking utility"""
    
    def __init__(self, duration_seconds: int = 30):
        self.duration = duration_seconds
        self.process = psutil.Process(os.getpid())
        
        # Metrics storage
        self.frame_times = deque(maxlen=1000)
        self.detection_times = deque(maxlen=1000)
        self.analysis_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        
        # Initialize components
        self.hand_landmarker = self._init_landmarker()
        self.hand_analyzer = HandAnalyzer()
        
        print("🔬 Performance Benchmark Tool")
        print("="*60)
        print(f"Duration: {duration_seconds} seconds")
        print("="*60)
    
    def _init_landmarker(self):
        """Initialize MediaPipe hand landmarker"""
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        base_opts = BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU
        )
        
        options = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        return HandLandmarker.create_from_options(options)
    
    def benchmark_frame_processing(self, frame, timestamp_ms: int) -> Dict:
        """Benchmark a single frame processing"""
        metrics = {}
        
        # Total frame processing time
        start_total = time.perf_counter()
        
        # Convert to RGB
        start_convert = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        metrics['conversion_time'] = time.perf_counter() - start_convert
        
        # Hand detection
        start_detection = time.perf_counter()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        metrics['detection_time'] = time.perf_counter() - start_detection
        
        # Hand analysis
        start_analysis = time.perf_counter()
        if result.hand_landmarks and result.handedness:
            frame_height, frame_width = frame.shape[:2]
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                hand_info = self.hand_analyzer.analyze_hand(
                    landmarks, handedness, frame_width, frame_height
                )
        metrics['analysis_time'] = time.perf_counter() - start_analysis
        
        # Total time
        metrics['total_time'] = time.perf_counter() - start_total
        metrics['num_hands'] = len(result.hand_landmarks) if result.hand_landmarks else 0
        
        return metrics
    
    def run_benchmark(self):
        """Run the benchmark"""
        print("\n🎥 Initializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("✅ Camera ready")
        print("\n🏃 Running benchmark...")
        print("👋 Show your hands to test detection performance\n")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < self.duration:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                timestamp_ms = int(frame_count * 16.67)
                
                # Benchmark this frame
                metrics = self.benchmark_frame_processing(frame, timestamp_ms)
                
                # Store metrics
                self.frame_times.append(metrics['total_time'])
                self.detection_times.append(metrics['detection_time'])
                self.analysis_times.append(metrics['analysis_time'])
                
                # Memory usage (sample every 10 frames)
                if frame_count % 10 == 0:
                    mem_info = self.process.memory_info()
                    self.memory_usage.append(mem_info.rss / 1024 / 1024)  # MB
                
                # Display progress
                frame_count += 1
                elapsed = time.time() - start_time
                progress = (elapsed / self.duration) * 100
                
                # Add metrics to frame
                cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {1.0/metrics['total_time']:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Hands: {metrics['num_hands']}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Performance Benchmark', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit early
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️  Benchmark interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Generate report
        self.generate_report(frame_count, time.time() - start_time)
    
    def generate_report(self, total_frames: int, elapsed_time: float):
        """Generate and display benchmark report"""
        print("\n" + "="*60)
        print("📊 BENCHMARK REPORT")
        print("="*60)
        
        # Overall performance
        print("\n🎯 Overall Performance:")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {elapsed_time:.2f} seconds")
        print(f"  Average FPS: {total_frames / elapsed_time:.2f}")
        
        # Frame processing times
        if self.frame_times:
            frame_times_ms = [t * 1000 for t in self.frame_times]
            print("\n⏱️  Frame Processing Time (milliseconds):")
            print(f"  Mean: {np.mean(frame_times_ms):.2f} ms")
            print(f"  Median: {np.median(frame_times_ms):.2f} ms")
            print(f"  Min: {np.min(frame_times_ms):.2f} ms")
            print(f"  Max: {np.max(frame_times_ms):.2f} ms")
            print(f"  Std Dev: {np.std(frame_times_ms):.2f} ms")
            print(f"  P95: {np.percentile(frame_times_ms, 95):.2f} ms")
            print(f"  P99: {np.percentile(frame_times_ms, 99):.2f} ms")
        
        # Detection times
        if self.detection_times:
            detection_times_ms = [t * 1000 for t in self.detection_times]
            print("\n🔍 Hand Detection Time (milliseconds):")
            print(f"  Mean: {np.mean(detection_times_ms):.2f} ms")
            print(f"  Median: {np.median(detection_times_ms):.2f} ms")
            print(f"  P95: {np.percentile(detection_times_ms, 95):.2f} ms")
        
        # Analysis times
        if self.analysis_times:
            analysis_times_ms = [t * 1000 for t in self.analysis_times]
            print("\n📊 Hand Analysis Time (milliseconds):")
            print(f"  Mean: {np.mean(analysis_times_ms):.2f} ms")
            print(f"  Median: {np.median(analysis_times_ms):.2f} ms")
            print(f"  P95: {np.percentile(analysis_times_ms, 95):.2f} ms")
        
        # Memory usage
        if self.memory_usage:
            print("\n💾 Memory Usage (MB):")
            print(f"  Mean: {np.mean(self.memory_usage):.2f} MB")
            print(f"  Min: {np.min(self.memory_usage):.2f} MB")
            print(f"  Max: {np.max(self.memory_usage):.2f} MB")
            print(f"  Delta: {np.max(self.memory_usage) - np.min(self.memory_usage):.2f} MB")
        
        # FPS distribution
        if self.frame_times:
            fps_values = [1.0 / t for t in self.frame_times if t > 0]
            print("\n📈 FPS Distribution:")
            print(f"  Mean: {np.mean(fps_values):.2f} FPS")
            print(f"  Median: {np.median(fps_values):.2f} FPS")
            print(f"  Min: {np.min(fps_values):.2f} FPS")
            print(f"  Max: {np.max(fps_values):.2f} FPS")
            
            # FPS buckets
            fps_30_plus = sum(1 for fps in fps_values if fps >= 30)
            fps_60_plus = sum(1 for fps in fps_values if fps >= 60)
            total = len(fps_values)
            
            print(f"\n  Frames >= 30 FPS: {fps_30_plus}/{total} ({fps_30_plus/total*100:.1f}%)")
            print(f"  Frames >= 60 FPS: {fps_60_plus}/{total} ({fps_60_plus/total*100:.1f}%)")
        
        # Performance rating
        avg_fps = total_frames / elapsed_time
        print("\n🏆 Performance Rating:")
        if avg_fps >= 55:
            print("  ★★★★★ EXCELLENT - Smooth 60 FPS performance")
        elif avg_fps >= 45:
            print("  ★★★★☆ VERY GOOD - Near 60 FPS")
        elif avg_fps >= 30:
            print("  ★★★☆☆ GOOD - Smooth 30+ FPS")
        elif avg_fps >= 20:
            print("  ★★☆☆☆ FAIR - Usable but choppy")
        else:
            print("  ★☆☆☆☆ POOR - Performance issues detected")
        
        print("\n" + "="*60)
        
        # Recommendations
        print("\n💡 Recommendations:")
        if avg_fps < 30:
            print("  • Close other applications using the camera")
            print("  • Reduce camera resolution in config")
            print("  • Disable UI elements in config")
        if self.memory_usage and (max(self.memory_usage) - min(self.memory_usage)) > 100:
            print("  • Memory usage growing - possible memory leak")
            print("  • Check garbage collection settings")
        if self.frame_times and np.std([t * 1000 for t in self.frame_times]) > 5:
            print("  • High frame time variance detected")
            print("  • Close background applications for more consistent performance")
        
        print("\n" + "="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hand Tracking Performance Benchmark')
    parser.add_argument('-d', '--duration', type=int, default=30,
                       help='Benchmark duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Check for model file
    if not os.path.exists("hand_landmarker.task"):
        print("❌ Model file not found: hand_landmarker.task")
        print("Please run setup.sh first to download the model")
        return
    
    # Run benchmark
    benchmark = PerformanceBenchmark(duration_seconds=args.duration)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()