"""
Enhanced system actions and macOS integration for blazing performance
"""
import subprocess
import pyautogui
import time
from typing import Optional, Dict, Any


class SystemController:
    """Handle system-level actions and macOS integrations with enhanced performance"""
    
    def __init__(self):
        # Configure pyautogui for maximum performance
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.005  # Reduced pause for faster response
        
        # Action cooldowns to prevent spam
        self.last_mission_control = 0
        self.last_special_action = 0
        self.mission_control_cooldown = 0.8  # Reduced cooldown
        self.special_action_cooldown = 0.5
        
        # Performance tracking
        self.action_history = []
        self.max_history = 10
        
    def trigger_mission_control(self) -> bool:
        """Trigger macOS Mission Control (expose all windows)"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_mission_control < self.mission_control_cooldown:
            return False
        
        try:
            # Use AppleScript to trigger Mission Control
            script = '''
            tell application "System Events"
                key code 126 using {control down}
            end tell
            '''
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            
            self.last_mission_control = current_time
            print("🚀 Mission Control activated!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to trigger Mission Control: {e}")
            
            # Fallback: Try F3 key (common Mission Control shortcut)
            try:
                pyautogui.press('f3')
                self.last_mission_control = current_time
                print("🚀 Mission Control activated (fallback)!")
                return True
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                return False
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def trigger_desktop_zoom_out(self) -> bool:
        """Alternative: Zoom out to see desktop"""
        try:
            # F11 or Cmd+F3 for desktop view
            pyautogui.hotkey('cmd', 'f3')
            print("🖥️  Desktop view activated!")
            return True
        except Exception as e:
            print(f"❌ Failed to trigger desktop view: {e}")
            return False
    
    def left_click(self) -> bool:
        """Perform left mouse click"""
        try:
            pyautogui.click()
            return True
        except Exception as e:
            print(f"❌ Failed to click: {e}")
            return False
    
    def move_cursor(self, x: float, y: float) -> bool:
        """Move cursor to specified position"""
        try:
            pyautogui.moveTo(x, y)
            return True
        except Exception as e:
            print(f"❌ Failed to move cursor: {e}")
            return False
    
    def trigger_special_action(self, action_type: str = "desktop_view") -> bool:
        """Trigger special system actions"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_special_action < self.special_action_cooldown:
            return False
        
        try:
            if action_type == "desktop_view":
                # Show desktop (F11)
                pyautogui.press('f11')
                print("🖥️ Desktop view activated!")
            elif action_type == "expose":
                # Alternative expose
                pyautogui.hotkey('fn', 'f9')
                print("📺 Expose activated!")
            elif action_type == "launchpad":
                # Launchpad
                pyautogui.hotkey('fn', 'f4')
                print("🚀 Launchpad activated!")
            
            self.last_special_action = current_time
            self._log_action(action_type)
            return True
            
        except Exception as e:
            print(f"❌ Failed to trigger {action_type}: {e}")
            return False
    
    def quick_click(self, x: Optional[float] = None, y: Optional[float] = None) -> bool:
        """Perform optimized click with optional position"""
        try:
            if x is not None and y is not None:
                pyautogui.click(x, y)
            else:
                pyautogui.click()
            return True
        except Exception as e:
            print(f"❌ Failed to click: {e}")
            return False
    
    def drag_cursor(self, start_x: float, start_y: float, end_x: float, end_y: float, duration: float = 0.1) -> bool:
        """Perform drag operation"""
        try:
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration, button='left')
            return True
        except Exception as e:
            print(f"❌ Failed to drag: {e}")
            return False
    
    def scroll(self, direction: str, amount: int = 3) -> bool:
        """Perform scroll action"""
        try:
            if direction == "up":
                pyautogui.scroll(amount)
            elif direction == "down":
                pyautogui.scroll(-amount)
            elif direction == "left":
                pyautogui.hscroll(-amount)
            elif direction == "right":
                pyautogui.hscroll(amount)
            return True
        except Exception as e:
            print(f"❌ Failed to scroll: {e}")
            return False
    
    def mouse_down(self) -> bool:
        """Press and hold mouse button for drag operations"""
        try:
            pyautogui.mouseDown()
            return True
        except Exception as e:
            print(f"❌ Failed to mouse down: {e}")
            return False
    
    def mouse_up(self) -> bool:
        """Release mouse button to end drag operations"""
        try:
            pyautogui.mouseUp()
            return True
        except Exception as e:
            print(f"❌ Failed to mouse up: {e}")
            return False
    
    def move_cursor_while_dragging(self, x: float, y: float) -> bool:
        """Move cursor while maintaining drag state"""
        try:
            pyautogui.dragTo(x, y, duration=0.01, button='left')
            return True
        except Exception as e:
            print(f"❌ Failed to drag cursor: {e}")
            return False
    
    def right_click(self) -> bool:
        """Perform right mouse click"""
        try:
            pyautogui.rightClick()
            return True
        except Exception as e:
            print(f"❌ Failed to right click: {e}")
            return False

    def _log_action(self, action: str):
        """Log action for performance tracking"""
        self.action_history.append({
            "action": action,
            "timestamp": time.time()
        })
        
        # Keep history manageable
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get action performance statistics"""
        if not self.action_history:
            return {"total_actions": 0, "recent_actions": []}
        
        current_time = time.time()
        recent_actions = [
            action for action in self.action_history
            if current_time - action["timestamp"] < 10.0  # Last 10 seconds
        ]
        
        return {
            "total_actions": len(self.action_history),
            "recent_actions": len(recent_actions),
            "last_action": self.action_history[-1]["action"] if self.action_history else None
        }
