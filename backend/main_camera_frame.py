import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import queue

class CameraFrame(tk.Frame):
    def __init__(self, parent, theme_manager, gesture_detector):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.gesture_detector = gesture_detector
        
        self.is_running = False
        self.cap = None
        self.current_subtitle = ""
        self.current_confidence = 0.0
        self.subtitle_size = "medium"
        self.language = "english"
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the camera display UI"""
        # Main camera container
        self.camera_container = tk.Frame(self, bg='black', relief='solid', bd=2)
        self.camera_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera display label
        self.camera_label = tk.Label(self.camera_container, bg='black', text="📷 Camera Off", 
                                   font=('Arial', 18), fg='white')
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Subtitle container with better styling
        self.subtitle_container = tk.Frame(self, height=100, relief='solid', bd=1)
        self.subtitle_container.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=(0, 5))
        self.subtitle_container.pack_propagate(False)
        
        # Subtitle display with confidence
        self.subtitle_label = tk.Label(self.subtitle_container, text="Ready for gesture recognition...", 
                                     font=('Arial', 24, 'bold'), wraplength=800, justify='center')
        self.subtitle_label.pack(expand=True, pady=10)
        
        # Confidence indicator
        self.confidence_label = tk.Label(self.subtitle_container, text="", 
                                       font=('Arial', 12), fg='gray')
        self.confidence_label.pack()
        
    def set_language(self, language):
        """Set subtitle language"""
        self.language = language
        
    def start_camera(self):
        """Start camera capture"""
        if not self.is_running:
            self.is_running = True
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Could not open camera")
                    
                # Set camera properties for better quality
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.camera_thread = threading.Thread(target=self.camera_loop)
                self.camera_thread.daemon = True
                self.camera_thread.start()
                
                self.update_subtitle("Camera started - Show your hand gesture", 0.0)
                
            except Exception as e:
                self.is_running = False
                self.update_subtitle(f"Camera error: {str(e)}", 0.0)
                print(f"Camera error: {e}")
            
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_label.configure(image='', text="📷 Camera Off")
        self.current_subtitle = ""
        self.current_confidence = 0.0
        self.update_subtitle("Camera stopped", 0.0)
        
    def camera_loop(self):
        """Main camera capture loop with improved processing"""
        frame_count = 0
        gesture_history = []
        stable_gesture = None
        stable_count = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for gesture detection
            processed_frame, detected_gesture, confidence = self.gesture_detector.process_frame(frame)
            
            # Gesture stabilization - only update if gesture is consistent
            if detected_gesture is not None:
                gesture_history.append(detected_gesture)
                if len(gesture_history) > 5:
                    gesture_history.pop(0)
                
                # Check if gesture is stable (appears in most recent frames)
                if len(gesture_history) >= 3:
                    most_common = max(set(gesture_history), key=gesture_history.count)
                    if gesture_history.count(most_common) >= 2:
                        if stable_gesture != most_common:
                            stable_gesture = most_common
                            stable_count = 0
                        stable_count += 1
                        
                        # Update subtitle only if gesture is stable for a few frames
                        if stable_count >= 2:
                            gesture_text = self.gesture_detector.get_gesture_text(stable_gesture, self.language)
                            self.current_subtitle = gesture_text
                            self.current_confidence = confidence
            else:
                # Clear gesture history if no gesture detected
                if len(gesture_history) > 0:
                    gesture_history.clear()
                    stable_gesture = None
                    stable_count = 0
                    if frame_count % 30 == 0:  # Update less frequently when no gesture
                        self.current_subtitle = "Show your hand gesture..."
                        self.current_confidence = 0.0
            
            # Convert frame to display format
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize frame to fit display while maintaining aspect ratio
            display_width = 640
            display_height = 480
            frame_pil = frame_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update display (only every few frames to reduce CPU usage)
            if frame_count % 2 == 0:
                try:
                    self.camera_label.configure(image=frame_tk, text="")
                    self.camera_label.image = frame_tk
                    
                    # Update subtitle
                    self.update_subtitle(self.current_subtitle, self.current_confidence)
                except tk.TclError:
                    # Widget destroyed, exit loop
                    break
            
    def update_subtitle(self, text, confidence):
        """Update subtitle display with confidence"""
        try:
            if text:
                self.subtitle_label.configure(text=text)
                if confidence > 0:
                    confidence_text = f"Confidence: {confidence:.1%}"
                    color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.6 else "red"
                    self.confidence_label.configure(text=confidence_text, fg=color)
                else:
                    self.confidence_label.configure(text="")
            else:
                self.subtitle_label.configure(text="Ready for gesture recognition...")
                self.confidence_label.configure(text="")
        except tk.TclError:
            pass
            
    def set_subtitle_size(self, size):
        """Set subtitle size"""
        self.subtitle_size = size
        font_sizes = {"small": 18, "medium": 24, "large": 32}
        font_size = font_sizes.get(size, 24)
        self.subtitle_label.configure(font=('Arial', font_size, 'bold'))
        
    def apply_theme(self, colors):
        """Apply theme colors"""
        self.configure(bg=colors['bg'])
        self.camera_container.configure(bg='black', highlightbackground=colors['border'])
        self.subtitle_container.configure(bg=colors['panel_bg'], highlightbackground=colors['border'])
        self.subtitle_label.configure(bg=colors['panel_bg'], fg=colors['text'])
        
        # Update confidence label color based on theme
        current_fg = self.confidence_label.cget('fg')
        if current_fg == 'gray':
            self.confidence_label.configure(bg=colors['panel_bg'])