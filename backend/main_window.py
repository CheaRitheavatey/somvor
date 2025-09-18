import tkinter as tk
from tkinter import ttk
from main_camera_frame import CameraFrame
from main_control_panel import ControlPanel
from main_settings_panel import SettingsPanel
from main_gesture_detector import GestureDetector
from main_theme_manager import ThemeManager

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition System")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        
        # Initialize gesture detector
        self.gesture_detector = GestureDetector()
        
        # Configure root
        self.root.configure(bg=self.theme_manager.get_color('bg'))
        
        # Create main container
        self.main_container = tk.Frame(self.root, bg=self.theme_manager.get_color('bg'))
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel (camera and controls)
        self.left_panel = tk.Frame(self.main_container, bg=self.theme_manager.get_color('bg'))
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right panel (settings)
        self.right_panel = tk.Frame(self.main_container, bg=self.theme_manager.get_color('panel_bg'), 
                                  relief='raised', bd=1)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Initialize components
        self.camera_frame = CameraFrame(self.left_panel, self.theme_manager, self.gesture_detector)
        self.control_panel = ControlPanel(self.left_panel, self.theme_manager, self.camera_frame)
        self.settings_panel = SettingsPanel(self.right_panel, self.theme_manager, self.on_settings_change)
        
        # Pack components
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        self.control_panel.pack(fill=tk.X, pady=(10, 0))
        self.settings_panel.pack(fill=tk.BOTH, expand=True)
        
        # Apply initial theme
        self.apply_theme()
        
    def on_settings_change(self, setting_name, value):
        """Handle settings changes"""
        if setting_name == 'theme':
            self.theme_manager.set_theme(value)
            self.apply_theme()
        elif setting_name == 'language':
            self.camera_frame.set_language(value)
        elif setting_name == 'confidence':
            self.gesture_detector.set_confidence_threshold(value)
        elif setting_name == 'subtitle_size':
            self.camera_frame.set_subtitle_size(value)
            
    def apply_theme(self):
        """Apply the current theme to all components"""
        colors = self.theme_manager.get_all_colors()
        
        # Update root and main containers
        self.root.configure(bg=colors['bg'])
        self.main_container.configure(bg=colors['bg'])
        self.left_panel.configure(bg=colors['bg'])
        self.right_panel.configure(bg=colors['panel_bg'])
        
        # Update components
        self.camera_frame.apply_theme(colors)
        self.control_panel.apply_theme(colors)
        self.settings_panel.apply_theme(colors)