import tkinter as tk
from tkinter import ttk

class ControlPanel(tk.Frame):
    def __init__(self, parent, theme_manager, camera_frame):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.camera_frame = camera_frame
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup control panel UI"""
        # Control buttons frame
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(expand=True)
        
        # Start button
        self.start_btn = tk.Button(self.buttons_frame, text="▶ Start Camera", 
                                 font=('Arial', 14, 'bold'),
                                 command=self.start_camera,
                                 relief='flat', cursor='hand2')
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_btn = tk.Button(self.buttons_frame, text="⏹ Stop Camera", 
                                font=('Arial', 14, 'bold'),
                                command=self.stop_camera,
                                relief='flat', cursor='hand2',
                                state='disabled')
        self.stop_btn.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = tk.Label(self, text="Camera: Stopped", 
                                   font=('Arial', 12))
        self.status_label.pack(pady=(10, 0))
        
    def start_camera(self):
        """Start camera and update UI"""
        self.camera_frame.start_camera()
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.status_label.configure(text="Camera: Running")
        
    def stop_camera(self):
        """Stop camera and update UI"""
        self.camera_frame.stop_camera()
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.status_label.configure(text="Camera: Stopped")
        
    def apply_theme(self, colors):
        """Apply theme colors"""
        self.configure(bg=colors['bg'])
        self.buttons_frame.configure(bg=colors['bg'])
        
        # Update buttons
        self.start_btn.configure(
            bg=colors['accent'], 
            fg='white',
            activebackground=colors['accent_hover']
        )
        self.stop_btn.configure(
            bg=colors['error'], 
            fg='white',
            activebackground=colors['error_hover']
        )
        
        self.status_label.configure(bg=colors['bg'], fg=colors['text'])