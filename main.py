import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QTextEdit, QFrame, QGroupBox, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon
import random

class SignLanguageConverterUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Converter")
        self.setGeometry(100, 100, 900, 700)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccccd0;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #007aff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton#startButton {
                background-color: #34c759;
                font-size: 16px;
                padding: 15px 30px;
            }
            QPushButton#startButton:hover {
                background-color: #2ca44e;
            }
            QPushButton#inactiveButton {
                background-color: #ff3b30;
            }
            QPushButton#inactiveButton:hover {
                background-color: #d70015;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #ccccd0;
                border-radius: 6px;
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #ccccd0;
                border-radius: 6px;
                background-color: white;
                padding: 10px;
            }
            QLabel#titleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #1d1d1f;
            }
            QLabel#subtitleLabel {
                font-size: 16px;
                color: #6e6e73;
            }
            QLabel#liveLabel {
                color: #ff3b30;
                font-weight: bold;
                font-size: 16px;
            }
            QLabel#statusLabel {
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QLabel#inactiveStatus {
                background-color: #ff3b30;
                color: white;
            }
            QLabel#activeStatus {
                background-color: #34c759;
                color: white;
            }
        """)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(30, 30, 30, 30)
        
        self.setup_ui()
        
        # Timer to simulate live updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_subtitles)
        self.detection_active = False
        
    def setup_ui(self):
        # Header section
        header_layout = QVBoxLayout()
        title_label = QLabel("Sign Language Converter")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        
        subtitle_label = QLabel("Real-time ASL to subtitle conversion")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        self.layout.addLayout(header_layout)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left column - Detection controls
        left_column = QVBoxLayout()
        left_column.setSpacing(20)
        
        # Start Detection button
        self.start_button = QPushButton("Start Detection")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.toggle_detection)
        left_column.addWidget(self.start_button)
        
        # Live indicator
        live_indicator = QHBoxLayout()
        live_label = QLabel("LIVE")
        live_label.setObjectName("liveLabel")
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #ff3b30; font-size: 16px;")
        live_indicator.addWidget(self.status_indicator)
        live_indicator.addWidget(live_label)
        live_indicator.addStretch()
        left_column.addLayout(live_indicator)
        
        # Subtitle History group
        history_group = QGroupBox("Subtitle History")
        history_layout = QVBoxLayout()
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setPlaceholderText("No subtitles generated yet\nDetected signs will appear here")
        history_layout.addWidget(self.history_text)
        history_group.setLayout(history_layout)
        left_column.addWidget(history_group)
        
        content_layout.addLayout(left_column, 2)
        
        # Right column - Status and settings
        right_column = QVBoxLayout()
        right_column.setSpacing(20)
        
        # Sign Detection status group
        status_group = QGroupBox("Sign Detection")
        status_layout = QVBoxLayout()
        
        status_row = QHBoxLayout()
        status_label = QLabel("Status:")
        self.status_value = QLabel("Inactive")
        self.status_value.setObjectName("statusLabel")
        self.status_value.setObjectName("inactiveStatus")
        status_row.addWidget(status_label)
        status_row.addWidget(self.status_value)
        status_row.addStretch()
        status_layout.addLayout(status_row)
        
        method_row = QHBoxLayout()
        method_label = QLabel("Detection Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["MedialPipe", "OpenCV", "TensorFlow", "Custom CNN"])
        method_row.addWidget(method_label)
        method_row.addWidget(self.method_combo)
        method_row.addStretch()
        status_layout.addLayout(method_row)
        
        status_group.setLayout(status_layout)
        right_column.addWidget(status_group)
        
        # Video Demo group
        demo_group = QGroupBox("Video Demo")
        demo_layout = QVBoxLayout()
        demo_label = QLabel("Activate Webcam\nGo to Settings")
        demo_label.setAlignment(Qt.AlignCenter)
        demo_label.setStyleSheet("color: #6e6e73; padding: 40px;")
        demo_layout.addWidget(demo_label)
        demo_group.setLayout(demo_layout)
        right_column.addWidget(demo_group)
        
        content_layout.addLayout(right_column, 1)
        
        self.layout.addLayout(content_layout)
        
    def toggle_detection(self):
        self.detection_active = not self.detection_active
        
        if self.detection_active:
            self.start_button.setText("Stop Detection")
            self.start_button.setStyleSheet("""
                QPushButton#startButton {
                    background-color: #ff3b30;
                    font-size: 16px;
                    padding: 15px 30px;
                }
                QPushButton#startButton:hover {
                    background-color: #d70015;
                }
            """)
            self.status_value.setText("Active")
            self.status_value.setObjectName("activeStatus")
            self.status_indicator.setStyleSheet("color: #34c759; font-size: 16px;")
            self.timer.start(2000)  # Update every 2 seconds
        else:
            self.start_button.setText("Start Detection")
            self.start_button.setStyleSheet("""
                QPushButton#startButton {
                    background-color: #34c759;
                    font-size: 16px;
                    padding: 15px 30px;
                }
                QPushButton#startButton:hover {
                    background-color: #2ca44e;
                }
            """)
            self.status_value.setText("Inactive")
            self.status_value.setObjectName("inactiveStatus")
            self.status_indicator.setStyleSheet("color: #ff3b30; font-size: 16px;")
            self.timer.stop()
            
        # Update the stylesheet
        self.setStyleSheet(self.styleSheet())
    
    def update_subtitles(self):
        # Simulate receiving new sign language translations
        phrases = [
            "Hello, how are you?",
            "Thank you very much",
            "I need help please",
            "Where is the restroom?",
            "Nice to meet you",
            "What is your name?",
            "I am learning sign language",
            "Have a good day!",
            "Can you repeat that?",
            "I don't understand"
        ]
        
        new_phrase = random.choice(phrases)
        current_text = self.history_text.toPlainText()
        
        if current_text == "No subtitles generated yet\nDetected signs will appear here":
            self.history_text.setText(new_phrase)
        else:
            self.history_text.append(new_phrase)
            
        # Auto-scroll to bottom
        self.history_text.verticalScrollBar().setValue(
            self.history_text.verticalScrollBar().maximum()
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = SignLanguageConverterUI()
    window.show()
    sys.exit(app.exec_())