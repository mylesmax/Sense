import asyncio
import sys
import requests
from bleak import BleakScanner, BleakClient
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QLineEdit, QSpinBox, QDialog, QCheckBox,
                             QMessageBox, QGridLayout, QDialogButtonBox, QProgressBar)
from PyQt6.QtCore import QTimer, QRunnable, QThreadPool, pyqtSignal, QObject, Qt, QRectF, QPointF
from PyQt6.QtGui import QIcon, QPainter, QPen, QBrush, QColor, QPainterPath, QFont
import pyqtgraph as pg
import numpy as np
from collections import deque
import os
from datetime import datetime
import csv
from qasync import QEventLoop, asyncSlot
import traceback
import pandas as pd
import scipy.signal as signal

# Constants
MULTI_SENSOR_SERVICE_UUID = "2cc12ee8-c5b6-4d7f-a3de-9c793653f271"
MULTI_SENSOR_CHARACTERISTIC_UUID = "15216e4f-bf54-4482-8a91-74a92ccfeb37"
MULTI_SENSOR_DEVICE_NAME = "03senseV3"

TIME_WINDOW = 30
MAX_DATA_POINTS = 1000

BASELINE_SECONDS = 10
QUADRANT_SECONDS = 10
COUNTDOWN_SECONDS = 3

API_ENDPOINT = "http://ec2-3-128-206-158.us-east-2.compute.amazonaws.com:5000/predict"

class BLEWorker(QObject, QRunnable):
    data = pyqtSignal(dict)
    connection_status = pyqtSignal(str)

    def __init__(self, service_uuid, characteristic_uuid, device_name):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self.current_json = ""
        self.client = None
        self.is_running = True
        self.service_uuid = service_uuid
        self.characteristic_uuid = characteristic_uuid
        self.device_name = device_name

    def notification_handler(self, sender, data):
        """Callback for when notification data is received"""
        try:
            decoded_data = data.decode('utf-8').strip('\x00')
            
            if decoded_data == "START":
                self.current_json = ""
            elif decoded_data == "END":
                self.process_json(self.current_json)
            else:
                self.current_json += decoded_data
            
        except Exception as e:
            print(f"Error in notification_handler: {e}", file=sys.stderr)

    def process_json(self, json_string):
        """Process complete JSON string"""
        try:
            # Clean up the JSON string
            json_string = json_string.rstrip(',')
            if not json_string.endswith('}'):
                json_string += '}'
            
            # Add debug output
            print(f"Processing JSON: {json_string}", file=sys.stderr)
            
            data = json.loads(json_string)
            self.data.emit(data)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)
            print(f"Problematic JSON string: {json_string}", file=sys.stderr)
        except Exception as e:
            print(f"Error in process_json: {e}", file=sys.stderr)

    async def connect_and_receive(self, address):
        while self.is_running:
            try:
                self.connection_status.emit("connecting")
                async with BleakClient(address) as client:
                    self.client = client
                    if client.is_connected:
                        print(f"Connected to {self.device_name} [{address}]", file=sys.stderr)
                        self.connection_status.emit("connected")
                        await client.start_notify(self.characteristic_uuid, self.notification_handler)
                        print(f"Subscribed to notifications for {self.device_name}. Awaiting data...", file=sys.stderr)
                        while client.is_connected and self.is_running:
                            await asyncio.sleep(1)
                    else:
                        print(f"Failed to connect to {self.device_name} [{address}]", file=sys.stderr)
                        self.connection_status.emit("disconnected")
            except Exception as e:
                print(f"Connection error for {self.device_name}: {e}", file=sys.stderr)
                self.connection_status.emit("disconnected")
            
            if self.is_running:
                print(f"Reconnecting to {self.device_name} in 5 seconds...")
                await asyncio.sleep(5)

    async def run(self):
        while self.is_running:
            address = await self.discover()
            if address:
                await self.connect_and_receive(address)
            else:
                print(f"Cannot proceed without connecting to the {self.device_name}.")
                self.connection_status.emit("disconnected")
                await asyncio.sleep(5)

    async def discover(self):
        self.connection_status.emit("searching")
        print(f"Scanning for {self.device_name}...", file=sys.stderr)
        devices = await BleakScanner.discover(timeout=5.0)
        device = next((d for d in devices if d.name == self.device_name), None)
        if device:
            print(f"Found target device: {device.name} [{device.address}]", file=sys.stderr)
            return device.address
        print(f"Device named '{self.device_name}' not found.", file=sys.stderr)
        self.connection_status.emit("disconnected")
        return None

    def stop(self):
        self.is_running = False 

# QuadrantDialog class for displaying quadrants during recording
class QuadrantDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recording Quadrants")
        self.setFixedSize(400, 450)  # Height includes space for countdown label
        self.setModal(True)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Countdown label
        self.countdown_label = QLabel("Starting in: 3")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(self.countdown_label)
        
        # Current phase label
        self.phase_label = QLabel("Preparing baseline...")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setStyleSheet("font-size: 14pt;")
        layout.addWidget(self.phase_label)
        
        # Circle widget for quadrants
        self.quadrant_widget = QuadrantWidget()
        layout.addWidget(self.quadrant_widget)
        
        # Initialize active quadrant (None, 'baseline', 'topright', 'bottomright', 'bottomleft', 'topleft')
        self.active_quadrant = None
        self.set_active_quadrant(None)
        
    def set_active_quadrant(self, quadrant):
        """Set the active quadrant to highlight"""
        self.active_quadrant = quadrant
        self.quadrant_widget.active_quadrant = quadrant
        self.quadrant_widget.update()
        
        # Update phase label based on active quadrant
        if quadrant == None:
            self.phase_label.setText("Preparing baseline...")
        elif quadrant == 'baseline':
            self.phase_label.setText("Recording baseline...")
        elif quadrant == 'topright':
            self.phase_label.setText("Top Right Quadrant")
        elif quadrant == 'bottomright':
            self.phase_label.setText("Bottom Right Quadrant")
        elif quadrant == 'bottomleft':
            self.phase_label.setText("Bottom Left Quadrant")
        elif quadrant == 'topleft':
            self.phase_label.setText("Top Left Quadrant")
            
    def set_countdown(self, seconds):
        """Update the countdown display"""
        self.countdown_label.setText(f"Time remaining: {seconds} s")

    def showEvent(self, event):
        # Center the dialog on the parent window
        if self.parent():
            parent_geo = self.parent().geometry()
            self.move(
                parent_geo.x() + (parent_geo.width() - self.width()) // 2,
                parent_geo.y() + (parent_geo.height() - self.height()) // 2
            )
        super().showEvent(event)

# Widget to draw the quadrants
class QuadrantWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 350)
        self.active_quadrant = None
        
        # Define colors for each quadrant
        self.quadrant_colors = {
            'topleft': QColor(128, 0, 255),     # Purple
            'topright': QColor(0, 120, 255),    # Blue
            'bottomleft': QColor(0, 180, 0),    # Green
            'bottomright': QColor(255, 120, 0),  # Orange
            'baseline': QColor(220, 50, 100)    # Pink/magenta for baseline
        }
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate circle dimensions
        width = self.width()
        height = self.height()
        size = min(width, height) - 20  # Leave some margin
        x = (width - size) // 2
        y = (height - size) // 2
        
        # Draw main circle with light gray background
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setBrush(QBrush(QColor(245, 245, 245)))  # Light gray background
        painter.drawEllipse(x, y, size, size)
        
        # Draw dividing lines
        painter.setPen(QPen(Qt.GlobalColor.darkGray, 1, Qt.PenStyle.DashLine))
        painter.drawLine(x + size // 2, y, x + size // 2, y + size)  # Vertical
        painter.drawLine(x, y + size // 2, x + size, y + size // 2)  # Horizontal
        
        # Baseline is the center circle
        center_size = size // 3
        center_x = x + (size - center_size) // 2
        center_y = y + (size - center_size) // 2
        
        # Define each quadrant
        quarters = [
            ('topleft', x, y, size // 2, size // 2),
            ('topright', x + size // 2, y, size // 2, size // 2),
            ('bottomleft', x, y + size // 2, size // 2, size // 2),
            ('bottomright', x + size // 2, y + size // 2, size // 2, size // 2)
        ]
        
        # First draw all quadrants (selected ones opaque, others transparent)
        for name, qx, qy, qw, qh in quarters:
            # Create a path for the quarter
            path = QPainterPath()
            
            # Calculate center of the entire circle
            circle_center_x = x + size // 2
            circle_center_y = y + size // 2
            
            # Starting point is the center of the circle
            path.moveTo(circle_center_x, circle_center_y)
            
            # Draw the arc for this quadrant
            if name == 'topleft':
                path.arcTo(QRectF(x, y, size, size), 180, -90)
            elif name == 'topright':
                path.arcTo(QRectF(x, y, size, size), 90, -90)
            elif name == 'bottomleft':
                path.arcTo(QRectF(x, y, size, size), 180, 90)
            elif name == 'bottomright':
                path.arcTo(QRectF(x, y, size, size), 270, 90)
            
            # Close the path back to center
            path.lineTo(circle_center_x, circle_center_y)
            
            # Fill with appropriate color
            color = self.quadrant_colors[name]
            
            # If this quadrant is selected, make it fully opaque
            is_selected = (self.active_quadrant == name)
            if is_selected:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
            else:
                # Make unselected quadrants transparent
                transparent_color = QColor(color)
                transparent_color.setAlpha(30)  # Almost transparent
                painter.setBrush(QBrush(transparent_color))
                painter.setPen(QPen(Qt.GlobalColor.gray, 1, Qt.PenStyle.DotLine))
                
            painter.drawPath(path)
            
            # Add a label to each quadrant
            text_x = qx + qw // 2
            text_y = qy + qh // 2
            
            # Adjust positions for better label placement
            if name == 'topleft':
                text_x -= qw // 4
                text_y -= qh // 4
            elif name == 'topright':
                text_x += qw // 4
                text_y -= qh // 4
            elif name == 'bottomleft':
                text_x -= qw // 4
                text_y += qh // 4
            elif name == 'bottomright':
                text_x += qw // 4
                text_y += qh // 4
            
            # Draw text with white background for better readability
            if is_selected:
                # For selected quadrant, make text more prominent
                font = painter.font()
                font.setBold(True)
                painter.setFont(font)
                
                # Text background
                painter.setPen(QPen(Qt.GlobalColor.white))
                painter.setBrush(QBrush(QColor(255, 255, 255, 180)))  # Semi-transparent white
                painter.drawRoundedRect(text_x - 25, text_y - 10, 50, 20, 5, 5)
                
                # Text
                painter.setPen(QPen(Qt.GlobalColor.black))
                painter.drawText(text_x - 25, text_y - 10, 50, 20, 
                            Qt.AlignmentFlag.AlignCenter, name.capitalize())
                
                # Reset font
                font.setBold(False)
                painter.setFont(font)
            else:
                # Simpler text for unselected quadrants
                painter.setPen(QPen(Qt.GlobalColor.darkGray))
                painter.drawText(text_x - 25, text_y - 10, 50, 20, 
                            Qt.AlignmentFlag.AlignCenter, name.capitalize())
        
        # Now draw the baseline circle last so it's on top
        if self.active_quadrant == 'baseline':
            # For active baseline, use a bright color
            painter.setBrush(QBrush(self.quadrant_colors['baseline']))
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            
            # Draw a glowing effect around the baseline circle when active
            glow = QPainterPath()
            glow.addEllipse(center_x - 5, center_y - 5, center_size + 10, center_size + 10)
            painter.setPen(QPen(QColor(220, 50, 100, 80), 8))  # Semi-transparent pink
            painter.drawPath(glow)
            
            # Draw the actual baseline circle
            painter.setPen(QPen(Qt.GlobalColor.black, 2))
            painter.drawEllipse(center_x, center_y, center_size, center_size)
            
            # Draw baseline text with bold, white text for better visibility
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(center_x, center_y, center_size, center_size,
                        Qt.AlignmentFlag.AlignCenter, "Baseline")
            font.setBold(False)
            painter.setFont(font)
        else:
            # For inactive baseline
            painter.setBrush(QBrush(QColor(240, 240, 240, 100)))  # Transparent light gray
            painter.setPen(QPen(Qt.GlobalColor.gray, 1))
            painter.drawEllipse(center_x, center_y, center_size, center_size)
            
            # Draw baseline text
            painter.setPen(QPen(Qt.GlobalColor.darkGray))
            painter.drawText(center_x, center_y, center_size, center_size,
                        Qt.AlignmentFlag.AlignCenter, "Baseline")
                    
        # Reset any painter settings
        painter.setPen(QPen(Qt.GlobalColor.black)) 

# Result Dialog for showing inference results
class ResultDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inference Results")
        self.setFixedSize(500, 650)  # Increased height for the redesigned interface
        self.setModal(True)
        
        # Reference to the original CSV file path
        self.csv_path = None
        
        # Session counter (incremented with each feedback submission)
        self.session_counter = 0
        
        # Set up window without default frame
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Container frame with border radius and styling
        self.container = QFrame()
        self.container.setObjectName("container")
        self.container.setStyleSheet("""
            #container {
                background-color: #f5f5f5;
                border-radius: 10px;
                border: 1px solid #ddd;
            }
        """)
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # Add container to main layout
        main_layout.addWidget(self.container)
        
        # Header bar
        header_bar = QFrame()
        header_bar.setStyleSheet("background-color: #4f46e5; border-top-left-radius: 10px; border-top-right-radius: 10px;")
        header_bar.setFixedHeight(50)
        header_layout = QHBoxLayout(header_bar)
        
        # Add colored dots
        dots_layout = QHBoxLayout()
        dots_layout.setSpacing(6)
        
        for color in ["#ef4444", "#f59e0b", "#10b981"]:  # Red, Yellow, Green
            dot = QFrame()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
            dots_layout.addWidget(dot)
        
        # Add title
        title = QLabel("Inference Results")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Empty placeholder for spacing
        placeholder = QFrame()
        placeholder.setFixedWidth(dots_layout.sizeHint().width())
        
        header_layout.addLayout(dots_layout)
        header_layout.addWidget(title, 1)
        header_layout.addWidget(placeholder)
        
        # Add header to container
        container_layout.addWidget(header_bar)
        
        # Content area
        content_area = QWidget()
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)
        
        # Prediction section title
        prediction_title = QLabel("Prediction Results")
        prediction_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1f2937;")
        prediction_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(prediction_title)
        
        # Prediction card
        prediction_card = QFrame()
        prediction_card.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        """)
        prediction_card_layout = QVBoxLayout(prediction_card)
        
        # Prediction text - using a more visible color and stronger text
        self.prediction_label = QLabel("Prediction: No data")
        self.prediction_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #166534;
            padding: 10px;
        """)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prediction_card_layout.addWidget(self.prediction_label)
        
        # Confidence section
        confidence_text = QLabel("Confidence:")
        confidence_text.setStyleSheet("font-size: 14px; font-weight: medium; color: #374151; margin-top: 10px;")
        prediction_card_layout.addWidget(confidence_text)
        
        # Confidence bar container
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e5e7eb;
                border-radius: 5px;
                background-color: #f3f4f6;
                height: 20px;
                text-align: center;
                font-weight: bold;
                color: #000000;
            }
            QProgressBar::chunk {
                background-color: #10b981;
                border-radius: 5px;
            }
        """)
        prediction_card_layout.addWidget(self.confidence_bar)
        
        # Add prediction card to content
        content_layout.addWidget(prediction_card)
        
        # Classification details section
        details_card = QFrame()
        details_card.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        """)
        details_layout = QVBoxLayout(details_card)
        
        # Details title
        details_title = QLabel("Classification Details")
        details_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1f2937;")
        details_layout.addWidget(details_title)
        
        # Table for classification details
        table_frame = QFrame()
        table_frame.setStyleSheet("""
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background-color: white;
        """)
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)
        
        # Table header
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #f3f4f6; border-top-left-radius: 6px; border-top-right-radius: 6px;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 8, 10, 8)
        
        class_header = QLabel("Class")
        class_header.setStyleSheet("color: #374151; font-weight: bold; font-size: 13px;")
        
        prob_header = QLabel("Probability")
        prob_header.setStyleSheet("color: #374151; font-weight: bold; font-size: 13px; text-align: right;")
        prob_header.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        header_layout.addWidget(class_header, 1)
        header_layout.addWidget(prob_header)
        
        table_layout.addWidget(header_frame)
        
        # Table rows
        self.class_rows = {}
        
        for class_name, bg_color in [("Contains Peanut", "white"), ("No Peanut", "#f0fdf4")]:
            row_frame = QFrame()
            row_frame.setStyleSheet(f"background-color: {bg_color}; border-top: 1px solid #e5e7eb;")
            row_layout = QHBoxLayout(row_frame)
            row_layout.setContentsMargins(10, 12, 10, 12)
            
            class_label = QLabel(class_name)
            class_label.setStyleSheet("color: #111827; font-size: 14px; font-weight: medium;")
            
            prob_value = QLabel("0.0%")
            prob_value.setStyleSheet("color: #374151; font-size: 14px; text-align: right;")
            prob_value.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            row_layout.addWidget(class_label, 1)
            row_layout.addWidget(prob_value)
            
            self.class_rows[class_name] = prob_value
            table_layout.addWidget(row_frame)
        
        details_layout.addWidget(table_frame)
        
        # Add threshold information
        self.threshold_label = QLabel("Classification threshold: 0.5")
        self.threshold_label.setStyleSheet("color: #6b7280; font-style: italic; margin-top: 5px;")
        details_layout.addWidget(self.threshold_label)
        
        # Add details card to content
        content_layout.addWidget(details_card)
        
        # ---- FEEDBACK SECTION ----
        feedback_card = QFrame()
        feedback_card.setObjectName("feedbackCard")
        feedback_card.setStyleSheet("""
            #feedbackCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E5E7EB;
            }
        """)
        feedback_card.setFrameShape(QFrame.Shape.StyledPanel)
        feedback_card.setFrameShadow(QFrame.Shadow.Raised)
        
        feedback_layout = QVBoxLayout(feedback_card)
        
        feedback_title = QLabel("Provide Ground Truth Feedback")
        feedback_title.setStyleSheet("""
            font-size: 14pt;
            font-weight: bold;
            color: #111827;
            margin-bottom: 5px;
        """)
        feedback_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feedback_layout.addWidget(feedback_title)
        
        feedback_instructions = QLabel("Was peanut actually present in the sample?")
        feedback_instructions.setStyleSheet("""
            font-size: 11pt;
            color: #6B7280;
            margin-bottom: 15px;
        """)
        feedback_instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feedback_layout.addWidget(feedback_instructions)
        
        # Feedback buttons layout
        feedback_buttons = QHBoxLayout()
        feedback_buttons.setSpacing(15)
        
        # Yes Peanut button (styled in green)
        self.yes_button = QPushButton("Yes, Contains Peanut")
        self.yes_button.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                font-size: 12pt;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
                color: #9CA3AF;
            }
        """)
        self.yes_button.clicked.connect(lambda: self.save_feedback(True))
        feedback_buttons.addWidget(self.yes_button)
        
        # No Peanut button (styled in red)
        self.no_button = QPushButton("No Peanut")
        self.no_button.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                font-size: 12pt;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #B91C1C;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
                color: #9CA3AF;
            }
        """)
        self.no_button.clicked.connect(lambda: self.save_feedback(False))
        feedback_buttons.addWidget(self.no_button)
        
        feedback_layout.addLayout(feedback_buttons)
        
        # Status message for feedback
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4b5563; font-style: italic; margin-top: 10px; text-align: center;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setVisible(False)
        feedback_layout.addWidget(self.status_label)
        
        content_layout.addWidget(feedback_card)
        
        # ---- CLOSE BUTTON ----
        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #E5E7EB;
                color: #374151;
                font-size: 14pt;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #D1D5DB;
            }
            QPushButton:pressed {
                background-color: #9CA3AF;
            }
        """)
        self.close_button.clicked.connect(self.accept)
        content_layout.addWidget(self.close_button)
        
        # Add content area to container
        container_layout.addWidget(content_area)
        
        # Store results
        self.last_results = None
        
    def set_results(self, results, csv_path=None):
        """Set the results data in the dialog"""
        try:
            # Print results for debugging
            print(f"Setting results in dialog: {results}")
            
            self.last_results = results
            self.csv_path = csv_path
            
            # Reset status and enable buttons
            self.status_label.setText("")
            self.status_label.setVisible(False)
            self.yes_button.setEnabled(True)
            self.no_button.setEnabled(True)
            
            # Extract data with safe fallbacks
            prediction = results.get('prediction', 'Unknown')
            confidence = 0
            
            # Safely convert confidence to float
            try:
                confidence = float(results.get('confidence', 0))
            except (ValueError, TypeError):
                print(f"Error converting confidence value: {results.get('confidence')}")
                confidence = 0
                
            # Get probabilities dict with fallback
            probabilities = results.get('probabilities', {})
            if not isinstance(probabilities, dict):
                print(f"Warning: probabilities is not a dict: {probabilities}")
                probabilities = {}
            
            # Calculate the confidence as a percentage
            confidence_pct = int(confidence * 100)
            
            # Update prediction text with appropriate color - handle different formats
            if prediction == "no_peanut" or prediction == 0:
                self.prediction_label.setText("Prediction: No peanut detected")
                self.prediction_label.setStyleSheet("""
                    font-size: 18px;
                    font-weight: bold;
                    color: #166534;
                    padding: 10px;
                """)
                self.confidence_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #e5e7eb;
                        border-radius: 5px;
                        background-color: #f3f4f6;
                        height: 20px;
                        text-align: center;
                        font-weight: bold;
                        color: #000000;
                    }
                    QProgressBar::chunk {
                        background-color: #10b981;
                        border-radius: 5px;
                    }
                """)
            else:
                self.prediction_label.setText("Prediction: Contains peanut")
                self.prediction_label.setStyleSheet("""
                    font-size: 18px;
                    font-weight: bold;
                    color: #b91c1c;
                    padding: 10px;
                """)
                self.confidence_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #e5e7eb;
                        border-radius: 5px;
                        background-color: #f3f4f6;
                        height: 20px;
                        text-align: center;
                        font-weight: bold;
                        color: #000000;
                    }
                    QProgressBar::chunk {
                        background-color: #ef4444;
                        border-radius: 5px;
                    }
                """)
            
            # Update confidence bar
            self.confidence_bar.setValue(confidence_pct)
            
            # Handle both potential key formats for probabilities
            peanut_keys = ['peanut', 'contains_peanut', 1, '1']
            no_peanut_keys = ['no_peanut', 'no peanut', 0, '0']
            
            # Extract probability values with fallbacks
            peanut_prob = 0
            no_peanut_prob = 0
            
            # Try different keys for peanut probability
            for key in peanut_keys:
                if key in probabilities:
                    try:
                        peanut_prob = float(probabilities[key])
                        break
                    except (ValueError, TypeError):
                        pass
                        
            # Try different keys for no peanut probability
            for key in no_peanut_keys:
                if key in probabilities:
                    try:
                        no_peanut_prob = float(probabilities[key])
                        break
                    except (ValueError, TypeError):
                        pass
            
            # If we still didn't get values, calculate them from each other or use confidence
            if peanut_prob == 0 and no_peanut_prob > 0:
                peanut_prob = 1 - no_peanut_prob
            elif no_peanut_prob == 0 and peanut_prob > 0:
                no_peanut_prob = 1 - peanut_prob
            elif peanut_prob == 0 and no_peanut_prob == 0:
                # Use confidence as a fallback
                if prediction == "peanut" or prediction == 1:
                    peanut_prob = confidence
                    no_peanut_prob = 1 - confidence
                else:
                    no_peanut_prob = confidence
                    peanut_prob = 1 - confidence
            
            # Format as percentage with one decimal place
            self.class_rows["Contains Peanut"].setText(f"{peanut_prob * 100:.1f}%")
            self.class_rows["No Peanut"].setText(f"{no_peanut_prob * 100:.1f}%")
            
            # Update threshold (with safe fallback)
            threshold = 0.5
            try:
                threshold = float(results.get('threshold', 0.5))
            except (ValueError, TypeError):
                threshold = 0.5
                
            self.threshold_label.setText(f"Classification threshold: {threshold}")
            
            # Highlight the active row
            row_frames = self.findChildren(QFrame)
            for frame in row_frames:
                # Add safety check to make sure the frame has QLabel children before trying to access them
                labels = frame.findChildren(QLabel)
                if not labels:  # Skip this frame if it has no labels
                    continue
                    
                label_text = labels[0].text()
                if "No Peanut" in label_text and (prediction == "no_peanut" or prediction == 0):
                    frame.setStyleSheet("background-color: #f0fdf4; border-top: 1px solid #e5e7eb;")
                elif "Contains Peanut" in label_text and (prediction == "peanut" or prediction == 1):
                    frame.setStyleSheet("background-color: #fef2f2; border-top: 1px solid #e5e7eb;")
                else:
                    if "No Peanut" in label_text or "Contains Peanut" in label_text:
                        frame.setStyleSheet("background-color: white; border-top: 1px solid #e5e7eb;")
        except Exception as e:
            print(f"Error setting results: {e}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            
            # Show a basic message if we encounter an error
            self.prediction_label.setText("Error processing results")
            self.prediction_label.setStyleSheet("""
                font-size: 18px;
                font-weight: bold;
                color: #ef4444;
                padding: 10px;
            """)
    
    def save_feedback(self, contains_peanut):
        """Save feedback about whether the sample actually contained peanut"""
        if not self.csv_path:
            self.status_label.setText("Cannot save feedback: No original CSV file")
            self.status_label.setStyleSheet("color: #ef4444;")
            self.status_label.setVisible(True)
            return
            
        try:
            # Create directory if it doesn't exist
            feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inferenceresults")
            os.makedirs(feedback_dir, exist_ok=True)
            
            # Get the base filename and next trial number
            base_filename = os.path.basename(self.csv_path)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            
            # Increment the session counter
            self.session_counter += 1
            
            # Create new filename with YES/NO prefix
            prefix = "YES" if contains_peanut else "NO"
            feedback_file = f"{prefix}_trial{self.session_counter}_{timestamp}.csv"
            feedback_path = os.path.join(feedback_dir, feedback_file)
            
            # Copy the original CSV file to the new path
            import shutil
            shutil.copy2(self.csv_path, feedback_path)
            
            # Show success message
            feedback_type = "CONTAINS PEANUT" if contains_peanut else "NO PEANUT"
            self.status_label.setText(f"✓ Feedback saved as {feedback_type}")
            self.status_label.setStyleSheet("color: #10b981; font-weight: bold;")
            self.status_label.setVisible(True)
            
            # Disable buttons to prevent multiple submissions
            self.yes_button.setEnabled(False)
            self.no_button.setEnabled(False)
            
            print(f"Feedback saved to {feedback_path}")
            
        except Exception as e:
            self.status_label.setText(f"Error saving feedback: {str(e)}")
            self.status_label.setStyleSheet("color: #ef4444;")
            self.status_label.setVisible(True)
            print(f"Error saving feedback: {e}")
            
    def showEvent(self, event):
        # Center the dialog on the parent window
        if self.parent():
            parent_geo = self.parent().geometry()
            self.move(
                parent_geo.x() + (parent_geo.width() - self.width()) // 2,
                parent_geo.y() + (parent_geo.height() - self.height()) // 2
            )
        super().showEvent(event)

    def mousePressEvent(self, event):
        # Allow dragging the dialog by clicking on the header bar
        if event.button() == Qt.MouseButton.LeftButton:
            self.offset = event.position()
            event.accept()
            
    def mouseMoveEvent(self, event):
        # Handle dragging the dialog
        if hasattr(self, 'offset') and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(self.pos() + event.position().toPoint() - self.offset.toPoint())
            event.accept()

# New SensorPlotDialog class for displaying baseline-corrected sensor data
class SensorPlotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sensor Visualization")
        self.setFixedSize(900, 700)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Baseline Corrected Sensor Data")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create plot widgets
        self.n_plot = pg.PlotWidget()
        self.n_plot.setTitle("Raw Voltage Sensors (n1-n8) - Baseline Corrected")
        self.n_plot.setLabel('left', 'Voltage Change (V)')
        self.n_plot.setLabel('bottom', 'Time (s)')
        self.n_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.g_plot = pg.PlotWidget()
        self.g_plot.setTitle("Resistance Sensors (g1-g2) - Percent Change from Baseline")
        self.g_plot.setLabel('left', 'Percent Change (%)')
        self.g_plot.setLabel('bottom', 'Time (s)')
        self.g_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Create a widget for environmental sensors
        env_widget = QWidget()
        env_layout = QHBoxLayout(env_widget)
        
        self.tmp_plot = pg.PlotWidget()
        self.tmp_plot.setTitle("Temperature")
        self.tmp_plot.setLabel('left', 'Temperature (°C)')
        self.tmp_plot.setLabel('bottom', 'Time (s)')
        self.tmp_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.pa_plot = pg.PlotWidget()
        self.pa_plot.setTitle("Pressure")
        self.pa_plot.setLabel('left', 'Pressure (Pa)')
        self.pa_plot.setLabel('bottom', 'Time (s)')
        self.pa_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.hum_plot = pg.PlotWidget()
        self.hum_plot.setTitle("Humidity")
        self.hum_plot.setLabel('left', 'Humidity (%)')
        self.hum_plot.setLabel('bottom', 'Time (s)')
        self.hum_plot.showGrid(x=True, y=True, alpha=0.3)
        
        env_layout.addWidget(self.tmp_plot)
        env_layout.addWidget(self.pa_plot)
        env_layout.addWidget(self.hum_plot)
        
        # Add plots to layout with appropriate sizing
        main_layout.addWidget(self.n_plot, stretch=35)
        main_layout.addWidget(self.g_plot, stretch=35)
        main_layout.addWidget(env_widget, stretch=30)
        
        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #E5E7EB;
                color: #374151;
                font-size: 12pt;
                font-weight: bold;
                padding: 10px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #D1D5DB;
            }
            QPushButton:pressed {
                background-color: #9CA3AF;
            }
        """)
        self.close_button.clicked.connect(self.accept)
        main_layout.addWidget(self.close_button)
        
        # Store references to data and plots
        self.n_curves = {}
        self.g_curves = {}
        self.env_curves = {}
        
        # Colors for different sensors
        self.n_colors = ['r', 'g', 'b', 'c', 'm', 'y', (255, 165, 0), (128, 0, 128)]
        self.g_colors = ['r', 'g']
        self.env_colors = {'t': 'r', 'p': 'b', 'h': 'g'}
        
        # Initialize empty curves
        self.initialize_curves()
        
        # Add zero reference lines
        self.n_plot.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.PenStyle.DashLine))
        self.g_plot.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.PenStyle.DashLine))
        
        # Add legends
        self.n_legend = self.n_plot.addLegend()
        self.g_legend = self.g_plot.addLegend()
    
    def initialize_curves(self):
        """Initialize all curves for the plots"""
        # Initialize n1-n8 curves
        for i in range(1, 9):
            sensor_name = f'n{i}'
            self.n_curves[sensor_name] = self.n_plot.plot(
                [], [], name=sensor_name, pen=pg.mkPen(self.n_colors[i-1], width=2))
        
        # Initialize g1-g2 curves
        for i in range(1, 3):
            sensor_name = f'g{i}'
            self.g_curves[sensor_name] = self.g_plot.plot(
                [], [], name=sensor_name, pen=pg.mkPen(self.g_colors[i-1], width=2))
        
        # Initialize environmental curves
        self.env_curves['t'] = self.tmp_plot.plot(
            [], [], name='Temperature', pen=pg.mkPen('r', width=2))
        self.env_curves['p'] = self.pa_plot.plot(
            [], [], name='Pressure', pen=pg.mkPen('b', width=2))
        self.env_curves['h'] = self.hum_plot.plot(
            [], [], name='Humidity', pen=pg.mkPen('g', width=2))
    
    def load_and_plot_data(self, csv_path):
        """Load data from CSV and create baseline-corrected plots"""
        try:
            # Load data from CSV
            data = pd.read_csv(csv_path)
            
            # Identify baseline period
            baseline_mask = data['Quadrant'] == 'baseline'
            
            if not baseline_mask.any():
                print("Warning: No baseline data found in the recording")
                return
            
            # Time values
            time_values = data['Time'].values
            
            # Process n1-n8 sensors with baseline correction
            n_filtered_data_all = []
            for i in range(1, 9):
                sensor_name = f'n{i}'
                if sensor_name in data.columns:
                    # Get sensor data
                    sensor_data = data[sensor_name].values
                    
                    # Calculate baseline mean
                    baseline_mean = sensor_data[baseline_mask].mean()
                    
                    # Apply baseline correction
                    corrected_data = sensor_data - baseline_mean
                    
                    # Apply low-pass filter to smooth the data
                    # First design a butterworth filter
                    b, a = signal.butter(4, 0.25, 'low')
                    filtered_data = signal.filtfilt(b, a, corrected_data)
                    
                    # Store for axis scaling
                    n_filtered_data_all.append(filtered_data)
                    
                    # Plot the data
                    self.n_curves[sensor_name].setData(time_values, filtered_data)
            
            # Process g1-g2 sensors with percent change
            g_filtered_data_all = []
            for i in range(1, 3):
                sensor_name = f'g{i}'
                if sensor_name in data.columns:
                    # Get sensor data
                    sensor_data = data[sensor_name].values
                    
                    # Calculate baseline mean
                    baseline_mean = sensor_data[baseline_mask].mean()
                    
                    # Calculate percent change from baseline
                    if baseline_mean != 0:  # Avoid division by zero
                        percent_change = ((sensor_data - baseline_mean) / baseline_mean) * 100
                    else:
                        percent_change = sensor_data - baseline_mean
                    
                    # Apply simple moving average filter for light smoothing
                    window_size = 3
                    filtered_data = np.convolve(percent_change, np.ones(window_size)/window_size, mode='valid')
                    
                    # Match filtered data length to time_values by padding
                    pad_size = len(time_values) - len(filtered_data)
                    padded_data = np.pad(filtered_data, (pad_size//2, pad_size - pad_size//2), 'edge')
                    
                    # Store for axis scaling
                    g_filtered_data_all.append(padded_data)
                    
                    # Plot the data
                    self.g_curves[sensor_name].setData(time_values, padded_data)
            
            # Process environmental sensors
            for sensor_name in ['t', 'p', 'h']:
                if sensor_name in data.columns:
                    # Get sensor data
                    sensor_data = data[sensor_name].values
                    
                    # Apply filtering
                    b, a = signal.butter(4, 0.25, 'low')
                    filtered_data = signal.filtfilt(b, a, sensor_data)
                    
                    # Plot the data
                    self.env_curves[sensor_name].setData(time_values, filtered_data)
            
            # Add quadrant shading regions
            self.add_quadrant_regions(data)
            
            # Set appropriate Y-axis scaling for n sensors
            if n_filtered_data_all:
                all_values = np.concatenate(n_filtered_data_all)
                y_min = np.min(all_values)
                y_max = np.max(all_values)
                
                # Add 10% padding
                padding = (y_max - y_min) * 0.1
                self.n_plot.setYRange(y_min - padding, y_max + padding)
            
            # Set appropriate Y-axis scaling for g sensors
            if g_filtered_data_all:
                all_values = np.concatenate(g_filtered_data_all)
                y_min = np.min(all_values)
                y_max = np.max(all_values)
                
                # Add 10% padding
                padding = (y_max - y_min) * 0.1
                padding = max(padding, 2.0)  # Ensure at least 2% padding
                
                # Ensure we show at least some range
                if y_max - y_min < 10.0:
                    mid_point = (y_min + y_max) / 2
                    y_min = mid_point - 5.0
                    y_max = mid_point + 5.0
                
                self.g_plot.setYRange(y_min - padding, y_max + padding)
            
            # Set X range for all plots
            x_min = np.min(time_values)
            x_max = np.max(time_values)
            padding = (x_max - x_min) * 0.02
            
            self.n_plot.setXRange(x_min - padding, x_max + padding)
            self.g_plot.setXRange(x_min - padding, x_max + padding)
            self.tmp_plot.setXRange(x_min - padding, x_max + padding)
            self.pa_plot.setXRange(x_min - padding, x_max + padding)
            self.hum_plot.setXRange(x_min - padding, x_max + padding)
            
        except Exception as e:
            print(f"Error loading and plotting data: {str(e)}")
            traceback.print_exc()
    
    def add_quadrant_regions(self, data):
        """Add shaded regions for each quadrant"""
        # Define colors for each quadrant
        quadrant_colors = {
            'baseline': (0, 0, 255, 50),    # Blue
            'topright': (0, 255, 0, 50),    # Green
            'bottomright': (255, 0, 0, 50), # Red
            'bottomleft': (255, 165, 0, 50),# Orange
            'topleft': (128, 0, 128, 50)    # Purple
        }
        
        # Get time values
        time_values = data['Time'].values
        
        # Find segments for each quadrant
        quadrants = data['Quadrant'].unique()
        
        for quadrant in quadrants:
            if quadrant not in quadrant_colors:
                continue
                
            # Find all segments where this quadrant is active
            in_segment = False
            segments = []
            current_start = None
            
            for i, q in enumerate(data['Quadrant']):
                if q == quadrant and not in_segment:
                    in_segment = True
                    current_start = time_values[i]
                elif q != quadrant and in_segment:
                    in_segment = False
                    segments.append((current_start, time_values[i-1]))
            
            # If still in segment at the end
            if in_segment:
                segments.append((current_start, time_values[-1]))
            
            # Add shaded regions to all plots
            color = quadrant_colors[quadrant]
            for start, end in segments:
                # Add regions to each plot
                for plot in [self.n_plot, self.g_plot, self.tmp_plot, self.pa_plot, self.hum_plot]:
                    region = pg.LinearRegionItem([start, end], movable=False, brush=pg.mkBrush(color))
                    region.setZValue(-10)  # Ensure regions are behind the data
                    plot.addItem(region)
                
                # Add text label for the quadrant (only to n_plot)
                mid_point = (start + end) / 2
                text = pg.TextItem(quadrant, anchor=(0.5, 0.5), color='k')
                text.setPos(mid_point, self.n_plot.getViewBox().viewRange()[1][1] * 0.9)
                self.n_plot.addItem(text)
    
    def showEvent(self, event):
        # Center the dialog on the parent window
        if self.parent():
            parent_geo = self.parent().geometry()
            self.move(
                parent_geo.x() + parent_geo.width(), 
                parent_geo.y()
            )
        super().showEvent(event)

# Main window class
class MainWindow(QMainWindow):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.setWindowTitle("sense: inference")
        self.setGeometry(100, 100, 1200, 700)
        
        self.setWindowIcon(QIcon('../icon.png'))
        self.setMinimumSize(1000, 500)
        
        # Recording variables
        self.is_recording = False
        self.current_recording = None
        self.last_csv_path = None  # Store the path to the last saved CSV
        self.recording_start_time = None
        self.recording_data = None
        self.countdown_timer = None
        self.quadrant_timer = None
        self.countdown_remaining = 0
        self.quadrant_dialog = None
        self.current_quadrant = None
        self.quadrant_sequence = ['baseline', 'topright', 'bottomright', 'bottomleft', 'topleft']
        self.current_quadrant_index = -1  # -1 means not yet started
        
        # Baseline data for deviation calculation - track each sensor separately
        self.baseline_n_values = {f'n{i}': [] for i in range(1, 9)}  # Dictionary to store values for each n sensor
        self.baseline_start_time = None  # When baseline recording started
        
        # Create the quadrant dialog (initialized when needed)
        self.quadrant_dialog = QuadrantDialog(self)
        
        # Create result dialog for showing inference results
        self.result_dialog = ResultDialog(self)
        
        # Main layout with left side for plots and right side for controls
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left side for plots
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Status bar at the top of left side
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.sensor_status_label = QLabel("Sense32: Disconnected")
        self.sensor_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sensor_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
        
        # Add battery information labels
        battery_widget = QWidget()
        battery_layout = QHBoxLayout(battery_widget)
        battery_layout.setContentsMargins(0, 0, 0, 0)
        
        self.battery_current_label = QLabel("Current: -- mA")
        self.battery_current_label.setStyleSheet("font-weight: bold; padding: 5px;")
        
        self.battery_percent_label = QLabel("Battery: --%")
        self.battery_percent_label.setStyleSheet("font-weight: bold; padding: 5px;")
        
        battery_layout.addWidget(self.battery_percent_label)
        battery_layout.addWidget(self.battery_current_label)
        
        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.clicked.connect(self.reconnect_sensor)
        
        status_layout.addWidget(self.sensor_status_label, 3)
        status_layout.addWidget(battery_widget, 2)
        status_layout.addWidget(self.reconnect_button, 1)
        
        left_layout.addWidget(status_widget)

        # Plot for n1-n8 (raw values)
        self.n_plot = pg.PlotWidget()
        self.n_plot.showGrid(x=True, y=True)
        self.n_plot.setLabel('left', 'Voltage (V)')
        self.n_plot.setLabel('bottom', 'Time (s)')
        self.n_plot.setTitle("n1-n8 Raw Values")
        left_layout.addWidget(self.n_plot, stretch=3)
        
        # Plot for g1-g2
        self.g_plot = pg.PlotWidget()
        self.g_plot.showGrid(x=True, y=True)
        self.g_plot.setLabel('left', 'Value')
        self.g_plot.setLabel('bottom', 'Time (s)')
        self.g_plot.setTitle("g1-g2")
        left_layout.addWidget(self.g_plot, stretch=2)
        
        # Environmental plots row
        env_widget = QWidget()
        env_layout = QHBoxLayout(env_widget)
        
        self.tmp_plot = pg.PlotWidget()
        self.tmp_plot.showGrid(x=True, y=True)
        self.tmp_plot.setLabel('left', 'tmp')
        self.tmp_plot.setLabel('bottom', 'Time (s)')
        self.tmp_plot.setTitle("Temperature")
        env_layout.addWidget(self.tmp_plot)
        
        self.pa_plot = pg.PlotWidget()
        self.pa_plot.showGrid(x=True, y=True)
        self.pa_plot.setLabel('left', 'pa')
        self.pa_plot.setLabel('bottom', 'Time (s)')
        self.pa_plot.setTitle("Pressure")
        env_layout.addWidget(self.pa_plot)
        
        self.hum_plot = pg.PlotWidget()
        self.hum_plot.showGrid(x=True, y=True)
        self.hum_plot.setLabel('left', 'hum')
        self.hum_plot.setLabel('bottom', 'Time (s)')
        self.hum_plot.setTitle("Humidity")
        env_layout.addWidget(self.hum_plot)
        
        # Give environmental plots more vertical space
        left_layout.addWidget(env_widget, stretch=2)
        
        # Store plot widgets in list for easy access
        self.plot_widgets = [
            self.n_plot,
            self.g_plot,
            self.tmp_plot,
            self.pa_plot,
            self.hum_plot
        ]
        
        # Initialize curves with different colors
        self.curves = {}
        n_colors = ['r', 'g', 'b', 'c', 'm', 'y', (255, 165, 0), (128, 0, 128)]
        g_colors = ['r', 'g']
        
        # Add curves for n1-n8
        n_legend = self.n_plot.addLegend()
        for i in range(1, 9):
            sensor = f'n{i}'
            curve = self.n_plot.plot(pen=n_colors[i-1], name=sensor)
            self.curves[sensor] = curve
        
        # Add curves for g1-g2
        g_legend = self.g_plot.addLegend()
        # Add temperature information to g1-g2 names
        g_temps = {1: 250, 2: 350}  # Temperature mapping for g sensors
        for i in range(1, 3):
            sensor = f'g{i}'
            curve = self.g_plot.plot(pen=g_colors[i-1], name=f"{sensor} ({g_temps[i]}°C)")
            self.curves[sensor] = curve
        
        # Add curves for environmental sensors with better colored pens
        self.curves['t'] = self.tmp_plot.plot(pen=(255, 0, 0), name='Temperature')  # Red
        self.curves['p'] = self.pa_plot.plot(pen=(0, 0, 255), name='Pressure')      # Blue
        self.curves['h'] = self.hum_plot.plot(pen=(0, 255, 0), name='Humidity')     # Green
        
        # Initialize data storage for plotting
        self.plot_data = {
            sensor: {'time': deque(maxlen=MAX_DATA_POINTS), 
                    'value': deque(maxlen=MAX_DATA_POINTS)} 
            for sensor in self.curves.keys()
        }
        
        # Add left widget to main layout
        main_layout.addWidget(left_widget, 7)  # Left side takes 70% of width
        
        # Right side for inference controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title for right panel
        right_title = QLabel("Inference Controls")
        right_title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(right_title)
        
        # Recording status label
        self.recording_status_label = QLabel("Not Recording")
        self.recording_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_status_label.setStyleSheet("background-color: transparent; color: black; font-weight: bold; padding: 5px;")
        right_layout.addWidget(self.recording_status_label)
        
        # Infer button widget
        infer_widget = QWidget()
        infer_layout = QVBoxLayout(infer_widget)
        infer_layout.setContentsMargins(20, 20, 20, 20)
        
        # Large, prominent infer button
        self.infer_button = QPushButton("Start Inference")
        self.infer_button.setStyleSheet("""
            QPushButton {
                background-color: #006699; 
                color: white; 
                font-size: 16pt; 
                font-weight: bold; 
                padding: 15px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #0088cc;
            }
            QPushButton:pressed {
                background-color: #005588;
            }
        """)
        self.infer_button.setMinimumHeight(60)
        self.infer_button.clicked.connect(self.start_inference)
        infer_layout.addWidget(self.infer_button)
        
        # Instructions label
        infer_instructions = QLabel("Click 'Start Inference' to begin recording and analyze the sample")
        infer_instructions.setStyleSheet("font-style: italic;")
        infer_instructions.setWordWrap(True)
        infer_instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        infer_layout.addWidget(infer_instructions)
        
        # Add the infer widget to the right layout
        right_layout.addWidget(infer_widget)
        
        # Section for most recent result
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.Shape.StyledPanel)
        result_frame.setFrameShadow(QFrame.Shadow.Raised)
        result_layout = QVBoxLayout(result_frame)
        
        result_title = QLabel("Last Result")
        result_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(result_title)
        
        self.last_result_label = QLabel("No inference performed yet")
        self.last_result_label.setWordWrap(True)
        self.last_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.last_result_label.setStyleSheet("font-size: 12pt;")
        result_layout.addWidget(self.last_result_label)
        
        self.last_confidence_label = QLabel("")
        self.last_confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.last_confidence_label)
        
        # Button to show detailed results again
        self.show_details_button = QPushButton("Show Details")
        self.show_details_button.setEnabled(False)
        self.show_details_button.clicked.connect(self.show_last_result)
        result_layout.addWidget(self.show_details_button)
        
        right_layout.addWidget(result_frame)
        
        # Add a stretch to push everything to the top
        right_layout.addStretch()
        
        # Add right widget to main layout
        main_layout.addWidget(right_widget, 3)  # Right side takes 30% of width
        
        # Initialize BLE worker
        self.sensor_worker = BLEWorker(MULTI_SENSOR_SERVICE_UUID, MULTI_SENSOR_CHARACTERISTIC_UUID, MULTI_SENSOR_DEVICE_NAME)
        self.sensor_worker.data.connect(self.update_plot)
        self.sensor_worker.connection_status.connect(self.update_sensor_status)

        self.start_time = None
        self.last_inference_result = None

        self.setup_additional_ui()
        
    def setup_additional_ui(self):
        # Create sensor plot dialog (initialized when needed)
        self.sensor_plot_dialog = SensorPlotDialog(self)

    def reconnect_sensor(self):
        self.sensor_worker.stop()
        self.sensor_worker = BLEWorker(MULTI_SENSOR_SERVICE_UUID, MULTI_SENSOR_CHARACTERISTIC_UUID, MULTI_SENSOR_DEVICE_NAME)
        self.sensor_worker.data.connect(self.update_plot)
        self.sensor_worker.connection_status.connect(self.update_sensor_status)
        asyncio.create_task(self.sensor_worker.run())

    def start_recording(self):
        # Implementation of start_recording method
        pass

    def stop_recording(self):
        # Implementation of stop_recording method
        pass

    def update_sensor_status(self, status):
        if status == "connected":
            self.sensor_status_label.setText("Sense32: Connected")
            self.sensor_status_label.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 5px;")
        elif status == "disconnected":
            self.sensor_status_label.setText("Sense32: Disconnected")
            self.sensor_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
            # Reset battery information
            self.battery_percent_label.setText("Battery: --%")
            self.battery_current_label.setText("Current: -- mA")
            self.battery_percent_label.setStyleSheet("font-weight: bold; padding: 5px;")
            self.battery_current_label.setStyleSheet("font-weight: bold; padding: 5px;")
        elif status == "connecting":
            self.sensor_status_label.setText("Sense32: Connecting...")
            self.sensor_status_label.setStyleSheet("background-color: orange; color: white; font-weight: bold; padding: 5px;")
        elif status == "searching":
            self.sensor_status_label.setText("Sense32: Searching...")
            self.sensor_status_label.setStyleSheet("background-color: blue; color: white; font-weight: bold; padding: 5px;")

    def update_battery_status(self, current, percent):
        # Implementation of update_battery_status method
        pass

    def update_plot_data(self, sensor, time, value):
        # Implementation of update_plot_data method
        pass

    def update_inference_results(self, results):
        # Implementation of update_inference_results method
        pass

    def update_ui(self):
        # Implementation of update_ui method
        pass

    def showEvent(self, event):
        # Implementation of showEvent method
        super().showEvent(event)

    def closeEvent(self, event):
        # Implementation of closeEvent method
        super().closeEvent(event)

    async def start_ble_worker(self):
        await self.sensor_worker.run()

    def update_plot(self, data):
        if self.start_time is None:
            self.start_time = data.get('x', 0) / 1000
        
        current_time = data.get('x', 0) / 1000 - self.start_time

        # If recording, immediately sample this data point
        if self.is_recording and self.current_recording:
            self.process_data_point(data, current_time)

        # Update battery information if available
        if 'b_p' in data:
            try:
                battery_percent = float(data['b_p'])
                self.battery_percent_label.setText(f"Battery: {battery_percent:.1f}%")
                
                # Update color based on battery level
                if battery_percent > 50:
                    self.battery_percent_label.setStyleSheet("font-weight: bold; padding: 5px; color: green;")
                elif battery_percent > 20:
                    self.battery_percent_label.setStyleSheet("font-weight: bold; padding: 5px; color: orange;")
                else:
                    self.battery_percent_label.setStyleSheet("font-weight: bold; padding: 5px; color: red;")
            except (ValueError, TypeError) as e:
                print(f"Error converting battery percentage: {e}")
        
        if 'b_i' in data:
            try:
                battery_current = float(data['b_i'])
                self.battery_current_label.setText(f"Current: {battery_current:.1f} mA")
            except (ValueError, TypeError) as e:
                print(f"Error converting battery current: {e}")

        # Process n1-n8 sensors
        for i in range(1, 9):
            sensor_name = f'n{i}'
            if sensor_name in data:
                try:
                    value = float(data[sensor_name])
                    self.plot_data[sensor_name]['time'].append(current_time)
                    self.plot_data[sensor_name]['value'].append(value)
                except (ValueError, TypeError) as e:
                    print(f"Error converting {sensor_name} value: {data[sensor_name]}, Error: {e}")
                    continue
        
        # Process g1-g2 sensors
        for i in range(1, 3):
            sensor_name = f'g{i}'
            if sensor_name in data:
                try:
                    value = float(data[sensor_name])
                    self.plot_data[sensor_name]['time'].append(current_time)
                    self.plot_data[sensor_name]['value'].append(value)
                except (ValueError, TypeError) as e:
                    print(f"Error converting {sensor_name} value: {data[sensor_name]}, Error: {e}")
                    continue
        
        # Process environmental sensors
        for sensor_name in ['t', 'p', 'h']:
            if sensor_name in data:
                try:
                    value = float(data[sensor_name])
                    self.plot_data[sensor_name]['time'].append(current_time)
                    self.plot_data[sensor_name]['value'].append(value)
                except (ValueError, TypeError) as e:
                    print(f"Error converting {sensor_name} value: {data[sensor_name]}, Error: {e}")
                    continue

        # Update all curves
        try:
            x_min = max(0, current_time - TIME_WINDOW)
            x_max = max(TIME_WINDOW, current_time)
            
            # Update n1-n8 curves
            n_values = []
            for i in range(1, 9):
                sensor_name = f'n{i}'
                if len(self.plot_data[sensor_name]['time']) > 0:
                    x_data = list(self.plot_data[sensor_name]['time'])
                    y_data = list(self.plot_data[sensor_name]['value'])
                    
                    if len(x_data) == len(y_data):
                        self.curves[sensor_name].setData(x_data, y_data)
                        n_values.extend(y_data)
                    else:
                        print(f"Data length mismatch for {sensor_name}: x={len(x_data)}, y={len(y_data)}")
            
            # Update g1-g2 curves
            g_values = []
            for i in range(1, 3):
                sensor_name = f'g{i}'
                if len(self.plot_data[sensor_name]['time']) > 0:
                    x_data = list(self.plot_data[sensor_name]['time'])
                    y_data = list(self.plot_data[sensor_name]['value'])
                    
                    if len(x_data) == len(y_data):
                        self.curves[sensor_name].setData(x_data, y_data)
                        g_values.extend(y_data)
                    else:
                        print(f"Data length mismatch for {sensor_name}: x={len(x_data)}, y={len(y_data)}")
            
            # Update environmental sensor curves
            for sensor_name in ['t', 'p', 'h']:
                if len(self.plot_data[sensor_name]['time']) > 0:
                    x_data = list(self.plot_data[sensor_name]['time'])
                    y_data = list(self.plot_data[sensor_name]['value'])
                    
                    if len(x_data) == len(y_data):
                        self.curves[sensor_name].setData(x_data, y_data)
                    else:
                        print(f"Data length mismatch for {sensor_name}: x={len(x_data)}, y={len(y_data)}")
            
            # Update x-range for all plots
            for plot_widget in self.plot_widgets:
                plot_widget.setXRange(x_min, x_max)
            
            # Update y-range for n plot (n1-n8)
            if n_values:
                y_min = min(v for v in n_values if v > 0)  # Ignore zero values
                y_max = max(n_values)
                padding = (y_max - y_min) * 0.1
                self.n_plot.setYRange(max(0, y_min - padding), y_max + padding)
            
            # Update y-range for g plot (g1-g2)
            if g_values:
                y_min = min(g_values)
                y_max = max(g_values)
                padding = (y_max - y_min) * 0.1 if y_max != y_min else 1
                self.g_plot.setYRange(y_min - padding, y_max + padding)
            
            # Dynamic y-axis for environmental sensors
            for sensor_name, plot_widget in [('t', self.tmp_plot), ('p', self.pa_plot), ('h', self.hum_plot)]:
                if self.plot_data[sensor_name]['value']:
                    y_min = min(self.plot_data[sensor_name]['value'])
                    y_max = max(self.plot_data[sensor_name]['value'])
                    padding = (y_max - y_min) * 0.1 if y_max != y_min else 1
                    plot_widget.setYRange(y_min - padding, y_max + padding)
                    
        except Exception as e:
            print(f"Error updating plots: {e}")

    def start_inference(self):
        """Start the inference process by recording data from all quadrants"""
        if self.is_recording:
            print("Already recording! Please wait for the current recording to finish.")
            return
            
        # Make sure the sensor is connected
        if not self.sensor_worker or self.sensor_status_label.text() == "Sense32: Disconnected":
            QMessageBox.warning(self, "Sensor Not Connected", 
                                "Cannot start inference because the sensor is not connected.\n\n"
                                "Please connect the sensor and try again.")
            return
            
        # Update UI to show recording
        self.recording_status_label.setText("Preparing to record...")
        self.recording_status_label.setStyleSheet("background-color: orange; color: white; font-weight: bold; padding: 5px;")
        
        # Disable the infer button during recording
        self.infer_button.setEnabled(False)
        
        # Start the initial countdown (3 seconds before baseline recording)
        self.countdown_remaining = COUNTDOWN_SECONDS
        self.current_quadrant_index = -1  # Not yet started
        
        # Initialize the quadrant dialog
        self.quadrant_dialog.countdown_label.setText(f"Starting in: {self.countdown_remaining}")
        self.quadrant_dialog.set_active_quadrant(None)  # No active quadrant yet
        self.quadrant_dialog.show()
        
        # Initialize countdown timer if not already
        if not self.countdown_timer:
            self.countdown_timer = QTimer()
            self.countdown_timer.timeout.connect(self.update_countdown)
        
        # Start the countdown timer
        self.countdown_timer.start(1000)  # 1 second intervals

    def update_countdown(self):
        """Update the countdown timer"""
        self.countdown_remaining -= 1
        
        # Update the countdown display
        if self.current_quadrant_index == -1:
            # Initial countdown
            self.quadrant_dialog.countdown_label.setText(f"Starting in: {self.countdown_remaining}s")
        else:
            # Quadrant timer
            self.quadrant_dialog.countdown_label.setText(f"Time remaining: {self.countdown_remaining}s")
            
            # Force update of the UI
            QApplication.processEvents()
        
        if self.countdown_remaining <= 0:
            # Countdown finished
            self.countdown_timer.stop()
            
            # If we haven't started recording yet, begin baseline
            if self.current_quadrant_index == -1:
                self.start_baseline_recording()
            else:
                # If we're already recording, this is the end of a quadrant
                self.advance_to_next_quadrant()
                
    def start_baseline_recording(self):
        """Start the baseline recording"""
        self.is_recording = True
        self.recording_start_time = datetime.now()
        
        # Set the current quadrant to baseline
        self.current_quadrant_index = 0
        self.current_quadrant = self.quadrant_sequence[self.current_quadrant_index]
        
        # Reset baseline tracking variables
        self.baseline_n_values = {f'n{i}': [] for i in range(1, 9)}
        self.baseline_start_time = None
        
        # Update UI
        self.quadrant_dialog.set_active_quadrant('baseline')
        self.recording_status_label.setText("Recording Baseline")
        self.recording_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
        
        # Initialize recording data - create folder structure
        timestamp_str = self.recording_start_time.strftime('%m-%d-%Y-%I%M%p').lower()
        folder_name = "inference_data"
        os.makedirs(folder_name, exist_ok=True)
        
        # Create filename 
        file_name = f"inference_data_{timestamp_str}.csv"
        self.current_recording = os.path.join(folder_name, file_name)
        
        # Get the current plot time for synchronizing
        start_plot_time = 0
        for sensor_name in ['n1', 'g1', 't', 'p', 'h']:
            if self.plot_data[sensor_name]['time']:
                start_plot_time = self.plot_data[sensor_name]['time'][-1]
                break
        
        # Initialize recording data structure for all sensors
        self.recording_data = {
            'time': {},
            'phase': {},
            'quadrant': {}
        }
        
        # Add entries for all sensor types
        for sensor_type in ['n', 'g', 't', 'p', 'h']:
            if sensor_type in ['n', 'g']:
                # Multiple numbered sensors
                count = 8 if sensor_type == 'n' else 2
                for i in range(1, count + 1):
                    sensor_name = f"{sensor_type}{i}"
                    self.recording_data[sensor_name] = {}
            else:
                # Single environmental sensors
                self.recording_data[sensor_type] = {}
        
        # Store the actual plot time when recording started
        self.recording_real_start_time = start_plot_time
        
        # Track the last processed raw timestamp to avoid duplicates
        self.last_processed_raw_timestamp = None
        
        # Start the sampling timer
        self.tick_counter = 0
        self.sampling_timer = QTimer()
        self.sampling_timer.timeout.connect(self.sample_data)
        self.sampling_timer.start(50)  # Use a faster timer to check for new data more frequently
        
        # Start the quadrant timer
        self.countdown_remaining = BASELINE_SECONDS
        self.quadrant_dialog.countdown_label.setText(f"Time remaining: {self.countdown_remaining}s")
        self.countdown_timer.start(1000)
        
    def advance_to_next_quadrant(self):
        """Advance to the next quadrant in the sequence"""
        # Move to next quadrant
        self.current_quadrant_index += 1
        
        # Check if we've completed all quadrants
        if self.current_quadrant_index >= len(self.quadrant_sequence):
            self.stop_recording_and_infer()
            return
            
        # Set the current quadrant
        self.current_quadrant = self.quadrant_sequence[self.current_quadrant_index]
        
        # Update UI
        self.quadrant_dialog.set_active_quadrant(self.current_quadrant)
        
        # Start the quadrant timer
        self.countdown_remaining = QUADRANT_SECONDS
        self.quadrant_dialog.countdown_label.setText(f"Time remaining: {self.countdown_remaining}s")
        self.countdown_timer.start(1000)
        
    def process_data_point(self, data, current_time):
        """Process a new data point for recording"""
        # Skip if we've already processed this exact timestamp
        raw_timestamp = data.get('x', 0)
        if self.last_processed_raw_timestamp == raw_timestamp:
            return
            
        # Record all sensor values for this data point
        for sensor_type in ['n', 'g', 't', 'p', 'h']:
            if sensor_type in ['n', 'g']:
                # Multiple numbered sensors
                count = 8 if sensor_type == 'n' else 2
                for i in range(1, count + 1):
                    sensor_name = f"{sensor_type}{i}"
                    if sensor_name in data:
                        try:
                            value = float(data[sensor_name])
                            self.recording_data[sensor_name][self.tick_counter] = value
                        except (ValueError, TypeError):
                            pass
            else:
                # Single environmental sensors
                if sensor_type in data:
                    try:
                        value = float(data[sensor_type])
                        self.recording_data[sensor_type][self.tick_counter] = value
                    except (ValueError, TypeError):
                        pass
        
        # Record time and phase for this tick
        time_since_start = current_time - self.recording_real_start_time
        self.recording_data['time'][self.tick_counter] = time_since_start
        self.recording_data['phase'][self.tick_counter] = self.current_quadrant_index
        self.recording_data['quadrant'][self.tick_counter] = self.current_quadrant
        
        # In baseline phase, collect n sensor values for deviation check
        if self.current_quadrant == 'baseline':
            # Initialize baseline_start_time if not set
            if self.baseline_start_time is None:
                self.baseline_start_time = time_since_start
            
            # Collect n values for each sensor individually
            for i in range(1, 9):
                sensor_name = f"n{i}"
                if sensor_name in data:
                    try:
                        value = float(data[sensor_name])
                        self.baseline_n_values[sensor_name].append(value)
                    except (ValueError, TypeError):
                        pass
            
            # Check deviation at 5s mark (using data collected so far)
            if time_since_start >= 5.0 and time_since_start < 5.1:
                # Calculate max deviation across all n sensors
                max_deviation = 0.0
                worst_sensor = ""
                
                for i in range(1, 9):
                    sensor_name = f"n{i}"
                    if len(self.baseline_n_values[sensor_name]) > 1:
                        # Calculate deviation for this individual sensor
                        deviation = self.calculate_deviation(self.baseline_n_values[sensor_name])
                        if deviation > max_deviation:
                            max_deviation = deviation
                            worst_sensor = sensor_name
                
                # If any sensor's deviation exceeds 10%, cancel the test
                if max_deviation > 10.0:
                    deviation_msg = f"Test canceled! Excessive baseline variation detected at 5s mark.\n\nWorst sensor ({worst_sensor}) deviation: {max_deviation:.2f}%\n\nBaseline readings should have less than 10% variation for accurate results."
                    print(deviation_msg)
                    
                    # Stop the recording
                    self.stop_recording()
                    
                    # Show alert to the user
                    QMessageBox.critical(self, "Baseline Deviation Alert", deviation_msg)
                    return
            
            # Check deviation at end of baseline (10s mark)
            if time_since_start >= BASELINE_SECONDS - 0.1 and time_since_start < BASELINE_SECONDS:
                # Calculate max deviation across all n sensors
                max_deviation = 0.0
                worst_sensor = ""
                
                for i in range(1, 9):
                    sensor_name = f"n{i}"
                    if len(self.baseline_n_values[sensor_name]) > 1:
                        # Calculate deviation for this individual sensor
                        deviation = self.calculate_deviation(self.baseline_n_values[sensor_name])
                        if deviation > max_deviation:
                            max_deviation = deviation
                            worst_sensor = sensor_name
                
                # If any sensor's deviation exceeds 5%, cancel the test
                if max_deviation > 5.0:
                    deviation_msg = f"Test canceled! Excessive baseline variation detected at end of baseline.\n\nWorst sensor ({worst_sensor}) deviation: {max_deviation:.2f}%\n\nBaseline readings should have less than 5% variation for accurate results."
                    print(deviation_msg)
                    
                    # Stop the recording
                    self.stop_recording()
                    
                    # Show alert to the user
                    QMessageBox.critical(self, "Baseline Deviation Alert", deviation_msg)
                    return
        
        # Update the last processed timestamp
        self.last_processed_raw_timestamp = raw_timestamp
        
        # Increment the tick counter
        self.tick_counter += 1

    def sample_data(self):
        """Check if we need to sample any new data"""
        if not self.is_recording or not self.current_recording:
            self.sampling_timer.stop()
            return
            
        # This is just a fallback timer to ensure we're processing data
        # The actual sampling happens directly in update_plot when new data arrives
        # No need to do anything here as the data is processed in real-time
        pass

    def stop_recording(self):
        """Stop recording and reset state without inference"""
        if not self.is_recording:
            return
            
        # Stop timers
        if self.sampling_timer and self.sampling_timer.isActive():
            self.sampling_timer.stop()
        
        if self.countdown_timer and self.countdown_timer.isActive():
            self.countdown_timer.stop()
            
        # Close the quadrant dialog
        self.quadrant_dialog.hide()
        
        # Update UI
        self.recording_status_label.setText("Recording stopped")
        self.recording_status_label.setStyleSheet("background-color: orange; color: white; font-weight: bold; padding: 5px;")
        
        # Reset recording state
        self.is_recording = False
        self.current_recording = None
        self.recording_start_time = None
        self.recording_data = None
        self.current_quadrant = None
        self.current_quadrant_index = -1
        
        # Re-enable the infer button
        self.infer_button.setEnabled(True)
        
        # Reset status label after a delay
        QTimer.singleShot(3000, lambda: self.reset_status_label())
    
    def stop_recording_and_infer(self):
        """Stop recording, save the data, and send it for inference"""
        if not self.is_recording:
            return
            
        # Stop timers
        if self.sampling_timer and self.sampling_timer.isActive():
            self.sampling_timer.stop()
        
        if self.countdown_timer and self.countdown_timer.isActive():
            self.countdown_timer.stop()
            
        # Close the quadrant dialog
        self.quadrant_dialog.hide()
        
        # Update UI
        self.recording_status_label.setText("Processing data...")
        self.recording_status_label.setStyleSheet("background-color: orange; color: white; font-weight: bold; padding: 5px;")
        
        # Save recorded data to CSV
        csv_path = None
        if self.current_recording and self.recording_data:
            try:
                with open(self.current_recording, 'w', newline='') as csvfile:
                    # Create CSV writer
                    writer = csv.writer(csvfile)
                    
                    # Write header row
                    header = ['Time', 'Phase', 'Quadrant']
                    
                    # Add sensor names to header
                    for sensor_type in ['n', 'g', 't', 'p', 'h']:
                        if sensor_type in ['n', 'g']:
                            count = 8 if sensor_type == 'n' else 2
                            for i in range(1, count + 1):
                                header.append(f"{sensor_type}{i}")
                        else:
                            header.append(sensor_type)
                    
                    writer.writerow(header)
                    
                    # Write data rows
                    for i in range(self.tick_counter):
                        row = []
                        
                        # Add time, phase, and quadrant
                        time_value = self.recording_data['time'].get(i, 0)
                        row.append(f"{time_value:.3f}")
                        
                        phase_value = self.recording_data['phase'].get(i, '')
                        row.append(str(phase_value))
                        
                        quadrant_value = self.recording_data['quadrant'].get(i, '')
                        row.append(str(quadrant_value))
                        
                        # Add sensor values
                        for sensor_type in ['n', 'g', 't', 'p', 'h']:
                            if sensor_type in ['n', 'g']:
                                count = 8 if sensor_type == 'n' else 2
                                for j in range(1, count + 1):
                                    sensor_name = f"{sensor_type}{j}"
                                    sensor_value = self.recording_data[sensor_name].get(i, '')
                                    row.append(str(sensor_value))
                            else:
                                sensor_value = self.recording_data[sensor_type].get(i, '')
                                row.append(str(sensor_value))
                        
                        writer.writerow(row)
                        
                print(f"Data saved to {self.current_recording}")
                csv_path = self.current_recording
                
                # Store the path for reference in future dialogs
                self.last_csv_path = csv_path
                
            except Exception as e:
                print(f"Error saving data: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save recording data: {e}")
                self.stop_recording()
                return
        
        # Reset recording state
        self.is_recording = False
        self.recording_start_time = None
        self.recording_data = None
        self.current_quadrant = None
        self.current_quadrant_index = -1
        
        # Now send the CSV file for inference
        if csv_path:
            # Update UI to show we're sending the data
            self.recording_status_label.setText("Sending to server...")
            self.recording_status_label.setStyleSheet("background-color: blue; color: white; font-weight: bold; padding: 5px;")
            
            # Run the API request in a separate thread to avoid blocking UI
            QTimer.singleShot(100, lambda: self.send_to_inference_api(csv_path))
        else:
            # Re-enable the infer button
            self.infer_button.setEnabled(True)
            self.recording_status_label.setText("Error: No data to send")
            self.recording_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
            QTimer.singleShot(3000, lambda: self.reset_status_label())
    
    def send_to_inference_api(self, csv_path):
        """Send the CSV file to the inference API endpoint"""
        try:
            # Prepare the file for upload
            with open(csv_path, 'rb') as f:
                files = {'file': (os.path.basename(csv_path), f, 'text/csv')}
                
                # Send the request
                print(f"Sending request to {API_ENDPOINT}...")
                response = requests.post(API_ENDPOINT, files=files)
                
                # Check if request was successful
                if response.status_code == 200:
                    print("Inference request successful")
                    result = response.json()
                    
                    # Debug the response structure
                    print(f"API response structure: {result.keys()}")
                    
                    # Update UI with the result
                    self.last_inference_result = result
                    
                    try:
                        # Update the summary result in the UI
                        self.update_last_result(result)
                        
                        # Prepare the sensor visualization
                        self.sensor_plot_dialog.load_and_plot_data(csv_path)
                        
                        # Show both dialogs
                        self.result_dialog.set_results(result, csv_path)
                        self.result_dialog.show()
                        self.sensor_plot_dialog.show()
                    except Exception as display_error:
                        print(f"Error displaying results: {display_error}")
                        traceback_info = traceback.format_exc()
                        print(f"Traceback: {traceback_info}")
                        QMessageBox.critical(self, "Display Error", 
                                            f"Error displaying results: {display_error}\n\nTraceback: {traceback_info}")
                else:
                    print(f"API request failed with status code {response.status_code}")
                    print(f"Response content: {response.text}")
                    
                    error_msg = f"API request failed with status code {response.status_code}.\n\nResponse: {response.text}"
                    QMessageBox.critical(self, "API Error", error_msg)
                    
            # Re-enable the infer button and update status
            self.reset_status_label()
            self.infer_button.setEnabled(True)
            
        except Exception as e:
            print(f"Error sending data to API: {e}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            QMessageBox.critical(self, "Error", f"Failed to send data to inference server: {e}\n\nTraceback: {traceback_info}")
            
            # Re-enable the infer button and update status
            self.reset_status_label()
            self.infer_button.setEnabled(True)
    
    def update_last_result(self, result):
        """Update the last result summary on the UI"""
        try:
            # Print the full result for debugging
            print(f"Processing result: {result}")
            
            # Safely extract values with fallbacks
            prediction = result.get('prediction', None)
            prediction_label = result.get('prediction_label', '')
            
            # If prediction_label is missing, construct it from prediction
            if not prediction_label and prediction is not None:
                if prediction == "no_peanut" or prediction == 0:
                    prediction_label = "No Peanut Detected"
                elif prediction == "peanut" or prediction == 1:
                    prediction_label = "Contains Peanut"
                else:
                    prediction_label = f"Unknown ({prediction})"
            
            # Handle confidence - cast to float for safety
            confidence = 0
            try:
                confidence = float(result.get('confidence', 0)) * 100  # Convert to percentage
            except (ValueError, TypeError):
                print(f"Error converting confidence value: {result.get('confidence')}")
                confidence = 0
            
            # Update last result label - handle different prediction formats
            if prediction == 1 or prediction == "peanut":
                self.last_result_label.setText(f"Prediction: {prediction_label}")
                self.last_result_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #CC0000;")
            else:
                self.last_result_label.setText(f"Prediction: {prediction_label}")
                self.last_result_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #006600;")
            
            # Update confidence label
            self.last_confidence_label.setText(f"Confidence: {confidence:.1f}%")
            
            # Enable the show details button
            self.show_details_button.setEnabled(True)
            
        except Exception as e:
            print(f"Error updating last result: {e}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            self.last_result_label.setText(f"Error displaying result")
            self.last_result_label.setStyleSheet("font-size: 12pt; color: red;")
    
    def show_last_result(self):
        """Show the detailed results dialog again"""
        if self.last_inference_result:
            # Use the stored last CSV path
            self.result_dialog.set_results(self.last_inference_result, self.last_csv_path)
            self.result_dialog.show()
            
            # Also show the sensor visualization
            self.sensor_plot_dialog.load_and_plot_data(self.last_csv_path)
            self.sensor_plot_dialog.show()
    
    def reset_status_label(self):
        """Reset the recording status label"""
        self.recording_status_label.setText("Not Recording")
        self.recording_status_label.setStyleSheet("background-color: transparent; color: black; font-weight: bold; padding: 5px;")

    def calculate_deviation(self, values):
        """Calculate the percentage deviation of a list of values"""
        if not values or len(values) < 2:
            return 0.0
            
        mean_value = sum(values) / len(values)
        if mean_value == 0:
            return 0.0
            
        # Calculate standard deviation
        squared_diffs = [(x - mean_value) ** 2 for x in values]
        variance = sum(squared_diffs) / len(values)
        std_dev = np.sqrt(variance)
        
        # Calculate coefficient of variation (as percentage)
        cv_percent = (std_dev / mean_value) * 100.0
        
        return cv_percent
    
    def show_window(self):
        """Display the main window"""
        self.show()
        
    def closeEvent(self, event):
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Stop timers
        if self.countdown_timer and self.countdown_timer.isActive():
            self.countdown_timer.stop()
        
        if self.sampling_timer and self.sampling_timer.isActive():
            self.sampling_timer.stop()
        
        # Stop BLE worker
        self.sensor_worker.stop()
        
        # Stop the asyncio event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        
        event.accept()

# Main block
if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    app.setApplicationName("sense: inference")
    app.setApplicationDisplayName("sense: inference")
    app.setOrganizationName("sense")
    app.setOrganizationDomain("sites.wustl.edu/sense")
    
    icon_path = os.path.join(os.path.dirname(__file__), '../icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        print(f"Warning: Icon file not found at {icon_path}")
    
    window = MainWindow(loop)
    window.show_window()

    async def run_app():
        await window.start_ble_worker()

    with loop:
        loop.create_task(run_app())
        loop.run_forever() 