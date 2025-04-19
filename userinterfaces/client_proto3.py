#client.py
import asyncio
import sys
from bleak import BleakScanner, BleakClient
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QLabel, QGridLayout, QDoubleSpinBox, QFrame, QSpinBox, QComboBox,
                             QDialog, QListWidget, QListWidgetItem, QTextEdit)
from PyQt6.QtCore import QTimer, QRunnable, QThreadPool, pyqtSignal, QObject, Qt
from PyQt6.QtGui import QColor, QIcon
import pyqtgraph as pg
import numpy as np
from collections import deque
from scipy.signal import detrend
import os
from datetime import datetime
import csv
from worker import WorkerSignals
from qasync import QEventLoop, asyncSlot
from queue import Queue
from scipy import signal
from scipy.fft import fft, ifft
import random

#constants
MULTI_SENSOR_SERVICE_UUID = "2cc12ee8-c5b6-4d7f-a3de-9c793653f271"
MULTI_SENSOR_CHARACTERISTIC_UUID = "15216e4f-bf54-4482-8a91-74a92ccfeb37"
MULTI_SENSOR_DEVICE_NAME = "03"  # Changed to match preheating.py

ODOR_DISP_SERVICE_UUID = "c622be4a-3304-42d0-8646-9a64e643978c"
ODOR_DISP_CHARACTERISTIC_UUID = "bd9b0cb0-1113-435a-b446-df0b3e2465b5"
ODOR_DISP_DEVICE_NAME = "ESPOdorDisp"[0:2]#for some reason, on windows it has to be the first two letters LOL

TIME_WINDOW = 30
MAX_DATA_POINTS = 1000
WINDOW_SIZE = 20
SENSOR_RESISTANCES = [11880, 14850, 10050, 1470, 6720, 4630]

ENABLE_SIGNAL_PROCESSING = False


BASELINE_MODE = False
BASELINE_SAMPLES = 1

COUNTDOWN_SECONDS = 10

def convert_to_resistance(voltage, resistance):
    try:
        return (5/voltage - 1) * resistance if voltage > 0 else 0
    except ZeroDivisionError:
        return 0

#ble worker class
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
        self.command_queue = Queue()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_command_queue)
        self.timer.start(100)  # Check queue every 100ms

    def notification_handler(self, sender, data):
        """Callback for when notification data is received"""
        try:
            decoded_data = data.decode('utf-8').strip('\x00')
            # print(f"Received data: {decoded_data}", file=sys.stderr)  # Add debug output
            
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
            
            # Convert n1-n6 to resistance values
            processed_data = data.copy()
            for i in range(6):
                sensor = f'n{i+1}'
                if sensor in data:
                    voltage = float(data[sensor])
                    resistance = convert_to_resistance(voltage, SENSOR_RESISTANCES[i])
                    processed_data[sensor] = resistance / 1000  # Convert to kΩ here

            self.data.emit(processed_data)
            
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

    async def send_command(self, command, signals, odor_index, odor_name):
        if self.client and self.client.is_connected:
            try:
                await self.client.write_gatt_char(self.characteristic_uuid, command.encode())
                signals.success.emit(odor_index, odor_name, command)
            except Exception as e:
                print(f"Error sending command: {e}")
                signals.failure.emit(odor_index, odor_name, command)
        else:
            print("Client not connected")
            signals.failure.emit(odor_index, odor_name, command)

    def process_command_queue(self):
        if not self.command_queue.empty():
            command, odor_index, odor_name, signals = self.command_queue.get()
            asyncio.create_task(self.send_command(command, signals, odor_index, odor_name))

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

#color square widget
class ColorSquare(QFrame):
    def __init__(self, color):
        super().__init__()
        self.setFixedSize(15, 15)
        self.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

#main window class
class MainWindow(QMainWindow):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop  # Store the loop reference
        self.setWindowTitle("sense: odordisp")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setWindowIcon(QIcon('icon.png'))
        self.setMinimumSize(800, 500)  # Set minimum size to prevent too small window

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)  # Reduce spacing between main elements
        main_layout.setContentsMargins(10, 10, 10, 10)  # Reduce margins

        # Left side - plots
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setSpacing(5)  # Reduce spacing between plots
        
        # Create plot for n1-n6 (resistance values)
        self.resistance_plot = pg.PlotWidget()
        self.resistance_plot.showGrid(x=True, y=True)
        self.resistance_plot.setLabel('left', 'Resistance (kΩ)')
        self.resistance_plot.setLabel('bottom', 'Time (s)')
        plots_layout.addWidget(self.resistance_plot, stretch=3)
        
        # Create plot for n7-n10 (voltage values)
        self.voltage_plot = pg.PlotWidget()
        self.voltage_plot.showGrid(x=True, y=True)
        self.voltage_plot.setLabel('left', 'Voltage (V)')
        self.voltage_plot.setLabel('bottom', 'Time (s)')
        self.voltage_plot.setYRange(0, 5)
        plots_layout.addWidget(self.voltage_plot, stretch=2)
        
        # Create second row with gasr
        self.gasr_plot = pg.PlotWidget()
        self.gasr_plot.showGrid(x=True, y=True)
        self.gasr_plot.setLabel('left', 'gasr')
        self.gasr_plot.setLabel('bottom', 'Time (s)')
        plots_layout.addWidget(self.gasr_plot, stretch=2)
        
        # Create third row with remaining sensors
        remaining_widget = QWidget()
        remaining_layout = QHBoxLayout(remaining_widget)
        
        self.tmp_plot = pg.PlotWidget()
        self.tmp_plot.showGrid(x=True, y=True)
        self.tmp_plot.setLabel('left', 'tmp')
        self.tmp_plot.setLabel('bottom', 'Time (s)')
        remaining_layout.addWidget(self.tmp_plot)
        
        self.pa_plot = pg.PlotWidget()
        self.pa_plot.showGrid(x=True, y=True)
        self.pa_plot.setLabel('left', 'pa')
        self.pa_plot.setLabel('bottom', 'Time (s)')
        remaining_layout.addWidget(self.pa_plot)
        
        self.hum_plot = pg.PlotWidget()
        self.hum_plot.showGrid(x=True, y=True)
        self.hum_plot.setLabel('left', 'hum')
        self.hum_plot.setLabel('bottom', 'Time (s)')
        remaining_layout.addWidget(self.hum_plot)
        
        plots_layout.addWidget(remaining_widget, stretch=1)
        main_layout.addWidget(plots_widget, stretch=2)
        
        # Store plot widgets in list for easy access
        self.plot_widgets = [
            self.resistance_plot,
            self.voltage_plot,
            self.gasr_plot,
            self.tmp_plot,
            self.pa_plot,
            self.hum_plot
        ]
        
        # Right side - controls
        right_column_widget = QWidget()
        right_column_widget.setFixedWidth(300)
        right_column_layout = QVBoxLayout(right_column_widget)
        right_column_layout.setSpacing(5)
        right_column_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(right_column_widget, stretch=0)

        # Connection status and controls section
        connection_widget = QWidget()
        connection_layout = QHBoxLayout(connection_widget)
        connection_layout.setSpacing(5)  # Reduce spacing
        connection_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        # Left side - MultiSensor status and reconnect
        multi_sensor_group = QWidget()
        multi_sensor_layout = QVBoxLayout(multi_sensor_group)
        self.multi_sensor_status_label = QLabel("Sense32: Disconnected")
        self.multi_sensor_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.multi_sensor_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
        self.multi_sensor_reconnect_button = QPushButton("Reconnect Sense32")
        self.multi_sensor_reconnect_button.clicked.connect(self.reconnect_multi_sensor)
        multi_sensor_layout.addWidget(self.multi_sensor_status_label)
        multi_sensor_layout.addWidget(self.multi_sensor_reconnect_button)
        connection_layout.addWidget(multi_sensor_group)

        # Right side - OdorDisp status and reconnect
        odor_disp_group = QWidget()
        odor_disp_layout = QVBoxLayout(odor_disp_group)
        self.odor_disp_status_label = QLabel("ESPOdorDisp: Disconnected")
        self.odor_disp_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.odor_disp_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
        self.odor_disp_reconnect_button = QPushButton("Reconnect ESPOdorDisp")
        self.odor_disp_reconnect_button.clicked.connect(self.reconnect_odor_disp)
        odor_disp_layout.addWidget(self.odor_disp_status_label)
        odor_disp_layout.addWidget(self.odor_disp_reconnect_button)
        connection_layout.addWidget(odor_disp_group)

        right_column_layout.addWidget(connection_widget)

        # Add baseline correction button
        baseline_widget = QWidget()
        baseline_layout = QHBoxLayout(baseline_widget)
        self.baseline_button = QPushButton("Apply Baseline Correction")
        self.baseline_button.clicked.connect(self.toggle_baseline_correction)
        baseline_layout.addWidget(self.baseline_button)
        right_column_layout.addWidget(baseline_widget)

        self.config_file = 'config.json'
        self.odor_names = self.load_odor_names()

        #odor control section
        self.odor_labels = []
        self.odor_buttons = []
        self.odor_inputs = []
        self.odor_name_inputs = []
        self.odor_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']

        for i in range(8):
            odor_widget = QWidget()
            odor_layout = QHBoxLayout(odor_widget)

            color_square = ColorSquare(self.odor_colors[i])
            odor_layout.addWidget(color_square)

            name_input = QLineEdit(self.odor_names.get(f'odor_{i+1}', f'Odor {i+1}'))
            name_input.setPlaceholderText(f'Odor {i+1}')
            name_input.textChanged.connect(lambda text, idx=i: self.update_odor_name(idx, text))
            name_input.setMinimumWidth(150)
            odor_layout.addWidget(name_input, 3)

            input_field = QDoubleSpinBox()
            input_field.setRange(0, 99999)
            input_field.setDecimals(1)
            input_field.setSingleStep(0.1)
            input_field.setSuffix(" ms")
            input_field.setMaximumWidth(100)
            input_field.setValue(4000)  # Set default value to 4000 ms
            odor_layout.addWidget(input_field, 1)

            send_button = QPushButton("Send")
            send_button.clicked.connect(lambda _, idx=i: self.send_odor_command(idx))
            odor_layout.addWidget(send_button, 1)

            right_column_layout.addWidget(odor_widget)

            self.odor_labels.append(name_input)
            self.odor_buttons.append(send_button)
            self.odor_inputs.append(input_field)
            self.odor_name_inputs.append(name_input)

        right_column_layout.addStretch(1)

        #recording status label
        self.recording_status_label = QLabel("Not Recording")
        self.recording_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_status_label.setStyleSheet("background-color: transparent; color: black; font-weight: bold; padding: 5px;")
        right_column_layout.addWidget(self.recording_status_label)

        #duration field
        duration_widget = QWidget()
        duration_layout = QHBoxLayout(duration_widget)
        duration_label = QLabel("Recording Duration (seconds):")
        self.duration_input = QSpinBox()
        self.duration_input.setRange(1, 3600)
        self.duration_input.setValue(60)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_input)
        right_column_layout.addWidget(duration_widget)

        # Instead, create the countdown dialog
        self.countdown_dialog = CountdownDialog(self)
        
        # Initialize countdown timer
        self.countdown_timer = QTimer()
        self.countdown_timer.setInterval(1000)  # 1 second intervals
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_remaining = 0

        # Signal processing options
        processing_widget = QWidget()
        processing_layout = QHBoxLayout(processing_widget)
        processing_label = QLabel("Signal Processing:")
        self.processing_combo = QComboBox()
        self.processing_combo.addItems([
            "None", 
            "Smoothing Only", 
            "DC Removal Only",
            "Smoothing + DC Removal"
        ])
        self.processing_combo.currentTextChanged.connect(self.update_processing)
        processing_layout.addWidget(processing_label)
        processing_layout.addWidget(self.processing_combo)
        right_column_layout.addWidget(processing_widget)

        # Show/hide processing options based on flag
        processing_widget.setVisible(ENABLE_SIGNAL_PROCESSING)

        # Initialize processing flags
        self.do_smoothing = False
        self.do_dc_removal = False

        # Initialize curves with different colors
        self.curves = {}
        colors = ['r', 'b', 'g', 'c', 'm', 'y', (255, 165, 0), 'w', (128, 0, 128), (255, 192, 203)]
        
        # Add curves for n1-n6 (resistance values)
        resistance_legend = self.resistance_plot.addLegend()
        for i in range(6):
            sensor = f's{i+1}'
            curve = self.resistance_plot.plot(pen=colors[i], name=f'{sensor} (kΩ)')
            self.curves[sensor] = curve
            
        # Add curves for n7-n10 (voltage values)
        voltage_legend = self.voltage_plot.addLegend()
        for i in range(6, 10):
            sensor = f's{i+1}'
            curve = self.voltage_plot.plot(pen=colors[i], name=sensor)
            self.curves[sensor] = curve
            
        # Add curves for other sensors
        self.curves['gasr'] = self.gasr_plot.plot(pen='w', name='gasr')
        self.curves['tmp'] = self.tmp_plot.plot(pen='w', name='tmp')
        self.curves['pa'] = self.pa_plot.plot(pen='w', name='pa')
        self.curves['hum'] = self.hum_plot.plot(pen='w', name='hum')
        
        # Initialize data storage for plotting
        self.plot_data = {
            sensor: {'time': deque(maxlen=MAX_DATA_POINTS), 
                    'value': deque(maxlen=MAX_DATA_POINTS)} 
            for sensor in self.curves.keys()
        }
        
        # Initialize puff scatter plots for s1-s10
        self.puff_scatter = {
            sensor: pg.ScatterPlotItem(pen=pg.mkPen(None), size=10)
            for sensor in [f's{i}' for i in range(1, 11)]
        }
        # Add scatter plots to appropriate plots based on sensor number
        for sensor, scatter in self.puff_scatter.items():
            sensor_num = int(sensor[1:])
            if sensor_num <= 6:
                self.resistance_plot.addItem(scatter)
            else:
                self.voltage_plot.addItem(scatter)

        # Initialize recording end markers
        self.recording_end_scatter = {
            sensor: pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush('w'), size=10)
            for sensor in [f's{i}' for i in range(1, 11)]
        }
        # Add end markers to appropriate plots based on sensor number
        for sensor, scatter in self.recording_end_scatter.items():
            sensor_num = int(sensor[1:])
            if sensor_num <= 6:
                self.resistance_plot.addItem(scatter)
            else:
                self.voltage_plot.addItem(scatter)

        # Initialize tick counter for precise sampling
        self.tick_counter = 0

        # Add cycle generator button
        cycle_generator_widget = QWidget()
        cycle_generator_layout = QHBoxLayout(cycle_generator_widget)
        self.cycle_generator_button = QPushButton("Cycle Generator")
        self.cycle_generator_button.clicked.connect(self.show_cycle_generator)
        cycle_generator_layout.addWidget(self.cycle_generator_button)
        right_column_layout.addWidget(cycle_generator_widget)

        # Initialize ble workers
        self.multi_sensor_worker = BLEWorker(MULTI_SENSOR_SERVICE_UUID, MULTI_SENSOR_CHARACTERISTIC_UUID, MULTI_SENSOR_DEVICE_NAME)
        self.multi_sensor_worker.data.connect(self.update_plot)
        self.multi_sensor_worker.connection_status.connect(self.update_multi_sensor_status)

        self.odor_disp_worker = BLEWorker(ODOR_DISP_SERVICE_UUID, ODOR_DISP_CHARACTERISTIC_UUID, ODOR_DISP_DEVICE_NAME)
        self.odor_disp_worker.connection_status.connect(self.update_odor_disp_status)

        # We'll start the BLE workers in the start_ble_workers method

        self.start_time = None

        self.current_recording = None
        self.recording_start_time = None

        # Initialize a Thread Pool
        self.thread_pool = QThreadPool()

        # Initialize puff state
        self.puff_active = False
        self.puff_end_time = None
        self.puff_data = deque(maxlen=MAX_DATA_POINTS)  # To store puff states for CSV

        # Initialize a QTimer for sampling data every 250ms
        self.sampling_timer = QTimer()
        self.sampling_timer.setInterval(250)  # 250ms intervals
        self.sampling_timer.timeout.connect(self.sample_data)

    async def start_ble_workers(self):
        await asyncio.gather(
            self.multi_sensor_worker.run(),
            self.odor_disp_worker.run()
        )

    # Signal processing methods
    def moving_average(self, data, window_size):
        if len(data) < window_size:
            return list(data)
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    def fft_dc_removal(self, data):
        if len(data) == 0:
            return []
        # Use simple mean subtraction instead of FFT
        mean_val = np.mean(data)
        return [x - mean_val for x in data]

    def calculate_initial_baseline(self, sensor):
        """Calculate baseline using current values for each sensor"""
        if not hasattr(self, 'sensor_baselines'):
            self.sensor_baselines = {}
        
        if sensor not in self.sensor_baselines and len(self.plot_data[sensor]['value']) > 0:
            # Use the average of last few values as baseline for more stability
            num_samples = min(BASELINE_SAMPLES, len(self.plot_data[sensor]['value']))
            baseline = sum(list(self.plot_data[sensor]['value'])[-num_samples:]) / num_samples
            self.sensor_baselines[sensor] = baseline
            print(f"Established baseline for {sensor}: {baseline:.3f}V")
            return baseline
        
        return self.sensor_baselines.get(sensor, 0)

    def get_baseline(self, sensor):
        """Get the current baseline for a sensor"""
        if not hasattr(self, 'sensor_baselines'):
            self.sensor_baselines = {}
        return self.sensor_baselines.get(sensor, 0)

    # Plot update method
    def update_plot(self, data):
        if self.start_time is None:
            self.start_time = data.get('x', 0) / 1000
        
        current_time = data.get('x', 0) / 1000 - self.start_time

        # Process all sensors
        for sensor_name in self.curves.keys():
            if sensor_name.startswith('s'):
                # Map n1-n10 to s1-s10
                sensor_num = int(sensor_name[1:])  # Extract number from s1, s2, etc.
                input_sensor = f'n{sensor_num}'  # Convert to n1, n2, etc.
                
                if input_sensor in data:
                    if sensor_num <= 6:  # For s1-s6, use resistance values (already in kΩ)
                        processed_value = float(data[input_sensor])
                    else:  # For s7-s10, use voltage values
                        processed_value = float(data[input_sensor])
                    
                    if BASELINE_MODE:
                        baseline = self.get_baseline(sensor_name)
                        processed_value = processed_value - baseline
                    
                    processed_value = max(0, processed_value)
                    
                    self.plot_data[sensor_name]['time'].append(current_time)
                    self.plot_data[sensor_name]['value'].append(processed_value)
            else:
                # Handle other sensors (tmp, pa, hum, gasr)
                sensor_map = {'tmp': 't', 'pa': 'p', 'hum': 'h', 'gasr': 'g'}
                data_key = sensor_map.get(sensor_name, sensor_name)
                
                if data_key in data:
                    try:
                        value = float(data[data_key])
                        self.plot_data[sensor_name]['time'].append(current_time)
                        self.plot_data[sensor_name]['value'].append(value)
                    except (ValueError, TypeError) as e:
                        print(f"Error converting {sensor_name} value: {data[data_key]}, Error: {e}")
                        continue

        # Update curves with their current data
        try:
            x_min = max(0, current_time - TIME_WINDOW)
            x_max = max(TIME_WINDOW, current_time)
            
            # Update all curves and collect values for y-range calculation
            resistance_values = []
            voltage_values = []
            
            for sensor_name, curve in self.curves.items():
                if len(self.plot_data[sensor_name]['time']) > 0:
                    x_data = list(self.plot_data[sensor_name]['time'])
                    y_data = list(self.plot_data[sensor_name]['value'])
                    
                    if len(x_data) == len(y_data):
                        curve.setData(x_data, y_data)
                        
                        # Collect values for y-range calculation
                        if sensor_name.startswith('s'):
                            sensor_num = int(sensor_name[1:])
                            if sensor_num <= 6:  # s1-s6 are resistance values
                                resistance_values.extend(y_data)
                            else:  # s7-s10 are voltage values
                                voltage_values.extend(y_data)
                    else:
                        print(f"Data length mismatch for {sensor_name}: x={len(x_data)}, y={len(y_data)}")
            
            # Update x-range for all plots
            for plot_widget in self.plot_widgets:
                plot_widget.setXRange(x_min, x_max)
            
            # Update y-range for resistance plot (s1-s6)
            if resistance_values:
                y_min = min(v for v in resistance_values if v > 0)  # Ignore zero values
                y_max = max(resistance_values)
                padding = (y_max - y_min) * 0.1
                self.resistance_plot.setYRange(max(0, y_min - padding), y_max + padding)
            
            # Update y-range for voltage plot (s7-s10)
            if voltage_values:
                self.voltage_plot.setYRange(0, 5)  # Fixed range for voltage values
            
            # Dynamic y-axis for other sensors
            for sensor_name in ['gasr', 'tmp', 'pa', 'hum']:
                if self.plot_data[sensor_name]['value']:
                    y_min = min(self.plot_data[sensor_name]['value'])
                    y_max = max(self.plot_data[sensor_name]['value'])
                    padding = (y_max - y_min) * 0.1 if y_max != y_min else 1
                    if sensor_name == 'gasr':
                        self.gasr_plot.setYRange(y_min - padding, y_max + padding)
                    elif sensor_name == 'tmp':
                        self.tmp_plot.setYRange(y_min - padding, y_max + padding)
                    elif sensor_name == 'pa':
                        self.pa_plot.setYRange(y_min - padding, y_max + padding)
                    elif sensor_name == 'hum':
                        self.hum_plot.setYRange(y_min - padding, y_max + padding)
                    
        except Exception as e:
            print(f"Error updating plots: {e}")

    # Status update methods
    def update_multi_sensor_status(self, status):
        if status == "connected":
            self.multi_sensor_status_label.setText("Sense32: Connected")
            self.multi_sensor_status_label.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 5px;")
        elif status == "disconnected":
            self.multi_sensor_status_label.setText("Sense32: Disconnected")
            self.multi_sensor_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
        elif status == "connecting":
            self.multi_sensor_status_label.setText("Sense32: Connecting...")
            self.multi_sensor_status_label.setStyleSheet("background-color: orange; color: white; font-weight: bold; padding: 5px;")
        elif status == "searching":
            self.multi_sensor_status_label.setText("Sense32: Searching...")
            self.multi_sensor_status_label.setStyleSheet("background-color: blue; color: white; font-weight: bold; padding: 5px;")

    def update_odor_disp_status(self, status):
        if status == "connected":
            self.odor_disp_status_label.setText("ESPOdorDisp: Connected")
            self.odor_disp_status_label.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 5px;")
        elif status == "disconnected":
            self.odor_disp_status_label.setText("ESPOdorDisp: Disconnected")
            self.odor_disp_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
        elif status == "connecting":
            self.odor_disp_status_label.setText("ESPOdorDisp: Connecting...")
            self.odor_disp_status_label.setStyleSheet("background-color: orange; color: white; font-weight: bold; padding: 5px;")
        elif status == "searching":
            self.odor_disp_status_label.setText("ESPOdorDisp: Searching...")
            self.odor_disp_status_label.setStyleSheet("background-color: blue; color: white; font-weight: bold; padding: 5px;")
        
        for button in self.odor_buttons:
            button.setEnabled(status == "connected")

    # Odor name methods
    def load_odor_names(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def save_odor_names(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.odor_names, f)

    def update_odor_name(self, index, name):
        self.odor_names[f'odor_{index+1}'] = name
        self.save_odor_names()

    # Odor command method
    def send_odor_command(self, odor_index):
        ms_value = int(self.odor_inputs[odor_index].value())
        odor_name = self.odor_name_inputs[odor_index].text()
        
        # Start countdown instead of immediate execution
        self.start_countdown(odor_index, odor_name, ms_value)

    def start_countdown(self, odor_index, odor_name, ms_value):
        """Start countdown and begin recording baseline"""
        self.countdown_remaining = COUNTDOWN_SECONDS
        self.countdown_dialog.countdown_label.setText(f"Starting in: {self.countdown_remaining}")
        self.countdown_dialog.show()
        
        # Store parameters for when countdown finishes
        self.pending_recording = {
            'odor_index': odor_index,
            'odor_name': odor_name,
            'ms_value': ms_value
        }
        
        # Start recording immediately with the countdown period
        self.start_recording(odor_name, ms_value, self.plot_data['s1']['time'][-1] if self.plot_data['s1']['time'] else 0)
        
        # Start countdown timer
        self.countdown_timer.start()
        
        # Disable all odor buttons during countdown
        for button in self.odor_buttons:
            button.setEnabled(False)

    def update_countdown(self):
        """Update countdown timer display"""
        self.countdown_remaining -= 1
        if self.countdown_remaining > 0:
            self.countdown_dialog.countdown_label.setText(f"Starting in: {self.countdown_remaining}")
        else:
            self.countdown_timer.stop()
            self.countdown_dialog.hide()
            
            # Re-enable odor buttons
            for button in self.odor_buttons:
                button.setEnabled(True)
            
            # Execute the odor command at time=0
            self.execute_odor_command(**self.pending_recording)
            self.pending_recording = None

    def execute_odor_command(self, odor_index, odor_name, ms_value):
        command = f"OD{odor_index + 1}.{ms_value:05d}"
        
        # Start puff
        self.puff_active = True
        puff_duration_sec = ms_value / 1000  # Convert ms to seconds
        self.puff_duration_sec = puff_duration_sec
        
        # Get the color for this odor
        odor_color = self.odor_colors[odor_index]

        # Get current time for plotting (actual plot time)
        current_plot_time = self.plot_data['s1']['time'][-1] if self.plot_data['s1']['time'] else 0

        # Plot puff start impulse at current real time
        for sensor in [f's{i}' for i in range(1, 9)]:
            self.puff_scatter[sensor].addPoints(x=[current_plot_time], y=[1], brush=pg.mkBrush(odor_color))

        # Schedule puff end impulse at real time + puff duration
        QTimer.singleShot(ms_value, lambda: self.plot_puff_end(odor_color, current_plot_time + puff_duration_sec))
        
        signals = WorkerSignals()
        signals.success.connect(self.on_command_success)
        signals.failure.connect(self.on_command_failure)
        
        self.odor_disp_worker.command_queue.put((command, odor_index, odor_name, signals))

    def plot_puff_end(self, odor_color, end_time):
        # Insert impulse at puff_end_time
        for sensor in [f's{i}' for i in range(1, 9)]:
            self.puff_scatter[sensor].addPoints(x=[end_time], y=[1], brush=pg.mkBrush(odor_color))

        # Update puff_active state
        self.puff_active = False

    def on_command_success(self, odor_index, odor_name, command):
        self.odor_buttons[odor_index].setStyleSheet("background-color: green;")
        QTimer.singleShot(500, lambda: self.odor_buttons[odor_index].setStyleSheet(""))
        
        print(f"Sent command for {odor_name}: {command}")

    def on_command_failure(self, odor_index, odor_name, command):
        self.odor_buttons[odor_index].setStyleSheet("background-color: red;")
        QTimer.singleShot(500, lambda: self.odor_buttons[odor_index].setStyleSheet(""))
        print(f"Failed to send command for {odor_name}: {command}")

    # Recording methods
    def start_recording(self, odor_name, ms_value, start_plot_time):
        self.recording_start_time = datetime.now()
        duration = self.duration_input.value()
        self.puff_duration_sec = ms_value / 1000
        
        self.recording_status_label.setText("Recording Baseline")
        self.recording_status_label.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
        
        sanitized_odor_name = odor_name.replace('/', '_').replace('\\', '_')
        timestamp_str = self.recording_start_time.strftime('%m-%d-%Y-%I%M%p').lower()
        file_name = f"{ms_value}mspuff_{duration}s_{timestamp_str}.csv"
        
        # Determine folder path based on whether this is part of a cycle
        if hasattr(self, 'cycle_params') and self.cycle_params:
            # For cycled recordings, use cycled/odorname/
            folder_name = f"cycled/{sanitized_odor_name}"
        else:
            # For individual recordings, use results/odorname/
            folder_name = f"results/{sanitized_odor_name}"
            
        os.makedirs(folder_name, exist_ok=True)
        self.current_recording = os.path.join(folder_name, file_name)
        
        # Calculate total recording duration including baseline and final point
        total_duration = duration + COUNTDOWN_SECONDS
        self.total_ticks = int(total_duration / 0.250) + 1  # Add 1 to include final point
        
        # Initialize recording data with negative time for baseline, including final point
        self.recording_data = {
            'time': [round((i * 0.250) - COUNTDOWN_SECONDS, 3) for i in range(self.total_ticks)],
            'puff': [1 if 0 <= (i * 0.250 - COUNTDOWN_SECONDS) < self.puff_duration_sec else 0 
                    for i in range(self.total_ticks)]
        }
        
        # Initialize data structures for each sensor
        for sensor_name in self.curves.keys():
            if sensor_name.startswith('s'):
                # For s1-s10, store both raw and processed values
                self.recording_data[sensor_name] = {
                    'raw': [None] * self.total_ticks,
                    'processed': [None] * self.total_ticks
                }
            else:
                # For environmental sensors, store only processed values
                self.recording_data[sensor_name] = {
                    'processed': [None] * self.total_ticks
                }
        
        # Store the actual plot time when recording started
        self.recording_real_start_time = start_plot_time
        
        # Start the sampling timer
        self.tick_counter = 0
        self.sampling_timer.start()

    def sample_data(self):
        if not self.current_recording:
            return
        
        if self.tick_counter >= self.total_ticks:
            self.sampling_timer.stop()
            self.stop_recording()
            return
        
        # Sample data at current tick
        for sensor_name in self.curves.keys():
            if len(self.plot_data[sensor_name]['value']) > 0:
                latest_value = self.plot_data[sensor_name]['value'][-1]
                
                if sensor_name.startswith('s'):
                    # For s1-s10, store both raw and processed values
                    self.recording_data[sensor_name]['raw'][self.tick_counter] = latest_value
                    self.recording_data[sensor_name]['processed'][self.tick_counter] = latest_value
                else:
                    # For environmental sensors, store only processed values
                    self.recording_data[sensor_name]['processed'][self.tick_counter] = latest_value
        
        self.tick_counter += 1

        # Update recording status with current time
        current_time = (self.tick_counter * 0.250) - COUNTDOWN_SECONDS
        if current_time < 0:
            self.recording_status_label.setText(f"Recording Baseline (t={current_time:.1f}s)")
        else:
            self.recording_status_label.setText(f"Recording (t={current_time:.1f}s)")

    def stop_recording(self):
        if self.current_recording and self.recording_start_time:
            # Ensure we've captured all data points
            while self.tick_counter < self.total_ticks:
                self.sample_data()
            
            # Stop the sampling timer
            self.sampling_timer.stop()
            
            # Write to CSV
            with open(self.current_recording, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Define header
                header = ['Time']
                # Add raw values for s1-s10
                for sensor_name in self.curves.keys():
                    if sensor_name.startswith('s'):
                        header.append(f'{sensor_name}_raw')
                # Add processed values for all sensors
                for sensor_name in self.curves.keys():
                    header.append(f'{sensor_name}_processed')
                header.append('Puff')
                writer.writerow(header)
                
                # Write data rows
                for i in range(self.total_ticks):
                    row = [f"{self.recording_data['time'][i]:.3f}"]
                    
                    # Add raw values for s1-s10
                    for sensor_name in self.curves.keys():
                        if sensor_name.startswith('s'):
                            raw = self.recording_data[sensor_name]['raw'][i]
                            row.append(raw if raw is not None else '')
                    
                    # Add processed values for all sensors
                    for sensor_name in self.curves.keys():
                        processed = self.recording_data[sensor_name]['processed'][i]
                        row.append(processed if processed is not None else '')
                    
                    # Add puff state
                    row.append(self.recording_data['puff'][i])
                    writer.writerow(row)
            
            print(f"Data saved to {self.current_recording}")
            
            # Plot white dot at actual end time - account for countdown period
            actual_end_time = self.recording_real_start_time + COUNTDOWN_SECONDS + self.duration_input.value()
            for sensor in [f's{i}' for i in range(1, 11)]:  # Changed to include s1-s10
                self.recording_end_scatter[sensor].addPoints(x=[actual_end_time], y=[1])
            
            # Clear recording state
            self.current_recording = None
            self.recording_start_time = None
            self.recording_data = None
            
            # Update recording status
            if hasattr(self, 'cycle_params') and self.cycle_params:
                remaining = (self.cycle_params['repeats'] - self.cycle_params['current_repeat']) * len(self.cycle_params['selected_odors']) - self.cycle_params['current_odor_index']
                if remaining > 0:
                    self.recording_status_label.setText(f"Saving... ({remaining} cycles remaining)")
                    self.recording_status_label.setStyleSheet("background-color: orange; color: white; font-weight: bold; padding: 5px;")
                else:
                    self.recording_status_label.setText("Cycle complete")
                    self.recording_status_label.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 5px;")
            else:
                self.recording_status_label.setText("Not Recording")
                self.recording_status_label.setStyleSheet("background-color: transparent; color: black; font-weight: bold; padding: 5px;")

    # Reconnect methods
    def reconnect_multi_sensor(self):
        self.multi_sensor_worker.stop()
        self.multi_sensor_worker = BLEWorker(MULTI_SENSOR_SERVICE_UUID, MULTI_SENSOR_CHARACTERISTIC_UUID, MULTI_SENSOR_DEVICE_NAME)
        self.multi_sensor_worker.data.connect(self.update_plot)
        self.multi_sensor_worker.connection_status.connect(self.update_multi_sensor_status)
        asyncio.create_task(self.multi_sensor_worker.run())

    def reconnect_odor_disp(self):
        self.odor_disp_worker.stop()
        self.odor_disp_worker = BLEWorker(ODOR_DISP_SERVICE_UUID, ODOR_DISP_CHARACTERISTIC_UUID, ODOR_DISP_DEVICE_NAME)
        self.odor_disp_worker.connection_status.connect(self.update_odor_disp_status)
        asyncio.create_task(self.odor_disp_worker.run())

    def closeEvent(self, event):
        # Stop BLE workers
        self.multi_sensor_worker.stop()
        self.odor_disp_worker.stop()
        
        # Stop the asyncio event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        
        event.accept()

    def show_window(self):
        self.show()

    def update_processing(self, selection):
        self.do_smoothing = "Smoothing" in selection
        self.do_dc_removal = "DC Removal" in selection

    def toggle_baseline_correction(self):
        """Toggle baseline correction and update button text"""
        global BASELINE_MODE
        BASELINE_MODE = not BASELINE_MODE
        
        if BASELINE_MODE:
            self.baseline_button.setText("Remove Baseline Correction")
            self.baseline_button.setStyleSheet("background-color: green; color: white;")
            
            # Calculate new baselines for all sensors
            for sensor_name in self.curves.keys():
                if sensor_name.startswith('s') and len(self.plot_data[sensor_name]['value']) > 0:
                    self.calculate_initial_baseline(sensor_name)
                
        else:
            self.baseline_button.setText("Apply Baseline Correction")
            self.baseline_button.setStyleSheet("")
            
            # Clear the baselines
            if hasattr(self, 'sensor_baselines'):
                self.sensor_baselines = {}
        
        # Force update of all plotted data
        if hasattr(self, 'plot_data'):
            for sensor_name in self.curves.keys():
                if sensor_name.startswith('s') and len(self.plot_data[sensor_name]['value']) > 0:
                    last_data = {'time': self.plot_data[sensor_name]['time'][-1]}
                    last_data[f'n{sensor_name[1:]}'] = self.plot_data[sensor_name]['value'][-1]
                    self.update_plot(last_data)

    def show_cycle_generator(self):
        dialog = CycleGeneratorDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Get configuration values
            repeats = dialog.repeats_spinbox.value()
            baseline_duration = dialog.baseline_spinbox.value()
            recording_duration = dialog.recording_spinbox.value()
            pulse_duration = dialog.pulse_spinbox.value()
            selected_odors = dialog.get_selected_odors()  # List of (index, name) tuples
            
            if not selected_odors:
                return  # Don't start if no odors selected
            
            # Start the cycle
            self.start_cycle(repeats, baseline_duration, recording_duration, pulse_duration, selected_odors)

    def start_cycle(self, repeats, baseline_duration, recording_duration, pulse_duration, selected_odors):
        # Create a list of randomized odor sequences for each repeat
        randomized_sequences = []
        for _ in range(repeats):
            # Create a copy of selected_odors and shuffle it
            repeat_sequence = selected_odors.copy()
            random.shuffle(repeat_sequence)
            randomized_sequences.append(repeat_sequence)
        
        # Store cycle parameters
        self.cycle_params = {
            'repeats': repeats,
            'baseline_duration': baseline_duration,
            'recording_duration': recording_duration,
            'pulse_duration': pulse_duration,
            'selected_odors': selected_odors,  # Keep original list for reference
            'randomized_sequences': randomized_sequences,  # Store the randomized sequences
            'current_repeat': 0,
            'current_odor_index': 0
        }
        
        # Start first cycle
        self.execute_next_cycle()

    def execute_next_cycle(self):
        if not hasattr(self, 'cycle_params') or not self.cycle_params:
            return
            
        # Check if we've completed all repeats
        if self.cycle_params['current_repeat'] >= self.cycle_params['repeats']:
            self.cycle_params = None
            self.recording_status_label.setText("Cycle complete")
            self.recording_status_label.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 5px;")
            return
            
        # Get current odor from the randomized sequence for this repeat
        current_odor_idx = self.cycle_params['current_odor_index']
        current_sequence = self.cycle_params['randomized_sequences'][self.cycle_params['current_repeat']]
        
        if current_odor_idx >= len(current_sequence):
            # Move to next repeat
            self.cycle_params['current_repeat'] += 1
            self.cycle_params['current_odor_index'] = 0
            
            if self.cycle_params['current_repeat'] >= self.cycle_params['repeats']:
                self.cycle_params = None
                self.recording_status_label.setText("Cycle complete")
                self.recording_status_label.setStyleSheet("background-color: green; color: white; font-weight: bold; padding: 5px;")
                return
            
            current_odor_idx = 0
            current_sequence = self.cycle_params['randomized_sequences'][self.cycle_params['current_repeat']]
            
        # Get odor details from the randomized sequence
        odor_index, odor_name = current_sequence[current_odor_idx]
        self.cycle_params['current_odor_index'] += 1
        
        # Update duration for recording
        self.duration_input.setValue(self.cycle_params['recording_duration'])
        
        # Update status with current cycle info
        remaining_cycles = (self.cycle_params['repeats'] - self.cycle_params['current_repeat']) * len(self.cycle_params['selected_odors']) - self.cycle_params['current_odor_index']
        self.recording_status_label.setText(f"Cycle {self.cycle_params['current_repeat'] + 1}/{self.cycle_params['repeats']}, {odor_name} ({remaining_cycles} cycles remaining)")
        
        # Start the cycle
        self.start_countdown(odor_index, odor_name, self.cycle_params['pulse_duration'])
        
        # Calculate total duration including:
        # 1. Recording duration
        # 2. Baseline duration (waiting period)
        # 3. Extra buffer time for saving (1 second)
        # 4. Countdown duration
        total_duration = ((self.cycle_params['recording_duration'] + 
                         self.cycle_params['baseline_duration']) * 1000 +  # Convert to ms
                         1000 +  # 1 second buffer for saving
                         COUNTDOWN_SECONDS * 1000)  # Countdown duration in ms
        
        # Schedule next cycle after ensuring current one is complete
        QTimer.singleShot(total_duration, self.execute_next_cycle)

class CountdownDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Create semi-transparent background
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(0, 0, 0, 150);
            }
            QLabel {
                color: white;
                font-size: 48px;
                font-weight: bold;
                background-color: transparent;
                padding: 20px;
            }
        """)
        
        layout = QVBoxLayout()
        self.countdown_label = QLabel()
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.countdown_label)
        self.setLayout(layout)

    def showEvent(self, event):
        # Center the dialog on the parent window
        if self.parent():
            self.move(
                self.parent().x() + (self.parent().width() - self.width()) // 2,
                self.parent().y() + (self.parent().height() - self.height()) // 2
            )
        super().showEvent(event)

class CycleGeneratorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cycle Generator")
        self.setModal(True)
        self.setFixedWidth(400)  # Increased width for preview
        self.parent = parent
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Number of repeats
        repeats_layout = QHBoxLayout()
        repeats_label = QLabel("Number of repeats:")
        self.repeats_spinbox = QSpinBox()
        self.repeats_spinbox.setRange(1, 1000)
        self.repeats_spinbox.setValue(3)
        repeats_layout.addWidget(repeats_label)
        repeats_layout.addWidget(self.repeats_spinbox)
        layout.addLayout(repeats_layout)
        
        # Baseline duration
        baseline_layout = QHBoxLayout()
        baseline_label = QLabel("Baseline duration (s):")
        self.baseline_spinbox = QSpinBox()
        self.baseline_spinbox.setRange(1, 3600)
        self.baseline_spinbox.setValue(10)
        baseline_layout.addWidget(baseline_label)
        baseline_layout.addWidget(self.baseline_spinbox)
        layout.addLayout(baseline_layout)
        
        # Recording duration
        recording_layout = QHBoxLayout()
        recording_label = QLabel("Recording duration (s):")
        self.recording_spinbox = QSpinBox()
        self.recording_spinbox.setRange(1, 3600)
        self.recording_spinbox.setValue(60)  # Changed default to 60 seconds
        recording_layout.addWidget(recording_label)
        recording_layout.addWidget(self.recording_spinbox)
        layout.addLayout(recording_layout)
        
        # Pulse duration
        pulse_layout = QHBoxLayout()
        pulse_label = QLabel("Pulse duration (ms):")
        self.pulse_spinbox = QSpinBox()
        self.pulse_spinbox.setRange(1, 60000)
        self.pulse_spinbox.setValue(4000)
        pulse_layout.addWidget(pulse_label)
        pulse_layout.addWidget(self.pulse_spinbox)
        layout.addLayout(pulse_layout)
        
        # Odor selection
        odor_layout = QVBoxLayout()  # Changed to vertical for better organization
        odor_label = QLabel("Odors to cycle (select multiple):")
        self.odor_list = QListWidget()
        self.odor_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        # Add odors with their actual names from the parent window
        for i in range(8):
            odor_name = parent.odor_name_inputs[i].text()
            item = QListWidgetItem(odor_name)
            item.setData(Qt.ItemDataRole.UserRole, i)  # Store the odor index
            self.odor_list.addItem(item)
        
        odor_layout.addWidget(odor_label)
        odor_layout.addWidget(self.odor_list)
        layout.addLayout(odor_layout)
        
        # Protocol Preview
        preview_label = QLabel("Protocol Preview:")
        layout.addWidget(preview_label)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(100)
        layout.addWidget(self.preview_text)
        
        # Connect value changes to preview update
        self.repeats_spinbox.valueChanged.connect(self.update_preview)
        self.baseline_spinbox.valueChanged.connect(self.update_preview)
        self.recording_spinbox.valueChanged.connect(self.update_preview)
        self.pulse_spinbox.valueChanged.connect(self.update_preview)
        self.odor_list.itemSelectionChanged.connect(self.update_preview)
        
        # Buttons
        button_layout = QHBoxLayout()
        start_button = QPushButton("Start")
        start_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(start_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Initial preview update
        self.update_preview()
    
    def update_preview(self):
        preview = []
        repeats = self.repeats_spinbox.value()
        baseline = self.baseline_spinbox.value()
        recording = self.recording_spinbox.value()
        pulse = self.pulse_spinbox.value()
        selected_odors = [item.text() for item in self.odor_list.selectedItems()]
        
        if not selected_odors:
            self.preview_text.setText("Please select at least one odor.")
            return
        
        total_time = 0
        preview.append("Protocol summary:")
        
        for repeat in range(repeats):
            for odor in selected_odors:
                if repeat == 0:  # Only show detailed timing for first repeat
                    preview.append(f"\nRepeat 1, {odor}:")
                    preview.append(f"- Baseline: {baseline}s")
                    preview.append(f"- Pulse: {pulse/1000}s")
                    preview.append(f"- Recording: {recording}s")
                total_time += baseline + recording
        
        total_odors = len(selected_odors) * repeats
        preview.append(f"\nTotal cycles: {total_odors}")
        preview.append(f"Total time: {total_time}s ({total_time/60:.1f}min)")
        
        self.preview_text.setText("\n".join(preview))
    
    def get_selected_odors(self):
        """Return list of tuples with (odor_index, odor_name)"""
        return [(item.data(Qt.ItemDataRole.UserRole), item.text()) 
                for item in self.odor_list.selectedItems()]

#main block
if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    app.setApplicationName("sense: odordisp")
    app.setApplicationDisplayName("sense: odordisp")
    app.setOrganizationName("sense")
    app.setOrganizationDomain("sites.wustl.edu/sense")
    
    icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        print(f"Warning: Icon file not found at {icon_path}")
    
    window = MainWindow(loop)
    window.show_window()

    async def run_app():
        await window.start_ble_workers()

    with loop:
        loop.create_task(run_app())
        loop.run_forever()

