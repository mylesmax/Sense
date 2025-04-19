import asyncio
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QTimer
from bleak import BleakScanner, BleakClient
import json
import sys
from queue import Queue

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
        try:
            data = json.loads(json_string)
            self.data.emit(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}", file=sys.stderr)

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
        for device in devices:
            if device.name == self.device_name:
                print(f"Found target device: {device.name} [{device.address}]", file=sys.stderr)
                return device.address
        print(f"Device named '{self.device_name}' not found.", file=sys.stderr)
        self.connection_status.emit("disconnected")
        return None

    def stop(self):
        self.is_running = False

class WorkerSignals(QObject):
    success = pyqtSignal(int, str, str)
    failure = pyqtSignal(int, str, str)
