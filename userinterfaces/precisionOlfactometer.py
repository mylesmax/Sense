import serial
import time
import glob
import subprocess
import re
import sys

def setup_permissions():
    """Set up permissions for serial port access"""
    try:
        # Check if user is in dialout group
        groups_output = subprocess.check_output(['groups'], text=True)
        if 'dialout' not in groups_output:
            print("Adding user to dialout group...")
            subprocess.run(['pkexec', 'usermod', '-a', '-G', 'dialout', subprocess.check_output(['whoami'], text=True).strip()], check=True)
            print("Added to dialout group. Please log out and log back in for changes to take effect.")
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up permissions: {e}")
        return False

def find_serial_port():
    try:
        # Run lspci to find the Asix serial controller
        lspci_output = subprocess.check_output(['lspci'], text=True)
        for line in lspci_output.split('\n'):
            if 'Serial controller: Asix' in line:
                pci_id = line.split()[0]
                print(f"Found Asix serial controller at PCI ID: {pci_id}")
                
                # Use pkexec to run dmesg with elevated privileges
                try:
                    dmesg_output = subprocess.check_output(['pkexec', 'dmesg'], text=True)
                except subprocess.CalledProcessError:
                    # Fallback to regular dmesg if pkexec fails
                    try:
                        dmesg_output = subprocess.check_output(['dmesg'], text=True)
                    except subprocess.CalledProcessError:
                        print("Could not access dmesg output")
                        return '/dev/ttyS4'
                
                for dmesg_line in dmesg_output.split('\n'):
                    if pci_id in dmesg_line and 'ttyS' in dmesg_line:
                        match = re.search(r'ttyS\d+', dmesg_line)
                        if match:
                            port = f"/dev/{match.group(0)}"
                            print(f"Found serial port: {port}")
                            return port
                
                print("Serial controller found but no ttyS device detected, defaulting to ttyS4")
                return '/dev/ttyS4'
    except subprocess.CalledProcessError as e:
        print(f"Error running system commands: {e}")
    except Exception as e:
        print(f"Unexpected error in port detection: {e}")
    
    print("Could not detect serial port automatically, defaulting to ttyS4")
    return '/dev/ttyS4'

def list_available_ports():
    usb_ports = glob.glob('/dev/ttyUSB*')
    serial_ports = glob.glob('/dev/ttyS*')
    acm_ports = glob.glob('/dev/ttyACM*')
    
    all_ports = usb_ports + serial_ports + acm_ports
    return sorted(all_ports)

class MKS946Controller:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            port = find_serial_port()
            cls._instance = cls(port=port)
        return cls._instance
    
    def __init__(self, port=None, baudrate=9600, timeout=1):
        if port is None:
            port = find_serial_port()
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            if not self.check_baud_rate():
                raise Exception("Failed to verify baud rate")
        except serial.SerialException as e:
            print(f"Failed to open serial port {port}: {e}")
            raise
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def check_baud_rate(self):
        """Check baud rate using @254BR?;FF command as first communication"""
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        cmd = "@254BR?;FF\r"
        self.ser.write(cmd.encode())
        time.sleep(0.5)
        
        if self.ser.in_waiting:
            response = self.ser.read(self.ser.in_waiting)
            try:
                decoded = response.decode('ascii', errors='ignore')
                if "9600" in decoded:
                    return True
            except Exception:
                pass
        return False
    
    def send_command(self, command):
        """Send a command and return the response"""
        self.ser.write(command.encode())
        time.sleep(0.5)
        
        if self.ser.in_waiting:
            response = self.ser.read(self.ser.in_waiting)
            try:
                return response.decode('ascii', errors='ignore').strip()
            except Exception:
                return None
        return None
    
    def read_pressure(self, channel):
        """Read pressure from specified channel using @254PRx?;FF format"""
        cmd = f"@254PR{channel}?;FF\r"
        response = self.send_command(cmd)
        if response:
            return response
        return None
    
    def scan_all_channels(self):
        """Scan all pressure channels"""
        readings = {}
        for channel in range(1, 7):
            print(f"\nReading pressure on channel {channel}...")
            reading = self.read_pressure(channel)
            if reading:
                readings[channel] = reading
        return readings
    
    def close(self):
        self.ser.close()

def parse_response(response):
    """Parse response between @253ACK and ;FF"""
    if not response:
        return None
    
    try:
        match = re.search(r'@253ACK(.*?);FF', response)
        if not match:
            return None
        
        value = match.group(1)
        if value == "MISCONN":
            return None
        
        try:
            return float(value)
        except ValueError:
            return None
            
    except Exception:
        return None

def queryPressure(n):
    """Query pressure on channel n (1-6)"""
    if not 1 <= n <= 6:
        return "hey, channel must be 1-6!"
    
    try:
        # Use existing controller instance
        controller = MKS946Controller.get_instance()
        response = controller.read_pressure(n)
        return parse_response(response)
    except Exception as e:
        return f"oof, something went wrong: {e}"

def scan_ports():
    print("Looking for MKS 946 on Serial connection...")
    port = find_serial_port()
    print(f"Found device on {port}")
    print("Using device settings:")
    print("- Baud rate: 9600")
    print("- Communication: RS232")
    
    try:
        print(f"\nOpening {port}...")
        controller = MKS946Controller(port=port)
        
        print("\nReading all channels...")
        readings = controller.scan_all_channels()
        
        print("\nPressure Readings:")
        for channel, reading in readings.items():
            parsed = parse_response(reading)
            print(f"Channel {channel}: {parsed}")
        
        controller.close()
        
    except serial.SerialException as e:
        print(f"Error opening port: {e}")
        print("\nPlease check:")
        print("1. The serial device is properly connected")
        print("2. You have permission to access the port (try running with sudo)")
        print("3. No other program is using the port")
    except Exception as e:
        print(f"Error: {e}")

def test_mks946():
    print("MKS 946 Basic Communication Test")
    print("--------------------------------")
    print("Settings:")
    print("- Address: 253")
    print("- Baud rate: 9600")
    print("- Parity: Odd")
    print("- Stop bits: 1")
    
    ports = ['/dev/ttyS0', '/dev/ttyS1', '/dev/ttyS4', '/dev/ttyS5']
    
    for port in ports:
        try:
            print(f"\nTrying port {port}...")
            ser = serial.Serial(
                port=port,
                baudrate=9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_ODD,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            
            print(f"Port opened successfully:")
            print(f"- Baudrate: {ser.baudrate}")
            print(f"- Bytesize: {ser.bytesize}")
            print(f"- Parity: {ser.parity}")
            print(f"- Stopbits: {ser.stopbits}")
            
            print("Clearing buffers...")
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(1)
            
            commands = [
                "@253AD?;FF",
                "@253VER?;FF",
                "AD?;FF",
                "VER?;FF",
            ]
            
            for cmd in commands:
                print(f"\nSending: {cmd!r}")
                ser.write(cmd.encode())
                print(f"Sent bytes: {cmd.encode().hex()}")
                time.sleep(1)
                
                if ser.in_waiting:
                    response = ser.read(ser.in_waiting)
                    print(f"Response (hex): {response.hex()}")
                    try:
                        text = response.decode('ascii')
                        print(f"Response (text): {text!r}")
                    except:
                        print("Could not decode response as ASCII")
                else:
                    print("No response received")
                    response = ser.read(100)
                    if response:
                        print(f"Forced read (hex): {response.hex()}")
                
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                time.sleep(0.5)
            
            ser.close()
            print(f"\nPort {port} closed")
            
        except serial.SerialException as e:
            print(f"Error on {port}: {e}")
            continue
        except Exception as e:
            print(f"Error on {port}: {e}")
            continue
    
    print("\nFinished testing all ports")

def display_pressures():
    """Continuously display pressures in a clean format"""
    try:
        controller = MKS946Controller.get_instance()
        
        while True:
            # Get all pressures
            readings = []
            for i in range(1, 7):
                response = controller.read_pressure(i)
                value = parse_response(response)
                if isinstance(value, float):
                    readings.append(f"{value:.1f}")
                else:
                    readings.append("---")
            
            # Display in single line
            print(f"CH1: {readings[0]} | CH2: {readings[1]} | CH3: {readings[2]} | CH4: {readings[3]} | CH5: {readings[4]} | CH6: {readings[5]}")
            
            # Wait before next update
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        controller.close()
    except Exception as e:
        print(f"Error: {e}")
        try:
            controller.close()
        except:
            pass

if __name__ == "__main__":
    try:
        # Check permissions first
        if not setup_permissions():
            print("\nPlease restart the program after logging back in.")
            sys.exit(1)
        
        # Start continuous monitoring
        display_pressures()
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        try:
            MKS946Controller._instance.close()
        except:
            pass 