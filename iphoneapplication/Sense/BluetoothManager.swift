import CoreBluetooth
import SwiftUI

class BluetoothManager: NSObject, ObservableObject {
    @Published var isConnected = false {
        didSet {
            print("🔵 BLE: isConnected changed from \(oldValue) to \(isConnected)")
        }
    }
    @Published var debugLogs: [String] = []
    @Published private(set) var sensorData: [String: Double] = [:]  // Current sensor values
    @Published private(set) var sensorHistory: [(timestamp: Double, values: [String: Double])] = []  // Historical data
    @Published var batteryLevel: Int = 0  // New variable for battery level (0-100%)
    
    // Debugging flag - set to false to reduce console output
    private let enableDebug = false
    
    private let maxHistoryPoints = 100  // Keep last 100 data points
    private var centralManager: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var isReceivingPacket = false  // Track if we're in the middle of receiving a packet
    private var currentJson = ""  // Buffer for accumulating JSON data
    private var debugPacketCount = 0  // For debugging packet assembly
    
    // Update device name and UUIDs
    let deviceName = "03senseV3"
    let initialDeviceName = "03" // New device name for first-time connections
    let serviceUUID = CBUUID(string: "2cc12ee8-c5b6-4d7f-a3de-9c793653f271")
    let characteristicUUID = CBUUID(string: "15216e4f-bf54-4482-8a91-74a92ccfeb37")
    
    // Battery service and characteristic UUIDs (standard BLE battery service)
    let batteryServiceUUID = CBUUID(string: "180F")
    let batteryCharacteristicUUID = CBUUID(string: "2A19")
    
    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: .main)
    }
    
    func addLog(_ message: String) {
        if enableDebug {
            print("🔵 BLE: \(message)")  // Print to Xcode console
        }
        
        DispatchQueue.main.async {
            let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
            self.debugLogs.append("[\(timestamp)] \(message)")
        }
    }
    
    func clearLogs() {
        debugLogs.removeAll()
    }
    
    func startScanning() {
        guard let centralManager = centralManager, centralManager.state == .poweredOn else {
            addLog("❌ Bluetooth not ready")
            return
        }
        
        // Clear previous connection
        if isConnected {
            disconnect()
        }
        
        // Reset connection state
        DispatchQueue.main.async {
            self.isConnected = false
            self.objectWillChange.send()
        }
        
        addLog("🔍 Starting scan for device...")
        addLog("  Target names: \(deviceName) or \(initialDeviceName)")
        addLog("  Scanning for nearby devices...")
        
        // Start scanning for the specific service
        centralManager.scanForPeripherals(withServices: nil, options: nil)
    }
    
    func disconnect() {
        guard let peripheral = peripheral, let centralManager = centralManager else {
            return
        }
        
        centralManager.cancelPeripheralConnection(peripheral)
        addLog("🔌 Disconnected from device")
        
        DispatchQueue.main.async {
            self.isConnected = false
            self.objectWillChange.send()  // Explicitly notify observers
        }
    }
    
    func clearData() {
        sensorData.removeAll()
        sensorHistory.removeAll()
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic,
                   error: Error?) {
        if let error = error {
            addLog("❌ Update failed: \(error.localizedDescription)")
            return
        }
        
        // Handle battery characteristic
        if characteristic.uuid == batteryCharacteristicUUID, let data = characteristic.value {
            if let batteryLevel = data.first { // Battery level is a single byte (0-100)
                DispatchQueue.main.async {
                    self.batteryLevel = Int(batteryLevel)
                    self.objectWillChange.send()
                }
                addLog("🔋 Battery level updated: \(batteryLevel)%")
            }
            return
        }
        
        // Handle main data characteristic
        guard let data = characteristic.value,
              let str = String(data: data, encoding: .utf8) else {
            addLog("❌ Received invalid data")
            if let data = characteristic.value {
                addLog("  Raw data: \(data.map { String(format: "%02X", $0) }.joined())")
            }
            return
        }
        
        let value = str.trimmingCharacters(in: .whitespacesAndNewlines)
        addLog("📥 Raw packet: \(value)")
        
        // Handle START marker
        if value.contains("START") {
            isReceivingPacket = true
            currentJson = ""
            debugPacketCount = 0
            addLog("📥 Started new packet")
            return
        }
        
        // Handle END marker
        if value.contains("END") {
            if isReceivingPacket && !currentJson.isEmpty {
                debugPacketCount += 1
                addLog("📥 Received final chunk (#\(debugPacketCount))")
                
                // Clean up the JSON string
                var cleanJson = currentJson.trimmingCharacters(in: .whitespacesAndNewlines)
                
                // Remove any trailing commas
                if cleanJson.hasSuffix(",") {
                    cleanJson.removeLast()
                }
                
                // Ensure it's a complete JSON object
                if !cleanJson.hasPrefix("{") {
                    cleanJson = "{\(cleanJson)"
                }
                if !cleanJson.hasSuffix("}") {
                    cleanJson += "}"
                }
                
                addLog("📦 Processing complete JSON: \(cleanJson)")
                processJsonData(cleanJson)
            } else {
                addLog("⚠️ Received END with no data")
            }
            isReceivingPacket = false
            currentJson = ""
            return
        }
        
        // If we're receiving a packet, accumulate the data
        if isReceivingPacket {
            debugPacketCount += 1
            
            // If this chunk starts with a JSON opening brace, it's the start of the actual JSON
            if value.hasPrefix("{") {
                currentJson = value
            } else {
                // Otherwise append it to our existing JSON
                currentJson += value
            }
            
            addLog("📥 Chunk #\(debugPacketCount) received, current JSON length: \(currentJson.count)")
        }
    }
    
    private func processJsonData(_ jsonString: String) {
        do {
            guard let jsonData = jsonString.data(using: .utf8),
                  let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
                addLog("❌ Failed to parse JSON: \(jsonString)")
                return
            }
            
            // Extract all values
            guard let timestamp = (json["x"] as? NSNumber)?.doubleValue else {
                addLog("❌ Missing timestamp value")
                return
            }
            
            var values: [String: Double] = ["x": timestamp]
            
            // Process sensor values n1-n10
            for i in 1...10 {
                if let value = (json["n\(i)"] as? NSNumber)?.doubleValue {
                    values["n\(i)"] = value
                }
            }
            
            // Process g1 and g2 values directly from the Bluetooth stream
            // IMPORTANT: Store the raw g1/g2 values with BOTH original keys AND bme688_* keys
            // This ensures both naming conventions are available for the app to use
            if let g1Value = (json["g1"] as? NSNumber)?.doubleValue {
                values["g1"] = g1Value
                values["bme688_g1"] = g1Value
                values["gas_resistance_1"] = g1Value
                addLog("📊 Gas resistance 1: \(g1Value)")
            }
            else if let g1String = json["g1"] as? String, let g1Value = Double(g1String) {
                values["g1"] = g1Value
                values["bme688_g1"] = g1Value
                values["gas_resistance_1"] = g1Value
                addLog("📊 Gas resistance 1: \(g1Value)")
            }
            
            if let g2Value = (json["g2"] as? NSNumber)?.doubleValue {
                values["g2"] = g2Value
                values["bme688_g2"] = g2Value
                values["gas_resistance_2"] = g2Value
                addLog("📊 Gas resistance 2: \(g2Value)")
            }
            else if let g2String = json["g2"] as? String, let g2Value = Double(g2String) {
                values["g2"] = g2Value
                values["bme688_g2"] = g2Value
                values["gas_resistance_2"] = g2Value
                addLog("📊 Gas resistance 2: \(g2Value)")
            }
            
            // Process environmental values - Store with BOTH original keys AND expanded names
            // This matches the Python code which expects keys 't', 'p', 'h'
            let envMapping = [
                "t": ["temp", "t"],
                "h": ["humidity", "h"],
                "p": ["pressure", "p"]
            ]
            
            for (key, newKeys) in envMapping {
                if let value = (json[key] as? NSNumber)?.doubleValue {
                    // Store with both naming conventions
                    for newKey in newKeys {
                        values[newKey] = value
                    }
                    addLog("📊 Env sensor \(key): \(value)")
                }
            }
            
            // Remove gas value completely instead of setting to negative
            DispatchQueue.main.async {
                // Remove gas entry from sensorData
                self.sensorData.removeValue(forKey: "gas")
                
                // Update current values with all the collected data points
                self.sensorData = values
                
                // Make sure all keys are available regardless of naming convention
                // This ensures consistent data whether accessed via n1/g1 or through bme688_g1
                
                // For g1/g2, make sure both naming conventions are available
                if let g1 = values["g1"] {
                    self.sensorData["g1"] = g1
                    self.sensorData["bme688_g1"] = g1
                } else if let g1 = values["bme688_g1"] {
                    self.sensorData["g1"] = g1
                }
                
                if let g2 = values["g2"] {
                    self.sensorData["g2"] = g2
                    self.sensorData["bme688_g2"] = g2
                } else if let g2 = values["bme688_g2"] {
                    self.sensorData["g2"] = g2
                }
                
                // For environmental sensors, make sure both naming conventions are available
                let envMapping = [
                    ("t", "temp"),
                    ("p", "pressure"),
                    ("h", "humidity")
                ]
                
                for (shortKey, longKey) in envMapping {
                    if let value = values[shortKey] {
                        self.sensorData[shortKey] = value
                        self.sensorData[longKey] = value
                    } else if let value = values[longKey] {
                        self.sensorData[shortKey] = value
                    }
                }
                
                // Add to history
                self.sensorHistory.append((timestamp: timestamp, values: values))
                
                // Trim history if needed
                if self.sensorHistory.count > self.maxHistoryPoints {
                    self.sensorHistory.removeFirst()
                }
                
                // Explicitly notify observers of the data update
                self.objectWillChange.send()
                
                // Log history size for debugging
                self.addLog("📈 History size: \(self.sensorHistory.count) points")
            }
            
            // Process battery level specifically
            // Check for b_p (battery percentage) first, then fall back to b if needed
            if let batteryPercentage = (json["b_p"] as? NSNumber)?.doubleValue {
                values["battery"] = batteryPercentage
                
                // Update the dedicated batteryLevel variable
                let batteryValue = Int(round(batteryPercentage))
                DispatchQueue.main.async {
                    self.batteryLevel = batteryValue
                    self.objectWillChange.send()
                }
                addLog("🔋 Battery level from b_p: \(batteryValue)%")
            } else if let batteryValue = (json["b"] as? NSNumber)?.doubleValue {
                values["battery"] = batteryValue
                
                // Update the dedicated batteryLevel variable using the original "b" key
                let batteryIntValue = Int(batteryValue)
                DispatchQueue.main.async {
                    self.batteryLevel = batteryIntValue
                    self.objectWillChange.send()
                }
                addLog("🔋 Battery level from b: \(batteryIntValue)%")
            }
            
            // Also check for b_i (battery indicator) and other possible battery information
            if let batteryIndicator = (json["b_i"] as? NSNumber)?.doubleValue {
                values["battery_indicator"] = batteryIndicator
                addLog("🔋 Battery indicator value: \(batteryIndicator)")
            }
        } catch {
            addLog("❌ Parse error: \(error.localizedDescription)")
            addLog("❌ Failed JSON: \(jsonString)")
        }
    }
}

extension BluetoothManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            addLog("📱 Bluetooth state changed: Powered On")
            addLog("✅ Bluetooth is ready to use")
        case .poweredOff:
            addLog("📱 Bluetooth state changed: Powered Off")
            addLog("❌ Please turn on Bluetooth")
        case .resetting:
            addLog("📱 Bluetooth state changed: Resetting")
        case .unauthorized:
            addLog("📱 Bluetooth state changed: Unauthorized")
            addLog("❌ Please authorize Bluetooth in Settings")
        case .unsupported:
            addLog("📱 Bluetooth state changed: Unsupported")
            addLog("❌ This device does not support Bluetooth")
        case .unknown:
            addLog("📱 Bluetooth state changed: Unknown")
        @unknown default:
            addLog("📱 Bluetooth state changed: Unknown state")
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        // Check if this is our target device by name
        if let name = peripheral.name {
            addLog("📱 Found device: \(name)")
            addLog("  RSSI: \(RSSI)dBm")
            
            if name == deviceName || name == initialDeviceName {
                addLog("✅ Found our target device!")
                addLog("  Name: \(name)")
                addLog("  Device ID: \(peripheral.identifier.uuidString)")
                
                // Log advertisement data
                for (key, value) in advertisementData {
                    addLog("  \(key): \(value)")
                }
                
                // Stop scanning
                central.stopScan()
                addLog("🔄 Attempting to connect...")
                
                // Connect to the peripheral
                self.peripheral = peripheral
                central.connect(peripheral, options: nil)
            }
        }
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        addLog("✅ Connected successfully!")
        
        DispatchQueue.main.async {
            self.isConnected = true
            self.objectWillChange.send()  // Explicitly notify observers
        }
        
        // Set up peripheral delegate
        peripheral.delegate = self
        
        // Discover services
        addLog("🔍 Discovering services...")
        peripheral.discoverServices([serviceUUID])
    }
    
    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        if let error = error {
            addLog("❌ Failed to connect: \(error.localizedDescription)")
        } else {
            addLog("❌ Failed to connect")
        }
        
        DispatchQueue.main.async {
            self.isConnected = false
            self.objectWillChange.send()  // Explicitly notify observers
        }
    }
    
    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        if let error = error {
            addLog("❌ Disconnected with error: \(error.localizedDescription)")
        } else {
            addLog("🔌 Disconnected from peripheral")
        }
        
        DispatchQueue.main.async {
            self.isConnected = false
            self.objectWillChange.send()  // Explicitly notify observers
        }
        
        // Restart scanning
        central.scanForPeripherals(withServices: nil, options: nil)
    }
}

extension BluetoothManager: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        if let error = error {
            addLog("❌ Service discovery failed: \(error.localizedDescription)")
            return
        }
        
        guard let services = peripheral.services else {
            addLog("❌ No services found")
            return
        }
        
        addLog("✅ Found \(services.count) services:")
        for service in services {
            addLog("  Service: \(service.uuid)")
            
            if service.uuid == serviceUUID {
                // Discover characteristics for our main data service
                peripheral.discoverCharacteristics([characteristicUUID], for: service)
            } else if service.uuid == batteryServiceUUID {
                // Discover characteristics for battery service
                peripheral.discoverCharacteristics([batteryCharacteristicUUID], for: service)
                addLog("  🔋 Found battery service")
            } else {
                // For other services, discover all characteristics
                peripheral.discoverCharacteristics(nil, for: service)
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        if let error = error {
            addLog("❌ Characteristic discovery failed: \(error.localizedDescription)")
            return
        }
        
        guard let characteristics = service.characteristics else {
            addLog("❌ No characteristics found")
            return
        }
        
        addLog("✅ Found \(characteristics.count) characteristics for \(service.uuid):")
        for characteristic in characteristics {
            addLog("  Characteristic: \(characteristic.uuid)")
            addLog("    Properties: \(characteristicPropertiesString(characteristic.properties))")
            
            // Handle main data characteristic
            if characteristic.uuid == characteristicUUID {
                addLog("    ✅ Found target characteristic")
                peripheral.setNotifyValue(true, for: characteristic)
                addLog("    ✅ Enabled notifications")
            }
            
            // Handle battery level characteristic
            if characteristic.uuid == batteryCharacteristicUUID {
                addLog("    🔋 Found battery characteristic")
                
                // Enable notifications for battery level changes
                if characteristic.properties.contains(.notify) {
                    peripheral.setNotifyValue(true, for: characteristic)
                    addLog("    🔋 Enabled battery notifications")
                }
                
                // Read the current battery level
                if characteristic.properties.contains(.read) {
                    peripheral.readValue(for: characteristic)
                    addLog("    🔋 Reading battery level")
                }
            }
        }
    }
    
    // Convert characteristic properties to a readable string
    private func characteristicPropertiesString(_ properties: CBCharacteristicProperties) -> String {
        var props: [String] = []
        
        if properties.contains(.broadcast) { props.append("Broadcast") }
        if properties.contains(.read) { props.append("Read") }
        if properties.contains(.writeWithoutResponse) { props.append("Write Without Response") }
        if properties.contains(.write) { props.append("Write") }
        if properties.contains(.notify) { props.append("Notify") }
        if properties.contains(.indicate) { props.append("Indicate") }
        if properties.contains(.authenticatedSignedWrites) { props.append("Authenticated Signed Writes") }
        
        return props.joined(separator: ", ")
    }
} 
