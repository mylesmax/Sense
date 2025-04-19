import SwiftUI

struct BluetoothDebugView: View {
    @EnvironmentObject var appSettings: AppSettings
    @State private var showingDataView = true  // Default to showing data view
    
    var body: some View {
        VStack(spacing: 0) {
            // Header with Connection Status and Controls
            VStack(spacing: 12) {
                // Connection Status Indicator
                HStack {
                    Image(systemName: appSettings.bluetoothManager.isConnected ? "bluetooth.connected" : "bluetooth")
                        .foregroundColor(appSettings.bluetoothManager.isConnected ? .green : .gray)
                    Text(appSettings.bluetoothManager.isConnected ? "Connected" : "Disconnected")
                        .foregroundColor(appSettings.bluetoothManager.isConnected ? .green : .gray)
                        .font(.headline)
                }
                .padding(.top)
                
                // Connection Button
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Sense")
                            .font(.nexaTrialBook(size: 28))
                            .foregroundColor(.white)
                        Text("Bluetooth Debug")
                            .font(.nexaTrialBook(size: 16))
                            .foregroundColor(.white.opacity(0.8))
                    }
                    
                    Spacer()
                    
                    Button(action: {
                        if appSettings.bluetoothManager.isConnected {
                            appSettings.bluetoothManager.disconnect()
                        } else {
                            appSettings.bluetoothManager.startScanning()
                        }
                    }) {
                        HStack(spacing: 8) {
                            Image(systemName: appSettings.bluetoothManager.isConnected ? "xmark.circle.fill" : "play.circle.fill")
                                .font(.system(size: 20))
                            Text(appSettings.bluetoothManager.isConnected ? "Disconnect" : "Start")
                                .font(.nexaTrialBook(size: 16))
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(appSettings.bluetoothManager.isConnected ? Color.red.opacity(0.8) : Color.blue.opacity(0.8))
                                .shadow(color: Color.black.opacity(0.2), radius: 5, x: 0, y: 2)
                        )
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                
                // View Toggle
                Picker("View", selection: $showingDataView) {
                    Label("Graphs", systemImage: "waveform.path.ecg").tag(true)
                    Label("Logs", systemImage: "text.alignleft").tag(false)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
            }
            .padding(.bottom, 8)
            .background(Color(.systemBackground))
            
            // Main Content
            if showingDataView {
                SensorDashboardView(sensorData: appSettings.bluetoothManager.sensorHistory)
            } else {
                LogView(logs: appSettings.bluetoothManager.debugLogs)
            }
        }
        .navigationTitle("Sensor Dashboard")
        .navigationBarItems(trailing: Button("Clear") {
            appSettings.bluetoothManager.clearLogs()
        })
    }
}

struct SensorDashboardView: View {
    let sensorData: [(timestamp: Double, values: [String: Double])]
    @State private var selectedSection = 0
    
    var body: some View {
        ScrollView {
            ScrollViewReader { proxy in
                VStack(spacing: 20) {
                    // Section Selector
                    Picker("Data Type", selection: $selectedSection) {
                        Text("All Sensors").tag(0)
                        Text("Environment").tag(1)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding(.horizontal)
                    .padding(.top, 8)
                    .id("picker")  // Fixed position anchor
                    
                    if selectedSection == 0 {
                        SensorOverviewSection(data: sensorData)
                            .id("sensors")  // Fixed position anchor
                    } else {
                        EnvironmentOverviewSection(data: sensorData)
                            .id("environment")  // Fixed position anchor
                    }
                }
                .onChange(of: selectedSection) { _ in
                    // Scroll to top when switching sections
                    withAnimation {
                        proxy.scrollTo("picker", anchor: .top)
                    }
                }
            }
            .padding(.bottom, 20)
        }
        .background(Color(.systemGroupedBackground))
    }
}

struct SensorOverviewSection: View {
    let data: [(timestamp: Double, values: [String: Double])]
    private let sensorColors: [String: Color] = [
        "n1": .blue, "n2": .green, "n3": .red, "n4": .orange, "n5": .purple,
        "n6": .yellow, "n7": .pink, "n8": .mint, "n9": .cyan, "n10": .indigo,
        "bme688_g1": .teal, "bme688_g2": .indigo
    ]
    
    var body: some View {
        VStack(spacing: 16) {
            // Current Values Grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                ForEach(1...10, id: \.self) { i in
                    let key = "n\(i)"
                    if let value = data.last?.values[key] {
                        SensorValueCard(
                            label: "Sensor \(i)",
                            value: String(format: "%.3fV", value),
                            color: sensorColors[key] ?? .gray
                        )
                    }
                }
                
                // Add BME688 gas resistance values
                if let g1Value = data.last?.values["bme688_g1"] {
                    SensorValueCard(
                        label: "BME688 G1",
                        value: String(format: "%.0f", g1Value),
                        color: sensorColors["bme688_g1"] ?? .teal
                    )
                }
                
                if let g2Value = data.last?.values["bme688_g2"] {
                    SensorValueCard(
                        label: "BME688 G2",
                        value: String(format: "%.0f", g2Value),
                        color: sensorColors["bme688_g2"] ?? .indigo
                    )
                }
            }
            .padding(.horizontal)
            
            // Combined Graph
            VStack(alignment: .leading, spacing: 8) {
                Text("Voltage Readings")
                    .font(.headline)
                    .padding(.horizontal)
                
                CombinedSensorGraph(data: data, colors: sensorColors)
                    .frame(height: 300)
                    .padding(.horizontal)
            }
            .background(Color(.systemBackground))
            .cornerRadius(15)
            .shadow(radius: 2)
            .padding(.horizontal)
            
            // BME688 Gas Resistance Graphs
            VStack(alignment: .leading, spacing: 8) {
                Text("BME688 Gas Resistance")
                    .font(.headline)
                    .padding(.horizontal)
                
                BME688ResistanceGraph(data: data)
                    .frame(height: 300)
                    .padding(.horizontal)
            }
            .background(Color(.systemBackground))
            .cornerRadius(15)
            .shadow(radius: 2)
            .padding(.horizontal)
        }
    }
}

struct EnvironmentOverviewSection: View {
    let data: [(timestamp: Double, values: [String: Double])]
    private let envColors: [String: Color] = [
        "temp": .red,
        "humidity": .blue,
        "pressure": .green
    ]
    
    var body: some View {
        VStack(spacing: 16) {
            // Current Values
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                if let lastValues = data.last?.values {
                    EnvironmentValueCard(
                        title: "Temperature",
                        value: String(format: "%.1f°C", lastValues["temp"] ?? 0),
                        icon: "thermometer",
                        color: envColors["temp"] ?? .gray
                    )
                    
                    EnvironmentValueCard(
                        title: "Humidity",
                        value: String(format: "%.1f%%", lastValues["humidity"] ?? 0),
                        icon: "humidity",
                        color: envColors["humidity"] ?? .gray
                    )
                    
                    EnvironmentValueCard(
                        title: "Pressure",
                        value: String(format: "%.0f hPa", lastValues["pressure"] ?? 0),
                        icon: "gauge",
                        color: envColors["pressure"] ?? .gray
                    )
                }
            }
            .padding(.horizontal)
            
            // Individual Graphs
            ForEach(["temp", "humidity", "pressure"], id: \.self) { key in
                EnvironmentGraphCard(
                    title: key.capitalized,
                    data: data,
                    valueKey: key,
                    color: envColors[key] ?? .gray
                )
                .padding(.horizontal)
            }
        }
    }
}

struct CombinedSensorGraph: View {
    let data: [(timestamp: Double, values: [String: Double])]
    let colors: [String: Color]
    
    private var startTime: Double { data.first?.timestamp ?? 0 }
    private var timeRange: Double {
        guard let first = data.first?.timestamp,
              let last = data.last?.timestamp else { return 1 }
        return max(last - first, 1)
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Grid
                VStack(spacing: 0) {
                    ForEach(0..<6) { i in
                        Divider()
                        Spacer()
                    }
                }
                
                // Y-axis labels
                HStack {
                    VStack(alignment: .trailing, spacing: 0) {
                        ForEach((0...5).reversed(), id: \.self) { i in
                            Text("\(i)V")
                                .font(.caption2)
                                .foregroundColor(.gray)
                                .frame(height: geometry.size.height / 6)
                        }
                    }
                    .frame(width: 30)
                    Spacer()
                }
                
                // Data lines
                ForEach(1...10, id: \.self) { i in
                    let key = "n\(i)"
                    SensorLine(
                        data: data,
                        key: key,
                        color: colors[key] ?? .gray,
                        geometry: geometry,
                        startTime: startTime,
                        timeRange: timeRange
                    )
                }
            }
        }
        .background(Color(.systemBackground))
        .cornerRadius(10)
        // Only update the graph content
        .id("graph-\(data.count)")
    }
}

struct SensorLine: View {
    let data: [(timestamp: Double, values: [String: Double])]
    let key: String
    let color: Color
    let geometry: GeometryProxy
    let startTime: Double
    let timeRange: Double
    
    private func normalizedValue(_ value: Double) -> Double {
        min(max(value, 0), 5) / 5.0
    }
    
    var body: some View {
        Path { path in
            var started = false
            for point in data {
                guard let value = point.values[key] else { continue }
                let x = geometry.size.width * (point.timestamp - startTime) / timeRange
                let y = geometry.size.height * (1 - normalizedValue(value))
                
                if !started {
                    path.move(to: CGPoint(x: x, y: y))
                    started = true
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
        }
        .stroke(color, lineWidth: 2)
    }
}

struct EnvironmentGraphCard: View {
    let title: String
    let data: [(timestamp: Double, values: [String: Double])]
    let valueKey: String
    let color: Color
    
    private var range: (min: Double, max: Double) {
        let values = data.compactMap { $0.values[valueKey] }
        guard !values.isEmpty else { return (0, 1) }
        let min = values.min() ?? 0
        let max = values.max() ?? 1
        let padding = (max - min) * 0.1
        return (min - padding, max + padding)
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
                .foregroundColor(color)
            
            GeometryReader { geometry in
                EnvironmentGraphContent(
                    data: data,
                    valueKey: valueKey,
                    color: color,
                    range: range,
                    geometry: geometry
                )
            }
            .frame(height: 200)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 2)
    }
}

struct EnvironmentGraphContent: View {
    let data: [(timestamp: Double, values: [String: Double])]
    let valueKey: String
    let color: Color
    let range: (min: Double, max: Double)
    let geometry: GeometryProxy
    
    private var startTime: Double { data.first?.timestamp ?? 0 }
    private var timeRange: Double {
        guard let first = data.first?.timestamp,
              let last = data.last?.timestamp else { return 1 }
        return max(last - first, 1)
    }
    
    var body: some View {
        ZStack {
            // Grid lines
            VStack(spacing: 0) {
                ForEach(0..<5) { i in
                    Divider()
                    Spacer()
                }
            }
            
            // Y-axis labels
            HStack {
                VStack(alignment: .trailing, spacing: 0) {
                    ForEach(0...4, id: \.self) { i in
                        let value = range.min + (range.max - range.min) * Double(i) / 4
                        Text(formatValue(value))
                            .font(.caption2)
                            .foregroundColor(.gray)
                            .frame(height: geometry.size.height / 5)
                    }
                }
                .frame(width: 50)
                Spacer()
            }
            
            // Data line
            Path { path in
                var started = false
                for point in data {
                    guard let value = point.values[valueKey] else { continue }
                    let x = geometry.size.width * (point.timestamp - startTime) / timeRange
                    let normalized = (value - range.min) / (range.max - range.min)
                    let y = geometry.size.height * (1 - normalized)
                    
                    if !started {
                        path.move(to: CGPoint(x: x, y: y))
                        started = true
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .stroke(color, lineWidth: 2)
        }
    }
    
    private func formatValue(_ value: Double) -> String {
        switch valueKey {
        case "temp": return String(format: "%.1f°", value)
        case "humidity": return String(format: "%.1f%%", value)
        case "pressure": return String(format: "%.0f", value)
        default: return String(format: "%.2f", value)
        }
    }
}

struct SensorValueCard: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.nexaTrialBook(size: 12))
                .foregroundColor(.gray)
            Text(value)
                .font(.nexaTrialBook(size: 16))
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(color.opacity(0.1))
        .cornerRadius(10)
    }
}

struct EnvironmentValueCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                Text(title)
            }
            .font(.nexaTrialBook(size: 12))
            .foregroundColor(.gray)
            
            Text(value)
                .font(.nexaTrialBook(size: 24))
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(15)
    }
}

struct LogView: View {
    let logs: [String]
    
    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading) {
                    ForEach(Array(logs.enumerated()), id: \.offset) { index, log in
                        Text(log)
                            .font(.nexaTrialBook(size: 14))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 2)
                            .id(index)
                    }
                }
                .onChange(of: logs.count) { _ in
                    if let lastIndex = logs.indices.last {
                        withAnimation {
                            proxy.scrollTo(lastIndex, anchor: .bottom)
                        }
                    }
                }
            }
            .background(Color(.systemGroupedBackground))
        }
    }
}

struct BME688ResistanceGraph: View {
    let data: [(timestamp: Double, values: [String: Double])]
    
    private var startTime: Double { data.first?.timestamp ?? 0 }
    private var timeRange: Double {
        guard let first = data.first?.timestamp,
              let last = data.last?.timestamp else { return 1 }
        return max(last - first, 1)
    }
    
    private var g1Range: (min: Double, max: Double) {
        let values = data.compactMap { $0.values["bme688_g1"] }
        guard !values.isEmpty else { return (0, 1) }
        let min = values.min() ?? 0
        let max = values.max() ?? 1
        let padding = (max - min) * 0.1
        return (min - padding, max + padding)
    }
    
    private var g2Range: (min: Double, max: Double) {
        let values = data.compactMap { $0.values["bme688_g2"] }
        guard !values.isEmpty else { return (0, 1) }
        let min = values.min() ?? 0
        let max = values.max() ?? 1
        let padding = (max - min) * 0.1
        return (min - padding, max + padding)
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Grid
                VStack(spacing: 0) {
                    ForEach(0..<6) { i in
                        Divider()
                        Spacer()
                    }
                }
                
                // Y-axis labels
                HStack {
                    VStack(alignment: .trailing, spacing: 0) {
                        ForEach(0...5, id: \.self) { i in
                            let g1Min = g1Range.min
                            let g1Max = g1Range.max
                            let value = g1Min + (g1Max - g1Min) * Double(i) / 5.0
                            Text("\(Int(value))")
                                .font(.caption2)
                                .foregroundColor(.teal)
                                .frame(height: geometry.size.height / 6)
                        }
                    }
                    .frame(width: 80)
                    Spacer()
                    
                    // Right Y-axis for G2
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(0...5, id: \.self) { i in
                            let g2Min = g2Range.min
                            let g2Max = g2Range.max
                            let value = g2Min + (g2Max - g2Min) * Double(i) / 5.0
                            Text("\(Int(value))")
                                .font(.caption2)
                                .foregroundColor(.indigo)
                                .frame(height: geometry.size.height / 6)
                        }
                    }
                    .frame(width: 80)
                }
                
                // G1 Line (teal)
                BME688Line(
                    data: data,
                    key: "bme688_g1",
                    color: .teal,
                    geometry: geometry,
                    startTime: startTime,
                    timeRange: timeRange,
                    valueRange: g1Range
                )
                
                // G2 Line (indigo)
                BME688Line(
                    data: data,
                    key: "bme688_g2",
                    color: .indigo,
                    geometry: geometry,
                    startTime: startTime,
                    timeRange: timeRange,
                    valueRange: g2Range
                )
                
                // Legend
                VStack {
                    Spacer()
                    HStack(spacing: 20) {
                        HStack {
                            Rectangle()
                                .fill(Color.teal)
                                .frame(width: 20, height: 4)
                            Text("G1")
                                .font(.caption)
                                .foregroundColor(.teal)
                        }
                        HStack {
                            Rectangle()
                                .fill(Color.indigo)
                                .frame(width: 20, height: 4)
                            Text("G2")
                                .font(.caption)
                                .foregroundColor(.indigo)
                        }
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 4)
                }
            }
        }
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .id("bme688-graph-\(data.count)")
    }
}

struct BME688Line: View {
    let data: [(timestamp: Double, values: [String: Double])]
    let key: String
    let color: Color
    let geometry: GeometryProxy
    let startTime: Double
    let timeRange: Double
    let valueRange: (min: Double, max: Double)
    
    private func normalizedValue(_ value: Double) -> Double {
        let range = valueRange.max - valueRange.min
        if range == 0 { return 0.5 }
        return (value - valueRange.min) / range
    }
    
    var body: some View {
        Path { path in
            var started = false
            for point in data {
                guard let value = point.values[key] else { continue }
                let x = geometry.size.width * (point.timestamp - startTime) / timeRange
                let y = geometry.size.height * (1 - normalizedValue(value))
                
                if !started {
                    path.move(to: CGPoint(x: x, y: y))
                    started = true
                } else {
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
        }
        .stroke(color, lineWidth: 2)
    }
}

struct BluetoothDebugView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            BluetoothDebugView()
                .environmentObject(AppSettings())
        }
    }
} 