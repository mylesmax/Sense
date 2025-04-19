import SwiftUI
import Combine

struct SensorRecording: View {
    var onStart: () -> Void
    var onEscape: () -> Void
    
    @EnvironmentObject var appSettings: AppSettings
    @State private var totalTimeRemaining: Int = 50 // Total 50 seconds of recording
    @State private var currentPhaseTimeRemaining: Int = 10 // 10 seconds per phase
    @State private var timerFinished = false
    @State private var timer: Timer?
    @State private var activeQuadrant: Int = -1 // -1 means no active quadrant
    @State private var recordingPhase: Int = 0 // 0=baseline, 1-4=samples
    @State private var isRecording: Bool = false // Track if recording is in progress
    @State private var showingInstructions: Bool = true // Track if we're showing instructions
    @State private var instructionPage: Int = 1 // Track which instruction page we're showing
    @State private var lastSampleTimestamp: Date? // When was the last sample taken
    @State private var apiService = APIService() // API service for sending data
    @State private var cancellables = Set<AnyCancellable>() // For managing API requests
    @State private var animatedQuadrant: Int = 0 // For the instructional animation
    @State private var quadrantAnimationTimer: Timer? // Timer for quadrant animation
    @State private var phaseReadingsCount: Int = 0 // Track readings count in current phase
    
    private let phaseLabels = ["Baseline", "Sample 1", "Sample 2", "Sample 3", "Sample 4"]
    private let quadrantColors: [Color] = [.blue, .green, .orange, .purple]
    
    // Sample rate to match Python's exact timing (0.36 seconds)
    private let sampleInterval = 0.36
    // Fixed duration for each phase
    private let phaseDuration = 10
    
    // Use Font extension for consistency
    private let titleFont = Font.nexaFont(size: 32)
    private let subtitleFont = Font.nexaFont(size: 20)
    private let bodyFont = Font.nexaFont(size: 16)
    private let buttonFont = Font.nexaFont(size: 18)
    private let smallFont = Font.nexaFont(size: 14)
    private let numberFont = Font.nexaFont(size: 24)
    
    init(onStart: @escaping () -> Void, onEscape: @escaping () -> Void) {
        self.onStart = onStart
        self.onEscape = onEscape
    }
    
    var body: some View {
        VStack {
            if showingInstructions {
                // Instruction Screens
                if instructionPage == 1 {
                    VStack(spacing: 15) {
                        Text("Recording Process")
                            .font(titleFont)
                            .padding(.top, 40)
                            .padding(.bottom, 10)
                            .frame(maxWidth: .infinity, alignment: .center)
                        
                        Text("This is a 50-second continuous recording:")
                            .font(subtitleFont)
                            .padding(.bottom, 5)
                            .frame(maxWidth: .infinity, alignment: .center)
                        
                        VStack(alignment: .leading, spacing: 12) {
                            HStack(spacing: 12) {
                                Circle()
                                    .fill(Color.white)
                                    .frame(width: 26, height: 26)
                                    .overlay(Text("1").foregroundColor(.black).font(smallFont))
                                
                                Text("10 sec: Baseline")
                                    .font(bodyFont)
                            }
                            
                            ForEach(1..<5) { i in
                                HStack(spacing: 12) {
                                    Circle()
                                        .fill(i == animatedQuadrant ? quadrantColors[i-1] : Color.gray.opacity(0.3))
                                        .frame(width: 26, height: 26)
                                        .overlay(Text("\(i+1)").foregroundColor(.white).font(smallFont))
                                    
                                    Text("10 sec: Quadrant \(i)")
                                        .font(bodyFont)
                                        .foregroundColor(i == animatedQuadrant ? .white : .gray)
                                }
                                .animation(.easeInOut(duration: 0.3), value: animatedQuadrant)
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.bottom, 20)
                        .frame(maxWidth: .infinity)
                        
                        Button(action: {
                            instructionPage = 2
                        }) {
                            HStack {
                                Text("Next")
                                    .font(buttonFont)
                                Image(systemName: "arrow.right")
                            }
                            .padding()
                            .frame(width: 160)
                            .foregroundColor(.white)
                            .background(Color.blue)
                            .cornerRadius(10)
                        }
                        .frame(maxWidth: .infinity, alignment: .center)
                        
                        Spacer()
                        
                        // Add test button for debugging only
                        #if DEBUG
                        Button(action: testAPIConnection) {
                            Text("Test API Connection")
                                .font(smallFont)
                                .foregroundColor(.gray)
                        }
                        .padding(.bottom, 10)
                        .frame(maxWidth: .infinity, alignment: .center)
                        #endif
                    }
                    .padding()
                    .onAppear {
                        print("First instruction page appeared")
                        DispatchQueue.main.async {
                            stopQuadrantAnimation()
                            startQuadrantAnimation()
                        }
                    }
                    .onDisappear {
                        print("First instruction page disappeared")
                    }
                } else {
                    VStack(spacing: 20) {
                        Text("Quadrant Movement")
                            .font(titleFont)
                            .padding(.top, 40)
                            .padding(.bottom, 10)
                            .frame(maxWidth: .infinity, alignment: .center)
                        
                        Text("Move the sensor in real-time as the active quadrant changes:")
                            .font(subtitleFont)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .center)
                        
                        // Quadrant Circle with movement indicators
                        ZStack {
                            QuadrantCircleView(
                                activeQuadrant: Binding<Int>(
                                    get: { self.animatedQuadrant - 1 }, // Convert 1-4 to 0-3 for QuadrantCircleView
                                    set: { _ in }
                                ),
                                inactiveColor: Color.gray.opacity(0.3),
                                activeColors: quadrantColors,
                                size: 220
                            )
                            
                            // Numbered indicators for each quadrant (1=top right, clockwise)
                            VStack {
                                HStack {
                                    Text("4")
                                        .font(numberFont)
                                        .frame(width: 90, height: 90)
                                        .foregroundColor(animatedQuadrant == 4 ? .white : .gray)
                                    
                                    Text("1")
                                        .font(numberFont)
                                        .frame(width: 90, height: 90)
                                        .foregroundColor(animatedQuadrant == 1 ? .white : .gray)
                                }
                                
                                HStack {
                                    Text("3")
                                        .font(numberFont)
                                        .frame(width: 90, height: 90)
                                        .foregroundColor(animatedQuadrant == 3 ? .white : .gray)
                                    
                                    Text("2")
                                        .font(numberFont)
                                        .frame(width: 90, height: 90)
                                        .foregroundColor(animatedQuadrant == 2 ? .white : .gray)
                                }
                            }
                        }
                        .id("quadrant-circle-\(animatedQuadrant)") // Force redraw when quadrant changes
                        .padding(.vertical, 20)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .onAppear {
                            print("Second instruction page appeared")
                            DispatchQueue.main.async {
                                stopQuadrantAnimation()
                                startQuadrantAnimation()
                            }
                        }
                        .onDisappear {
                            print("Second instruction page disappeared")
                        }
                        
                        Text("The recording will automatically progress through all phases without pausing.")
                            .font(bodyFont)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                            .padding(.bottom, 10)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .center)
                        
                        Button(action: {
                            withAnimation {
                                showingInstructions = false
                                // Stop animation timer before starting the recording
                                stopQuadrantAnimation()
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                    startContinuousRecording() // Start the continuous recording
                                }
                            }
                        }) {
                            Text("Begin 50-Second Recording")
                                .font(buttonFont)
                                .padding()
                                .foregroundColor(.white)
                                .background(Color.blue)
                                .cornerRadius(10)
                        }
                        .padding(.bottom, 20)
                        .frame(maxWidth: .infinity, alignment: .center)
                        
                        Button(action: {
                            instructionPage = 1
                        }) {
                            HStack {
                                Image(systemName: "arrow.left")
                                Text("Previous")
                                    .font(smallFont)
                            }
                            .foregroundColor(.gray)
                        }
                        .frame(maxWidth: .infinity, alignment: .center)
                        
                        Spacer()
                    }
                    .padding()
                }
            } else {
                // Recording Screen - Keep existing implementation unchanged
                VStack(spacing: 10) {
                    // Main instruction text
                    Text(recordingPhase == 0 ? 
                        "Hold sensor in ambient air" : 
                        "Hold sensor above quadrant \(recordingPhase)")
                        .font(titleFont)
                        .multilineTextAlignment(.center)
                        .padding(.top, 30)
                        .padding(.horizontal)
                        .frame(maxWidth: .infinity, alignment: .center)
                    
                    // Quadrant Circle View (centered)
                    QuadrantCircleView(
                        activeQuadrant: $activeQuadrant,
                        inactiveColor: Color.gray.opacity(0.3),
                        activeColors: quadrantColors,
                        size: 200
                    )
                    .padding(.vertical, 20)
                    .frame(maxWidth: .infinity, alignment: .center)
                    
                    // Current recording phase label
                    Text("Phase: \(phaseLabels[recordingPhase])")
                        .font(subtitleFont)
                        .foregroundColor(recordingPhase == 0 ? .white : quadrantColors[activeQuadrant])
                        .frame(maxWidth: .infinity, alignment: .center)
                    
                    // Dual timers: phase timer and total timer
                    VStack(spacing: 5) {
                        Text("UI Timer: \(currentPhaseTimeRemaining) seconds")
                            .font(bodyFont)
                            .foregroundColor(.gray)
                        
                        Text("Samples: \(phaseReadingsCount)/28")
                            .font(bodyFont)
                            .foregroundColor(phaseReadingsCount == 28 ? .green : .white)
                        
                        Text("Total Samples: \(appSettings.sensorReadings.count)/140")
                            .font(smallFont)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .center)
                    
                    // Progress indicator showing current phase out of total
                    HStack {
                        ForEach(0..<5) { index in
                            Circle()
                                .fill(index <= recordingPhase ? (index == recordingPhase ? Color.white : Color.gray.opacity(0.7)) : Color.gray.opacity(0.3))
                                .frame(width: 20, height: 20)
                        }
                    }
                    .padding(.bottom, 30)
                    .frame(maxWidth: .infinity, alignment: .center)
                    
                    Spacer()
                    
                    Button(action: onEscape) {
                        Text("Cancel Recording")
                            .font(buttonFont)
                            .foregroundColor(.red)
                    }
                    .padding(.bottom, 20)
                }
                .padding()
            }
        }
        .onAppear {
            // Initialize view
            resetForPhase(0)
            isRecording = false
            
            // Reset sensor readings
            appSettings.resetSensorReadings()
            
            // Start animation immediately
            if showingInstructions {
                startQuadrantAnimation()
            }
        }
        .onDisappear {
            timer?.invalidate()
            quadrantAnimationTimer?.invalidate()
        }
    }
    
    // Reset the view for a specific phase
    private func resetForPhase(_ phase: Int) {
        recordingPhase = phase
        activeQuadrant = phase == 0 ? -1 : phase - 1 // No active quadrant for baseline
        currentPhaseTimeRemaining = phaseDuration
    }
    
    // Start a continuous 50-second recording session
    private func startContinuousRecording() {
        isRecording = true
        totalTimeRemaining = 5 * phaseDuration // 50 seconds total
        currentPhaseTimeRemaining = phaseDuration // 10 seconds per phase
        recordingPhase = 0 // Start with baseline
        activeQuadrant = -1 // No active quadrant for baseline
        phaseReadingsCount = 0 // Reset phase readings count
        
        // Reset sample timestamp
        lastSampleTimestamp = Date()
        
        // Start collecting sensor data
        startDataCollection()
        
        // Start the timer to update UI only - sampling and phase transitions are independent
        startContinuousTimer()
    }
    
    // Start collecting sensor data
    private func startDataCollection() {
        // Create a timer that fires at exactly 0.36 second intervals to collect samples
        let dataCollectionTimer = Timer(timeInterval: sampleInterval, repeats: true) { timer in
            // Stop collecting if the recording is done
            if !self.isRecording {
                timer.invalidate()
                return
            }
            
            // Get current sensor values from BluetoothManager
            self.collectSensorReading()
        }
        
        // Add the timer to the main run loop to ensure it fires consistently
        RunLoop.main.add(dataCollectionTimer, forMode: .common)
    }
    
    // Collect a sensor reading and add it to the history
    private func collectSensorReading() {
        guard let bluetoothManager = appSettings.bluetoothManager as? BluetoothManager else { return }
        
        // Get current data from BluetoothManager
        let sensorValues = bluetoothManager.sensorData
        
        // Get the total sample count so far
        let totalSamples = appSettings.sensorReadings.count
        
        // Calculate current phase based on sample count (every 28 samples = new phase)
        // Phase 0 = samples 0-27
        // Phase 1 = samples 28-55
        // Phase 2 = samples 56-83
        // Phase 3 = samples 84-111
        // Phase 4 = samples 112-139
        let calculatedPhase = min(totalSamples / 28, 4) // Cap at phase 4
        
        // Update UI if phase has changed
        if calculatedPhase != recordingPhase {
            print("✅ Phase transition based on sample count: \(recordingPhase) -> \(calculatedPhase)")
            recordingPhase = calculatedPhase
            activeQuadrant = calculatedPhase == 0 ? -1 : calculatedPhase - 1
            currentPhaseTimeRemaining = phaseDuration // Reset the UI timer
            phaseReadingsCount = 0 // Reset phase readings counter for UI
            
            // Check if we've reached the end (all 5 phases)
            if calculatedPhase >= 5 {
                // We have enough samples, finish recording
                isRecording = false
                timerFinished = true
                processAndSendData()
                return
            }
        }
        
        // Update UI counter for current phase
        phaseReadingsCount = totalSamples % 28 + 1
        
        // Calculate the time value using the exact interval
        let timeSinceStart = sampleInterval * Double(totalSamples + 1)
        
        // Skip if we don't have enough data
        guard sensorValues.count > 8 else { return }
        
        // Log readings count periodically
        if totalSamples % 5 == 0 {
            print("📊 Recording: Total samples: \(totalSamples), Phase \(calculatedPhase) readings: \(phaseReadingsCount)")
        }
        
        // Map the BluetoothManager data to the format needed for the API
        let gasSensors = GasSensors(
            n1: sensorValues["n1"] ?? 0,
            n2: sensorValues["n2"] ?? 0,
            n3: sensorValues["n3"] ?? 0,
            n4: sensorValues["n4"] ?? 0,
            n5: sensorValues["n5"] ?? 0,
            n6: sensorValues["n6"] ?? 0,
            n7: sensorValues["n7"] ?? 0,
            n8: sensorValues["n8"] ?? 0,
            g1: sensorValues["g1"] ?? sensorValues["bme688_g1"] ?? 0,
            g2: sensorValues["g2"] ?? sensorValues["bme688_g2"] ?? 0
        )
        
        let envSensors = EnvironmentSensors(
            t: sensorValues["temp"] ?? sensorValues["t"] ?? 0,
            p: sensorValues["pressure"] ?? sensorValues["p"] ?? 0,
            h: sensorValues["humidity"] ?? sensorValues["h"] ?? 0
        )
        
        // Create the reading with proper phase and quadrant information
        let reading = SensorReading(
            time: timeSinceStart,
            phase: calculatedPhase,
            quadrant: SensorReading.quadrantName(for: calculatedPhase),
            gasSensors: gasSensors,
            envSensors: envSensors
        )
        
        // Add to our collection
        appSettings.sensorReadings.append(reading)
        
        // Check if we've collected all samples for all phases (140 total = 5 phases * 28 samples)
        if appSettings.sensorReadings.count >= 140 {
            print("✅ Collected all 140 samples (5 phases × 28 samples). Recording complete.")
            isRecording = false
            timerFinished = true
            processAndSendData()
        }
    }
    
    // Start the continuous timer for UI updates only
    func startContinuousTimer() {
        timer?.invalidate() // Cancel any existing timer
        
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            if self.totalTimeRemaining > 0 {
                self.totalTimeRemaining -= 1
                
                // Update the countdown for UI only
                if self.currentPhaseTimeRemaining > 0 {
                    self.currentPhaseTimeRemaining -= 1
                } else {
                    // Just reset the timer for UI purposes
                    // The actual phase transitions are handled in collectSensorReading based on sample count
                    self.currentPhaseTimeRemaining = self.phaseDuration
                }
            } else {
                // Time is up, but we'll let the sampling continue until we reach our target sample count
                // Don't stop the timer, just keep it showing zero
                self.currentPhaseTimeRemaining = 0
                self.totalTimeRemaining = 0
            }
        }
    }
    
    // Ensure all phases have exactly 28 readings (with some tolerance for the last phase)
    private func validateRecordingData() -> (Bool, String) {
        var dataIsValid = true
        var errorMessage = ""
        
        // Group readings by phase
        let phaseGroups = Dictionary(grouping: appSettings.sensorReadings, by: { $0.phase })
        
        // Check each phase has exactly 28 readings (with tolerance for the last phase)
        for phase in 0...4 {
            let phaseCount = phaseGroups[phase]?.count ?? 0
            print("📊 Phase \(phase) (\(SensorReading.quadrantName(for: phase))): \(phaseCount) readings")
            
            if phase < 4 {
                // For phases 0-3, we expect exactly 28 samples
                if phaseCount != 28 {
                    dataIsValid = false
                    errorMessage = "Phase \(phase) has \(phaseCount) readings. Expected: exactly 28"
                    print("❌ Data validation error: \(errorMessage)")
                    break
                }
            } else {
                // For phase 4 (the last phase), allow 28-30 samples
                if phaseCount < 28 || phaseCount > 30 {
                    dataIsValid = false
                    errorMessage = "Phase \(phase) has \(phaseCount) readings. Expected: 28-30"
                    print("❌ Data validation error: \(errorMessage)")
                    break
                }
            }
        }
        
        return (dataIsValid, errorMessage)
    }
    
    // Process collected data and send to API
    private func processAndSendData() {
        print("🚀 Beginning API processing (\(appSettings.sensorReadings.count) total readings)")
        
        // Update API status in app settings
        appSettings.apiStatus = .processing
        
        // First validate the data
        let (dataIsValid, errorMessage) = validateRecordingData()
        
        if !dataIsValid {
            // Update status and show error
            appSettings.apiStatus = .error(errorMessage)
            onStart() // Proceed to results view to show error
            return
        }
        
        print("✅ Data validation successful - all phases have correct number of readings")
        
        // ENHANCED DEBUGGING - Print sample data comparison
        print("\n📊 SENSOR DATA DETAIL COMPARISON 📊")
        if let firstReading = appSettings.sensorReadings.first, let lastReading = appSettings.sensorReadings.last {
            print("FIRST READING:")
            printDetailedReading(firstReading)
            
            print("\nLAST READING:")
            printDetailedReading(lastReading)
        }
        
        // Use the original reading set without padding (no more padding needed)
        let processedReadings = appSettings.sensorReadings
        
        // Convert to CSV for debugging and save to file
        let csvData = convertToCSV(readings: processedReadings)
        if let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            // Generate a filename using current date and time
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "MM-dd-yyyy-hhmmssa"
            let timestamp = dateFormatter.string(from: Date())
            let filename = "quadrants_\(timestamp).csv"
            let filePath = documentsDirectory.appendingPathComponent(filename)
            
            do {
                try csvData.write(to: filePath, atomically: true, encoding: .utf8)
                print("✅ Saved data to \(filePath.path)")
                
                // Store the file path in app settings
                appSettings.lastSavedCSVPath = filePath.path
            } catch {
                print("❌ Failed to save CSV: \(error)")
            }
        }
        
        // For API submission, just use the raw data without any additional processing
        sendDataToAPI(processedReadings)
    }
    
    // Helper method to dump detailed reading information for debugging
    private func dumpDetailedReadingInfo(_ readings: [SensorReading]) {
        print("\n🔍 DETAILED READING INFO FOR DEBUGGING 🔍")
        print("Total readings: \(readings.count)")
        
        // Group by phase
        let phaseGroups = Dictionary(grouping: readings, by: { $0.phase })
        
        // Print count and timestamps for each phase
        for phase in 0...4 {
            let phaseReadings = phaseGroups[phase] ?? []
            print("\nPhase \(phase) (\(SensorReading.quadrantName(for: phase))): \(phaseReadings.count) readings")
            print("Timestamps:")
            
            // Sort by time and print each timestamp
            let sortedReadings = phaseReadings.sorted(by: { $0.time < $1.time })
            for reading in sortedReadings {
                print("  \(String(format: "%.3f", reading.time))")
            }
        }
        print("🔍 END DETAILED READING INFO 🔍\n")
    }
    
    // Helper to normalize gas sensor values to the appropriate range - not used anymore
    private func normalizeGasValue(_ value: Double, targetRange: ClosedRange<Double>) -> Double {
        return value
    }
    
    // Helper method to print detailed sensor reading
    private func printDetailedReading(_ reading: SensorReading) {
        print("""
        Time: \(reading.time)
        Phase: \(reading.phase) (\(reading.quadrant))
        Gas Sensors:
          n1: \(reading.gasSensors.n1)
          n2: \(reading.gasSensors.n2)
          n3: \(reading.gasSensors.n3)
          n4: \(reading.gasSensors.n4)
          n5: \(reading.gasSensors.n5)
          n6: \(reading.gasSensors.n6)
          n7: \(reading.gasSensors.n7)
          n8: \(reading.gasSensors.n8)
          g1: \(reading.gasSensors.g1) ← BME688 gas resistance 1
          g2: \(reading.gasSensors.g2) ← BME688 gas resistance 2
        Environmental:
          t: \(reading.envSensors.t)
          p: \(reading.envSensors.p)
          h: \(reading.envSensors.h)
        """)
        
        // Add detailed debug info about g1/g2 values
        if reading.gasSensors.g1 < 1.0 || reading.gasSensors.g2 < 1.0 {
            print("⚠️ WARNING: g1/g2 values are very low or zero. They should be in the 300,000+ range from BME688.")
        } else if reading.gasSensors.g1 > 100000 || reading.gasSensors.g2 > 100000 {
            print("✅ g1/g2 values are in the expected range for BME688 gas resistance.")
        } else {
            print("⚠️ WARNING: g1/g2 values are present but unusually low for BME688 gas resistance.")
        }
    }
    
    // Helper to convert a small subset of readings to CSV for debug output
    private func convertToCSV(readings: [SensorReading]) -> String {
        // CSV header to match Python: 'Time', 'Phase', 'Quadrant', 'n1'-'n8', 'g1', 'g2', 't', 'p', 'h'
        let header = "Time,Phase,Quadrant,n1,n2,n3,n4,n5,n6,n7,n8,g1,g2,t,p,h"
        
        // Convert each reading to a CSV row
        let rows = readings.map { reading -> String in
            let row = [
                // Format time with 3 decimal places to match Python's f"{time_value:.3f}"
                String(format: "%.3f", reading.time),
                String(reading.phase),
                reading.quadrant,
                // Use string conversion for all sensor values - don't specify decimal places to match Python
                String(reading.gasSensors.n1),
                String(reading.gasSensors.n2),
                String(reading.gasSensors.n3),
                String(reading.gasSensors.n4),
                String(reading.gasSensors.n5),
                String(reading.gasSensors.n6),
                String(reading.gasSensors.n7),
                String(reading.gasSensors.n8),
                // Don't format g1/g2 with specific decimal places - use raw values
                String(reading.gasSensors.g1),
                String(reading.gasSensors.g2),
                String(reading.envSensors.t),
                String(reading.envSensors.p),
                String(reading.envSensors.h)
            ]
            return row.joined(separator: ",")
        }
        
        // Combine header and rows
        return ([header] + rows).joined(separator: "\n")
    }
    
    // Helper to send data to API
    private func sendDataToAPI(_ readings: [SensorReading]) {
        // Create the APIService if needed
        apiService = APIService()
        
        // Show loading state
        appSettings.apiStatus = .connecting
        
        // Preprocess sensor readings to normalize format to match test data
        let processedReadings = preprocessSensorReadings(readings)
        
        // Print the entire first 5 rows (including header) that will be sent to the API
        printFirstRows(processedReadings)
        
        // Call the API to get the prediction
        apiService.getPrediction(readings: processedReadings)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { completion in
                    switch completion {
                    case .finished:
                        break
                    case .failure(let error):
                        print("❌ API request failed: \(error)")
                        self.appSettings.apiStatus = .error("API request failed: \(error.localizedDescription)")
                    }
                },
                receiveValue: { response in
                    print("✅ API response received")
                    
                    // Update app settings with result
                    self.appSettings.isPeanutDetected = response.prediction == 1
                    self.appSettings.predictionConfidence = response.confidence
                    self.appSettings.apiStatus = .success
                    
                    // Calculate API response time
                    if let startTime = self.appSettings.lastApiCallTime {
                        let responseTime = Date().timeIntervalSince(startTime)
                        self.appSettings.apiResponseTime = responseTime
                        print("⏱️ API response time: \(String(format: "%.2f", responseTime)) seconds")
                    }
                    
                    // Navigate to results
                    self.onStart()
                }
            )
            .store(in: &cancellables)
    }
    
    // Helper to preprocess sensor readings for the API
    private func preprocessSensorReadings(_ readings: [SensorReading]) -> [SensorReading] {
        // In this simplified version, we don't modify the readings
        return SensorReading.preprocessSensorReadings(readings)
    }
    
    // Test API connection
    private func testAPIConnection() {
        print("📱 Initiating API connection test...")
        apiService.testAPIConnection()
    }
    
    // Start animation to cycle through quadrants
    private func startQuadrantAnimation() {
        // Cancel any existing timer
        quadrantAnimationTimer?.invalidate()
        
        // Start with first quadrant
        withAnimation(.easeInOut(duration: 0.3)) {
            animatedQuadrant = 1
        }
        
        // Create a timer that cycles through quadrants
        quadrantAnimationTimer = Timer.scheduledTimer(withTimeInterval: 0.8, repeats: true) { _ in
            // Cycle through quadrants 0-3 (for activeQuadrant binding)
            DispatchQueue.main.async {
                withAnimation(.easeInOut(duration: 0.3)) {
                    if self.animatedQuadrant >= 4 {
                        self.animatedQuadrant = 1
                    } else {
                        self.animatedQuadrant += 1
                    }
                    print("Animation timer: changed to quadrant \(self.animatedQuadrant)")
                }
            }
        }
        
        // Add to run loop to ensure it fires consistently
        RunLoop.main.add(quadrantAnimationTimer!, forMode: .common)
    }
    
    // Stop the quadrant animation
    private func stopQuadrantAnimation() {
        quadrantAnimationTimer?.invalidate()
        quadrantAnimationTimer = nil
    }
    
    // Helper to print the first few rows of data
    private func printFirstRows(_ readings: [SensorReading]) {
        let csvString = convertToCSV(readings: Array(readings.prefix(5)))
        let lines = csvString.split(separator: "\n")
        print("\nCSV DATA SAMPLE (first 5 rows):")
        for line in lines {
            print(line)
        }
        print("\n")
    }
}

struct SensorRecording_Previews: PreviewProvider {
    static var previews: some View {
        SensorRecording(onStart: {}, onEscape: {})
            .environmentObject(AppSettings()) // Inject the environment object here
            .preferredColorScheme(.dark)
    }
}

