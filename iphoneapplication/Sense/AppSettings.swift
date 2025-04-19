//
//  AppSettings.swift
//  Sense
//
//  Created by Nicolas Chicoine on 1/25/25.
//

import SwiftUI
import Combine
import CoreBluetooth

// Define the settings as an ObservableObject so they can be shared across views
class AppSettings: ObservableObject {
    @Published var recordingDuration: Int = 10 // Sample recording duration - 10 seconds default
    @Published var baselineDuration: Int = 10 // Baseline recording duration - 10 seconds default
    @Published var backgroundColor: Color = Color(red: 0.11, green: 0.11, blue: 0.12) // Dark gray background
    @Published var bluetoothManager: BluetoothManager
    @Published var isDarkMode: Bool = true
    @Published var isPeanutDetected: Bool = false // Toggle for peanut detection result
    @Published var predictionConfidence: Double = 0.0 // Confidence level from the API
    @Published var apiStatus: APIStatus = .idle // Current status of the API communication
    @Published var sensorReadings: [SensorReading] = [] // Collection of readings for the API
    @Published var lastApiCallTime: Date? = nil
    @Published var apiResponseTime: Double = 0.0
    @Published var lastSavedCSVPath: String? = nil // Path to the last saved CSV file
    
    // Subscription to propagate changes
    private var cancellables = Set<AnyCancellable>()
    
    // API Status states
    enum APIStatus {
        case idle
        case connecting
        case processing
        case success
        case error(String)
    }
    
    init() {
        // Initialize BluetoothManager
        self.bluetoothManager = BluetoothManager()
        
        // Setup publisher to relay BluetoothManager objectWillChange events
        bluetoothManager.objectWillChange
            .sink { [weak self] _ in
                // Propagate BluetoothManager changes to AppSettings
                DispatchQueue.main.async {
                    self?.objectWillChange.send()
                }
            }
            .store(in: &cancellables)
        
        // Setup notification observers for API timing metrics
        setupNotificationObservers()
    }
    
    deinit {
        // Remove notification observers
        NotificationCenter.default.removeObserver(self)
    }
    
    // Setup notification observers for API timing
    private func setupNotificationObservers() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAPICallStarted(_:)),
            name: Notification.Name("APICallStarted"),
            object: nil
        )
        
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAPICallCompleted(_:)),
            name: Notification.Name("APICallCompleted"),
            object: nil
        )
    }
    
    // Handle API call started notification
    @objc private func handleAPICallStarted(_ notification: Notification) {
        DispatchQueue.main.async {
            if let timestamp = notification.userInfo?["timestamp"] as? Date {
                self.lastApiCallTime = timestamp
            }
        }
    }
    
    // Handle API call completed notification
    @objc private func handleAPICallCompleted(_ notification: Notification) {
        DispatchQueue.main.async {
            if let responseTime = notification.userInfo?["responseTime"] as? Double {
                self.apiResponseTime = responseTime
            }
        }
    }
    
    // Add method to reset settings to defaults
    func resetToDefaults() {
        recordingDuration = 10
        baselineDuration = 10
        backgroundColor = Color(red: 0.11, green: 0.11, blue: 0.12)
        sensorReadings = []
        apiStatus = .idle
        predictionConfidence = 0.0
        isPeanutDetected = false
        lastApiCallTime = nil
        apiResponseTime = 0.0
        lastSavedCSVPath = nil
    }
    
    // Reset sensor readings collection
    func resetSensorReadings() {
        sensorReadings = []
        lastSavedCSVPath = nil
    }
}
