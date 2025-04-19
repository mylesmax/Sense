//
//  HomeScreen.swift
//  Sense
//
//  Created by Nicolas Chicoine on 1/25/25.
//

import SwiftUI

struct HomeScreen: View {
    var onStart: () -> Void
    @EnvironmentObject var appSettings: AppSettings
    @State private var showingSettings = false
    @State private var showingBluetoothDebug = false
    @State private var showingInstructionManual = false
    @State private var showingAPITest = false
    
    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            VStack {
                Spacer()
                
                Image("SenseLogo") 
                    .resizable()
                    .renderingMode(.template)
                    .foregroundColor(.white)
                    .scaledToFit()
                    .frame(width: 200)
                    .padding()
                
                Text("Welcome to the App!")
                    .font(.nexaTitle)
                    .padding()
                
                // Status Section: Battery and Bluetooth
                VStack(spacing: 15) {
                    // Battery Status
                    if appSettings.bluetoothManager.isConnected {
                        HStack {
                            // Battery percentage with system font
                            batteryIcon
                                .foregroundColor(batteryColor)
                                .font(.system(size: 18))
                            
                            // Use Text with concatenated string to avoid font issues with percentage
                            Text("\(appSettings.bluetoothManager.batteryLevel)")
                                .foregroundColor(batteryColor)
                                .font(.system(size: 18, weight: .bold))
                            
                            Text("%")
                                .foregroundColor(batteryColor)
                                .font(.system(size: 18))
                                .padding(.leading, -5)
                            
                            Text("Battery")
                                .font(.system(size: 16))
                                .foregroundColor(.gray)
                                .padding(.leading, 5)
                        }
                        .padding(.horizontal)
                    }
                    
                    // Bluetooth Connection Status
                    HStack {
                        Image(systemName: appSettings.bluetoothManager.isConnected ? "bluetooth.connected" : "bluetooth")
                            .foregroundColor(appSettings.bluetoothManager.isConnected ? .green : .red)
                            .font(.system(size: 18))
                        
                        Text(appSettings.bluetoothManager.isConnected ? "Connected" : "Sensor not connected")
                            .foregroundColor(appSettings.bluetoothManager.isConnected ? .green : .red)
                            .font(.system(size: 18, weight: .semibold))
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 10)
                .background(Color.black.opacity(0.3))
                .cornerRadius(10)
                .padding(.horizontal)
                
                // Bluetooth Connect Button
                Button(action: {
                    print("Connect button tapped, current state: \(appSettings.bluetoothManager.isConnected)")
                    if appSettings.bluetoothManager.isConnected {
                        appSettings.bluetoothManager.disconnect()
                    } else {
                        appSettings.bluetoothManager.startScanning()
                        // Force UI refresh after a delay
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                            print("Checking connection state after delay: \(appSettings.bluetoothManager.isConnected)")
                        }
                    }
                }) {
                    Text(appSettings.bluetoothManager.isConnected ? "Disconnect" : "Connect to Sensor")
                        .font(.nexaHeadline)
                        .padding()
                        .foregroundColor(.white)
                        .background(appSettings.bluetoothManager.isConnected ? Color.red : Color.blue)
                        .cornerRadius(10)
                }
                .padding()
                
                // Start Button (Simplified Implementation)
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(appSettings.bluetoothManager.isConnected ? Color.green : Color.gray)
                        .shadow(color: Color.black.opacity(0.2), radius: 5, x: 0, y: 2)
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal)
                    
                    Text("Start")
                        .font(.nexaFont(size: 28))
                        .padding()
                        .foregroundColor(.white)
                }
                .frame(height: 60)
                .onTapGesture {
                    print("Start button tapped directly via ZStack onTapGesture")
                    if appSettings.bluetoothManager.isConnected {
                        print("Directly calling onStart()")
                        onStart()
                    } else {
                        print("Device not connected, tap ignored")
                    }
                }
                .opacity(appSettings.bluetoothManager.isConnected ? 1.0 : 0.5)
                .padding(.vertical)
                
                Spacer()
            }
            
            // Bottom right buttons row
            HStack(spacing: 20) {
                // Test API button for debugging (leftmost)
                #if DEBUG
                Button(action: { showingAPITest = true }) {
                    Image(systemName: "flask.fill")
                        .font(.nexaTitle)
                        .foregroundColor(.primary)
                }
                #endif
                
                // Info button
                Button(action: { showingInstructionManual = true }) {
                    Image(systemName: "info.circle")
                        .font(.nexaTitle)
                        .foregroundColor(.primary)
                }
                
                // Bluetooth debug button
                Button(action: { showingBluetoothDebug = true }) {
                    Image(systemName: "terminal")
                        .font(.nexaTitle)
                        .foregroundColor(.primary)
                }
                
                // Settings button (rightmost)
                Button(action: { showingSettings = true }) {
                    Image(systemName: "gear")
                        .font(.nexaTitle)
                        .foregroundColor(.primary)
                }
            }
            .padding(.trailing, 20)
            .padding(.bottom, 20)
        }
        .sheet(isPresented: $showingSettings) {
            SettingsSheet()
        }
        .sheet(isPresented: $showingBluetoothDebug) {
            NavigationView {
                BluetoothDebugView()
                    .environmentObject(appSettings)
            }
        }
        .sheet(isPresented: $showingInstructionManual) {
            NavigationView {
                InstructionManualView()
            }
        }
        .sheet(isPresented: $showingAPITest) {
            APITestView()
        }
    }
    
    // Battery icon based on battery level
    private var batteryIcon: some View {
        let level = appSettings.bluetoothManager.batteryLevel
        let iconName: String
        
        switch level {
        case 0..<10:
            iconName = "battery.0"
        case 10..<25:
            iconName = "battery.25"
        case 25..<50:
            iconName = "battery.50"
        case 50..<75:
            iconName = "battery.75"
        default:
            iconName = "battery.100"
        }
        
        return Image(systemName: level <= 10 ? iconName : "\(iconName).fill")
    }
    
    // Battery color based on battery level
    private var batteryColor: Color {
        let level = appSettings.bluetoothManager.batteryLevel
        
        if level <= 10 {
            return .red
        } else if level <= 25 {
            return .orange
        } else {
            return .green
        }
    }
}

// Instruction Manual View
struct InstructionManualView: View {
    @Environment(\.dismiss) var dismiss
    
    // Font styles using Nexa
    private let titleFont = Font.nexaFont(size: 32)
    private let bodyFont = Font.nexaFont(size: 18)
    
    var body: some View {
        VStack(spacing: 30) {
            Text("Instructions")
                .font(titleFont)
                .padding(.top, 40)
            
            Spacer()
            
            Text("Detailed instructions will be added in a future update.")
                .font(bodyFont)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)
            
            Spacer()
        }
        .navigationBarItems(trailing: Button("Close") {
            dismiss()
        }
        .font(.nexaHeadline))
    }
}

struct SettingsSheet: View {
    @EnvironmentObject var appSettings: AppSettings
    @Environment(\.dismiss) var dismiss
    @State private var showingResultsPreview = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Recording").font(.nexaHeadline)) {
                    Stepper(
                        "Baseline Duration: \(appSettings.baselineDuration) seconds",
                        value: $appSettings.baselineDuration,
                        in: 1...30
                    )
                    .font(.nexaBody)
                    
                    Stepper(
                        "Sample Duration: \(appSettings.recordingDuration) seconds",
                        value: $appSettings.recordingDuration,
                        in: 1...30
                    )
                    .font(.nexaBody)
                }
                
                Section(header: Text("Demo Result Settings").font(.nexaHeadline)) {
                    Text("Choose the result to be displayed (for demo purposes)")
                        .font(.nexaCaption)
                        .foregroundColor(.gray)
                    
                    Picker("Demo Result", selection: $appSettings.isPeanutDetected) {
                        Text("No Peanuts Detected (Safe)").tag(false)
                        Text("Peanuts Detected (Warning)").tag(true)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding(.vertical, 5)
                    
                    Button("Preview Result Screen") {
                        showingResultsPreview = true
                    }
                    .font(.nexaHeadline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(appSettings.isPeanutDetected ? Color.red.opacity(0.8) : Color.blue.opacity(0.8))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                
                Section(header: Text("Appearance").font(.nexaHeadline)) {
                    ColorPicker("Background Color", selection: $appSettings.backgroundColor)
                        .font(.nexaBody)
                }
                
                Section {
                    Button("Reset to Defaults") {
                        appSettings.resetToDefaults()
                    }
                    .font(.nexaHeadline)
                    .foregroundColor(.red)
                }
            }
            .navigationTitle("Settings")
            .navigationBarItems(trailing: Button("Done") {
                dismiss()
            }
            .font(.nexaHeadline))
            .sheet(isPresented: $showingResultsPreview) {
                NavigationView {
                    ResultsView(onBackToHome: {
                        showingResultsPreview = false
                    })
                    .navigationBarItems(trailing: Button("Close") {
                        showingResultsPreview = false
                    }
                    .font(.nexaHeadline))
                }
            }
        }
    }
}

struct HomeScreen_Previews: PreviewProvider {
    static var previews: some View {
        HomeScreen(onStart: {})
            .environmentObject(AppSettings())
    }
}
