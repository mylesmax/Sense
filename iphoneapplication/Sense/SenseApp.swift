//
//  SenseApp.swift
//  Sense
//
//  Created by Nicolas Chicoine on 1/25/25.
//

import SwiftUI

// Extension to set Nexa as the default font throughout the app
extension View {
    func nexaFont() -> some View {
        self.environment(\.font, Font.custom("Nexa-Regular", size: 16))
    }
}

// Class to manage orientation lock
class OrientationLock: ObservableObject {
    @Published var orientation: UIInterfaceOrientationMask = .portrait
    
    init() {
        // Initialize with portrait orientation
    }
    
    func lockOrientation() {
        // Force portrait mode
        UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
    }
}

// Add AppDelegate to handle orientation requests
class AppDelegate: NSObject, UIApplicationDelegate {
    var orientationLock = OrientationLock()
    
    func application(_ application: UIApplication, supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        return orientationLock.orientation
    }
    
    // Prevent device from sleeping when app becomes active
    func applicationDidBecomeActive(_ application: UIApplication) {
        // Disable idle timer to prevent the device from sleeping
        UIApplication.shared.isIdleTimerDisabled = true
        print("App became active: Disabled idle timer to prevent sleep")
    }
    
    // Re-enable device sleep when app goes to background
    func applicationDidEnterBackground(_ application: UIApplication) {
        // Re-enable idle timer when app is in background
        UIApplication.shared.isIdleTimerDisabled = false
        print("App entered background: Re-enabled idle timer")
    }
    
    // Also handle when app will terminate
    func applicationWillTerminate(_ application: UIApplication) {
        // Ensure idle timer is re-enabled when app terminates
        UIApplication.shared.isIdleTimerDisabled = false
        print("App will terminate: Re-enabled idle timer")
    }
}

@main
struct SenseApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var appSettings = AppSettings() // Instantiate the AppSettings object
    
    init() {
        // Set Nexa font as the default for navigation bars and other UIKit elements
        let nexaBold = UIFont(name: "Nexa-Bold", size: 17) ?? UIFont.systemFont(ofSize: 17, weight: .bold)
        let nexaRegular = UIFont(name: "Nexa-Regular", size: 16) ?? UIFont.systemFont(ofSize: 16)
        
        // Configure UINavigationBar appearance
        let navBarAppearance = UINavigationBar.appearance()
        navBarAppearance.titleTextAttributes = [.font: nexaBold]
        
        // Configure UIBarButtonItem appearance
        let barButtonAppearance = UIBarButtonItem.appearance()
        barButtonAppearance.setTitleTextAttributes([.font: nexaRegular], for: .normal)
        
        // Configure other UIKit elements as needed
        UILabel.appearance().font = nexaRegular
        UITextField.appearance().font = nexaRegular
        UIButton.appearance().titleLabel?.font = nexaBold
        
        print("SenseApp initializing with AppSettings: \(appSettings)")
        print("BluetoothManager initial isConnected: \(appSettings.bluetoothManager.isConnected)")
        
        // Lock orientation to portrait on app launch
        UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
        
        // Disable idle timer immediately on app start
        UIApplication.shared.isIdleTimerDisabled = true
        print("App initialized: Disabled idle timer to prevent sleep")
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appSettings) // Pass the AppSettings object to the environment
                .preferredColorScheme(.dark) // Force dark mode
                .nexaFont() // Apply Nexa font to the entire app hierarchy
                .onAppear {
                    // Force portrait orientation when the app appears
                    UIDevice.current.setValue(UIInterfaceOrientation.portrait.rawValue, forKey: "orientation")
                    
                    // Ensure idle timer is disabled when the view appears
                    UIApplication.shared.isIdleTimerDisabled = true
                }
        }
    }
}
