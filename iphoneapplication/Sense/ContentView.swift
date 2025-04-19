import SwiftUI

struct ContentView: View {
    @State private var currentStage: Stage = .home
    @EnvironmentObject var appSettings: AppSettings
    @State private var bluetoothConnected = false  // Track connection state locally
    @State private var showWelcomeAlert = true     // Show welcome alert about gas tile removal
    
    enum Stage {
        case home
        case instructions
        case sensorRecording
        case resultsView
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                switch currentStage {
                case .home:
                    HomeScreen(onStart: {
                        print("HomeScreen onStart called, switching to instructions stage")
                        currentStage = .instructions
                    })
                    
                case .instructions:
                    InstructionsView(onNext: {
                        print("InstructionsView onNext called, switching to sensorRecording stage")
                        currentStage = .sensorRecording
                    })
                    
                case .sensorRecording:
                    SensorRecording(
                        onStart: {
                            print("SensorRecording onStart called, switching to resultsView stage")
                            currentStage = .resultsView
                        },
                        onEscape: {
                            print("SensorRecording onEscape called, switching to home stage")
                            currentStage = .home  // Go back to HomeScreen
                        }
                    )
                    
                case .resultsView:
                    ResultsView(onBackToHome: {
                        print("ResultsView onBackToHome called, switching to home stage")
                        currentStage = .home
                    })
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(appSettings.backgroundColor.ignoresSafeArea())
        .onChange(of: appSettings.bluetoothManager.isConnected) { newValue in
            print("🔄 Bluetooth connection changed to: \(newValue)")
            bluetoothConnected = newValue
        }
        .id("content-\(bluetoothConnected)")  // Force view refresh on connection change
        .alert(
            "Disclaimer",
            isPresented: $showWelcomeAlert
        ) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("By clicking OK, you acknowledge that all results are preliminary and may not be used as a medical recommendation.")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(AppSettings())
    }
}
