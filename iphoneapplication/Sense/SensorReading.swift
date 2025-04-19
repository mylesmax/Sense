import Foundation

struct SensorReading: Codable {
    let time: Double
    let phase: Int
    let quadrant: String
    let gasSensors: GasSensors
    let envSensors: EnvironmentSensors
    
    // Custom coding keys to match the API's expected format
    enum CodingKeys: String, CodingKey {
        case time, phase, quadrant
        case gasSensors = "gas_sensors"
        case envSensors = "env_sensors"
    }
}

struct GasSensors: Codable {
    let n1: Double
    let n2: Double
    let n3: Double
    let n4: Double
    let n5: Double
    let n6: Double
    let n7: Double
    let n8: Double
    let g1: Double
    let g2: Double
}

struct EnvironmentSensors: Codable {
    let t: Double  // temperature
    let p: Double  // pressure
    let h: Double  // humidity
}

// Helper for mapping Phase to Quadrant string
extension SensorReading {
    static func quadrantName(for phase: Int) -> String {
        switch phase {
        case 0:
            return "baseline"
        case 1:
            return "topright"
        case 2:
            return "bottomright"
        case 3:
            return "bottomleft"
        case 4:
            return "topleft"
        default:
            return "unknown"
        }
    }
    
    // Preprocess readings to match reference data format
    static func preprocessSensorReadings(_ readings: [SensorReading]) -> [SensorReading] {
        // Return original readings without any normalization
        return readings
    }
    
    // Helper to normalize gas sensor values to appropriate range
    static private func normalizeGasValue(_ value: Double, targetRange: ClosedRange<Double>) -> Double {
        return value // Simply return original value without normalization
    }
} 