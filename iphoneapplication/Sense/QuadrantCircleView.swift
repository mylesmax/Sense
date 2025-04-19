import SwiftUI

struct QuadrantCircleView: View {
    // Current active quadrant (0-3, or -1 for none)
    @Binding var activeQuadrant: Int
    
    // Colors for the quadrants
    var inactiveColor: Color = Color.gray.opacity(0.3)
    var activeColors: [Color] = [.blue, .green, .orange, .purple]
    
    // Size of the circle
    var size: CGFloat = 200
    
    var body: some View {
        ZStack {
            // Background circle
            Circle()
                .stroke(inactiveColor, lineWidth: 2)
                .frame(width: size, height: size)
            
            // Quadrant 1 (Top-Right)
            QuadrantPath(quadrant: 0)
                .fill(activeQuadrant == 0 ? activeColors[0] : inactiveColor)
                .frame(width: size, height: size)
            
            // Quadrant 2 (Bottom-Right)
            QuadrantPath(quadrant: 1)
                .fill(activeQuadrant == 1 ? activeColors[1] : inactiveColor)
                .frame(width: size, height: size)
            
            // Quadrant 3 (Bottom-Left)
            QuadrantPath(quadrant: 2)
                .fill(activeQuadrant == 2 ? activeColors[2] : inactiveColor)
                .frame(width: size, height: size)
            
            // Quadrant 4 (Top-Left)
            QuadrantPath(quadrant: 3)
                .fill(activeQuadrant == 3 ? activeColors[3] : inactiveColor)
                .frame(width: size, height: size)
            
            // Dividing lines
            Path { path in
                path.move(to: CGPoint(x: size/2, y: 0))
                path.addLine(to: CGPoint(x: size/2, y: size))
                path.move(to: CGPoint(x: 0, y: size/2))
                path.addLine(to: CGPoint(x: size, y: size/2))
            }
            .stroke(Color.white, lineWidth: 2)
            .frame(width: size, height: size)
            
            // Center circle
            Circle()
                .fill(Color.white)
                .frame(width: size/5, height: size/5)
        }
        .frame(width: size, height: size)
        .animation(.easeInOut(duration: 0.5), value: activeQuadrant)
    }
}

struct QuadrantPath: Shape {
    let quadrant: Int
    
    func path(in rect: CGRect) -> Path {
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let radius = min(rect.width, rect.height) / 2
        
        var path = Path()
        path.move(to: center)
        
        switch quadrant {
        case 0: // Top-Right
            path.addLine(to: CGPoint(x: center.x, y: center.y - radius))
            path.addArc(center: center, radius: radius, startAngle: .degrees(270), endAngle: .degrees(0), clockwise: false)
        case 1: // Bottom-Right
            path.addLine(to: CGPoint(x: center.x + radius, y: center.y))
            path.addArc(center: center, radius: radius, startAngle: .degrees(0), endAngle: .degrees(90), clockwise: false)
        case 2: // Bottom-Left
            path.addLine(to: CGPoint(x: center.x, y: center.y + radius))
            path.addArc(center: center, radius: radius, startAngle: .degrees(90), endAngle: .degrees(180), clockwise: false)
        case 3: // Top-Left
            path.addLine(to: CGPoint(x: center.x - radius, y: center.y))
            path.addArc(center: center, radius: radius, startAngle: .degrees(180), endAngle: .degrees(270), clockwise: false)
        default:
            break
        }
        
        path.closeSubpath()
        return path
    }
}

struct QuadrantCircleView_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            QuadrantCircleView(activeQuadrant: .constant(-1))
                .previewDisplayName("No Active Quadrant")
            QuadrantCircleView(activeQuadrant: .constant(0))
                .previewDisplayName("First Quadrant")
            QuadrantCircleView(activeQuadrant: .constant(1))
                .previewDisplayName("Second Quadrant")
            QuadrantCircleView(activeQuadrant: .constant(2))
                .previewDisplayName("Third Quadrant")
            QuadrantCircleView(activeQuadrant: .constant(3))
                .previewDisplayName("Fourth Quadrant")
        }
        .padding()
        .background(Color.black)
        .previewLayout(.sizeThatFits)
    }
} 