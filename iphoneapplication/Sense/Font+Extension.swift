import SwiftUI

extension Font {
    // Main Nexa font function - uses the font file we have available
    static func nexaFont(size: CGFloat) -> Font {
        .custom("Nexa-Trial-Book", size: size)
    }
    
    // Convenience methods for common font sizes
    static let nexaTitle = nexaFont(size: 32)
    static let nexaSubtitle = nexaFont(size: 20)
    static let nexaHeadline = nexaFont(size: 18)
    static let nexaBody = nexaFont(size: 16)
    static let nexaCaption = nexaFont(size: 14)
    static let nexaSmall = nexaFont(size: 12)
    
    // Dynamic font sizes that adapt to the system text size setting
    static func nexaFont(size: CGFloat, relativeTo textStyle: TextStyle) -> Font {
        .custom("Nexa-Trial-Book", size: size, relativeTo: textStyle)
    }
    
    // Maintain backward compatibility
    static func nexaTrialBook(size: CGFloat) -> Font {
        nexaFont(size: size)
    }
    
    static func nexaTrialBook(size: CGFloat, relativeTo textStyle: TextStyle) -> Font {
        nexaFont(size: size, relativeTo: textStyle)
    }
} 