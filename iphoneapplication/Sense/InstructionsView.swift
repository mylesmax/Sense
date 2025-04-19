//
//  InstructionsView.swift
//  Sense
//
//  Created by Nicolas Chicoine on 1/25/25.
//

import SwiftUI

struct InstructionsView: View {
    var onNext: () -> Void
    
    // Font styles using Nexa
    private let titleFont = Font.nexaFont(size: 28)
    private let buttonFont = Font.nexaFont(size: 24)

    var body: some View {
        VStack(spacing: 40) {
            VStack(spacing: 30) {
                Text("The sensor will record a baseline followed by 4 quadrant recordings of the food.")
                    .font(titleFont)
                    .multilineTextAlignment(.center)
                
                Text("Start the recording with the inlet in the top right and continue clockwise.")
                    .font(titleFont)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal)
            .padding(.top, 60)
            
            Spacer()
            
            Button(action: onNext) {
                Text("Next")
                    .font(buttonFont)
                    .padding()
                    .foregroundColor(.white)
                    .background(Color.blue)
                    .cornerRadius(10)
                    .frame(width: 160)
            }
            .padding(.bottom, 60)
        }
        .padding()
    }
}

struct InstructionsView_Previews: PreviewProvider {
    static var previews: some View {
        InstructionsView(onNext: {})
            .preferredColorScheme(.dark)
    }
}
