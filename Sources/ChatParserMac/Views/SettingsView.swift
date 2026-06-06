import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var state: AppState

    private var isVoiceboxURLValid: Bool {
        state.configuration.isVoiceboxURLValid
    }

    var body: some View {
        Form {
            TextField("Voicebox URL", text: $state.configuration.voiceboxURL)
            if !isVoiceboxURLValid {
                Text("Enter a valid http or https URL with a host.")
                    .font(.callout)
                    .foregroundStyle(.red)
            }
            TextField("Default model", text: $state.configuration.model)
            TextField("Default language", text: $state.configuration.language)
        }
        .padding()
        .frame(width: 420)
    }
}
