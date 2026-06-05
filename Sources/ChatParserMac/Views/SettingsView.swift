import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        Form {
            TextField("Voicebox URL", text: $state.configuration.voiceboxURL)
            TextField("Default model", text: $state.configuration.model)
            TextField("Default language", text: $state.configuration.language)
        }
        .padding()
        .frame(width: 420)
    }
}
