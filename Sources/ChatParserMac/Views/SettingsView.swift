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
            HStack {
                Button("Start Voicebox") {
                    state.startVoiceboxServer()
                }
                .disabled(state.isVoiceboxServerStarting || state.isVoiceboxServerRunning)
                Button("Stop Voicebox") {
                    state.stopVoiceboxServer()
                }
                .disabled(!state.isVoiceboxServerManaged && !state.isVoiceboxServerStarting)
                if state.isVoiceboxServerStarting {
                    ProgressView()
                        .controlSize(.small)
                }
            }
            Text(state.voiceboxServerMessage.isEmpty ? "Uses the external/voicebox submodule for local loopback URLs." : state.voiceboxServerMessage)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(nil)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            TextField("Default model", text: $state.configuration.model)
            TextField("Default language", text: $state.configuration.language)
        }
        .padding()
        .frame(width: 560)
    }
}
