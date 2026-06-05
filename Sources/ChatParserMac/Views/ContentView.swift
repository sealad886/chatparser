import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        NavigationSplitView {
            SidebarView()
        } detail: {
            DetailView()
        }
    }
}

private struct SidebarView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        List {
            Section("Export") {
                Button {
                    state.chooseInputDirectory()
                } label: {
                    Label("Choose Folder", systemImage: "folder")
                }

                Text(state.configuration.inputDirectory?.lastPathComponent ?? "No folder selected")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Section("Transform") {
                Picker("Mode", selection: $state.configuration.mode) {
                    ForEach(TransformationMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }

                Toggle("Force redo", isOn: $state.configuration.forceRedo)
                Toggle("Show progress", isOn: $state.configuration.showProgress)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("ChatParser")
        .toolbar {
            ToolbarItem {
                Button {
                    state.run()
                } label: {
                    Label("Run", systemImage: "play.fill")
                }
                .disabled(!state.canRun)
            }
        }
    }
}

private struct DetailView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            Form {
                Section("Voicebox") {
                    TextField("URL", text: $state.configuration.voiceboxURL)
                    TextField("Transcription model", text: $state.configuration.model)
                    TextField("Voice profile id", text: $state.configuration.profileID)
                    TextField("Language", text: $state.configuration.language)
                    if state.configuration.mode == .synthesizeToAudio && state.configuration.profileID.isEmpty {
                        Text("Voice profile id is required for generated audio.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            .frame(maxHeight: 210)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Run Log")
                        .font(.headline)
                    Spacer()
                    if state.isRunning {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Button {
                        state.isRunning ? state.cancel() : state.run()
                    } label: {
                        Label(state.isRunning ? "Cancel" : "Run", systemImage: state.isRunning ? "stop.fill" : "play.fill")
                    }
                    .disabled(!state.canRun && !state.isRunning)
                }

                ScrollView {
                    Text(state.logText.isEmpty ? "Choose a WhatsApp export folder, confirm Voicebox is running locally, then run." : state.logText)
                        .font(.system(.body, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(12)
                }
                .background(.quaternary.opacity(0.35))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            .padding()
        }
    }
}
