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
        TabView {
            TransformRunView()
                .tabItem { Label("Transform", systemImage: "arrow.triangle.2.circlepath") }
            VoiceProfilesView()
                .tabItem { Label("Profiles", systemImage: "person.wave.2") }
            TextToVoiceView()
                .tabItem { Label("Speak Text", systemImage: "waveform") }
        }
    }
}

private struct TransformRunView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            VoiceboxSettingsForm()
                .frame(maxHeight: 250)

            Divider()

            RunLogView()
        }
    }
}

private struct VoiceboxSettingsForm: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        Form {
            Section("Voicebox") {
                TextField("URL", text: $state.configuration.voiceboxURL)
                TextField("Transcription model", text: $state.configuration.model)
                TextField("Fallback profile id", text: $state.configuration.profileID)
                TextField("Language", text: $state.configuration.language)
                TextEditor(text: $state.configuration.profileMap)
                    .font(.system(.body, design: .monospaced))
                    .frame(minHeight: 70)
                    .overlay(alignment: .topLeading) {
                        if state.configuration.profileMap.isEmpty {
                            Text("Alice=voice-profile-id")
                                .foregroundStyle(.tertiary)
                                .padding(.top, 8)
                                .padding(.leading, 5)
                        }
                    }
                if state.configuration.mode == .synthesizeToAudio && !state.canRun {
                    Text("Generated audio requires either a fallback profile id or one speaker=profile-id mapping.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }
}

private struct RunLogView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
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

private struct VoiceProfilesView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        HSplitView {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Voice Profiles")
                        .font(.headline)
                    Spacer()
                    Button {
                        state.refreshVoicebox()
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                }

                List(selection: $state.selectedProfileID) {
                    ForEach(state.profiles) { profile in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(profile.name)
                                .font(.body)
                            Text(profile.id)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("\(profile.language) · samples \(profile.sampleCount ?? 0)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .tag(Optional(profile.id))
                    }
                }
                .onChange(of: state.selectedProfileID) { _, value in
                    state.selectProfile(value)
                }
            }
            .padding()
            .frame(minWidth: 290)

            VStack(alignment: .leading, spacing: 12) {
                ProfileEditorView()
                Divider()
                ProfileSamplesView()
            }
            .padding()
            .frame(minWidth: 520)
        }
        .onAppear {
            if state.profiles.isEmpty {
                state.refreshVoicebox()
            }
        }
    }
}

private struct ProfileEditorView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
            GridRow {
                Text("Name")
                TextField("Profile name", text: $state.profileName)
            }
            GridRow {
                Text("Description")
                TextField("Optional description", text: $state.profileDescription)
            }
            GridRow {
                Text("Language")
                TextField("Language code", text: $state.profileLanguage)
                    .frame(maxWidth: 120)
            }
            GridRow {
                Text("Personality")
                TextField("Optional personality prompt", text: $state.profilePersonality)
            }
            GridRow {
                Text("Profile ID")
                Text(state.selectedProfileID ?? "No profile selected")
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }

        HStack {
            Button {
                state.createProfile()
            } label: {
                Label("Create", systemImage: "plus")
            }
            Button {
                state.updateProfile()
            } label: {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .disabled(state.selectedProfileID == nil)
            Button(role: .destructive) {
                state.deleteProfile()
            } label: {
                Label("Delete", systemImage: "trash")
            }
            .disabled(state.selectedProfileID == nil)
            Spacer()
            if state.isVoiceboxBusy {
                ProgressView()
                    .controlSize(.small)
            }
            Text(state.voiceboxMessage)
                .foregroundStyle(.secondary)
        }
    }
}

private struct ProfileSamplesView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Clips")
                    .font(.headline)
                Spacer()
                Button {
                    state.refreshSamples()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(state.selectedProfileID == nil)
            }

            List(selection: $state.selectedSampleID) {
                ForEach(state.samples) { sample in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(sample.referenceText)
                            .lineLimit(1)
                        Text(sample.audioPath)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .tag(Optional(sample.id))
                }
            }
            .frame(minHeight: 150)
            .onChange(of: state.selectedSampleID) { _, value in
                state.selectSample(value)
            }

            TextEditor(text: $state.sampleReferenceText)
                .font(.body)
                .frame(minHeight: 80)
                .overlay(alignment: .topLeading) {
                    if state.sampleReferenceText.isEmpty {
                        Text("Reference text spoken in the selected clip")
                            .foregroundStyle(.tertiary)
                            .padding(.top, 8)
                            .padding(.leading, 5)
                    }
                }

            HStack {
                Button {
                    state.chooseAndUploadSample()
                } label: {
                    Label("Add Clip", systemImage: "waveform.badge.plus")
                }
                .disabled(state.selectedProfileID == nil || state.sampleReferenceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button {
                    state.updateSample()
                } label: {
                    Label("Save Clip Text", systemImage: "square.and.arrow.down")
                }
                .disabled(state.selectedSampleID == nil)
                Button(role: .destructive) {
                    state.deleteSample()
                } label: {
                    Label("Delete Clip", systemImage: "trash")
                }
                .disabled(state.selectedSampleID == nil)
            }
        }
    }
}

private struct TextToVoiceView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Picker("Voice", selection: $state.selectedProfileID) {
                    Text("Choose profile").tag(Optional<String>.none)
                    ForEach(state.profiles) { profile in
                        Text(profile.name).tag(Optional(profile.id))
                    }
                }
                .frame(maxWidth: 360)
                Button {
                    state.refreshVoicebox()
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                Spacer()
                Button {
                    state.generateSelectedText()
                } label: {
                    Label("Generate Audio", systemImage: "waveform")
                }
                .disabled(state.selectedProfileID == nil || state.generationText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }

            TextEditor(text: $state.generationText)
                .font(.body)
                .frame(minHeight: 260)
                .overlay(alignment: .topLeading) {
                    if state.generationText.isEmpty {
                        Text("Paste selected chat text here")
                            .foregroundStyle(.tertiary)
                            .padding(.top, 8)
                            .padding(.leading, 5)
                    }
                }

            if let generatedAudioURL = state.generatedAudioURL {
                Text("Saved: \(generatedAudioURL.path)")
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            Spacer()
        }
        .padding()
        .onAppear {
            if state.profiles.isEmpty {
                state.refreshVoicebox()
            }
        }
    }
}
