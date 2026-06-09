import AppKit
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 300)
        } detail: {
            DetailView()
        }
        .task {
            await state.startVoiceboxServerIfNeeded()
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
                Button {
                    state.loadChatExport()
                } label: {
                    Label("Reload Chat", systemImage: "text.bubble")
                }
                .disabled(state.configuration.inputDirectory == nil)

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
            ConversationView()
                .tabItem { Label("Conversation", systemImage: "bubble.left.and.bubble.right") }
            TransformRunView()
                .tabItem { Label("Transform", systemImage: "arrow.triangle.2.circlepath") }
            VoiceProfilesView()
                .tabItem { Label("Profiles", systemImage: "person.wave.2") }
            TextToVoiceView()
                .tabItem { Label("Speak Text", systemImage: "waveform") }
        }
    }
}

private struct ConversationView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Chat")
                        .font(.headline)
                    Spacer()
                    Picker("Me", selection: $state.meParticipant) {
                        ForEach(state.chatParticipants, id: \.self) { participant in
                            Text(participant).tag(participant)
                        }
                    }
                    .frame(maxWidth: 220)
                    .disabled(state.chatParticipants.isEmpty)
                }

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(state.chatMessages) { message in
                                ChatBubbleView(message: message)
                                    .id(message.id)
                                    .onTapGesture {
                                        state.selectChatMessage(message)
                                    }
                            }
                        }
                        .padding(.vertical, 12)
                    }
                    .onChange(of: state.selectedChatMessageID) { _, value in
                        if let value {
                            proxy.scrollTo(value, anchor: .center)
                        }
                    }
                }
                .background(.quaternary.opacity(0.2))
                .clipShape(RoundedRectangle(cornerRadius: 8))

                HStack {
                    Text(state.chatMessage)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if state.isConversationGenerating {
                        ProgressView()
                            .controlSize(.small)
                        Text(state.conversationProgress)
                            .foregroundStyle(.secondary)
                        Button {
                            state.cancelConversationGeneration()
                        } label: {
                            Label("Cancel", systemImage: "stop.fill")
                        }
                    }
                    Button {
                        state.generateSelectedChatMessageAudio()
                    } label: {
                        Label("Speak Selection", systemImage: "waveform")
                    }
                    .disabled(state.selectedChatMessage == nil)
                    Button {
                        state.generateConversationAudio()
                    } label: {
                        Label("Render Conversation", systemImage: "play.circle")
                    }
                    .disabled(state.chatMessages.isEmpty || state.isConversationGenerating)
                }
            }
            .padding()
            .frame(minWidth: 620, maxWidth: .infinity)

            Divider()
            SpeakerProfilePanel()
                .frame(width: 340)
        }
    }
}

private struct ChatBubbleView: View {
    @EnvironmentObject private var state: AppState
    let message: ChatMessage

    private var isMe: Bool {
        message.speaker == state.meParticipant && message.speaker != nil
    }

    private var isSelected: Bool {
        state.selectedChatMessageID == message.id
    }

    var body: some View {
        HStack {
            if isMe { Spacer(minLength: 80) }
            VStack(alignment: isMe ? .trailing : .leading, spacing: 4) {
                if message.speaker != nil {
                    Text(message.participant)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text(message.text)
                        .textSelection(.enabled)
                    if let attachment = message.attachment {
                        AttachmentView(attachment: attachment)
                    }
                }
                .padding(10)
                .background(isMe ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
                )
                Text(message.timestamp.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: 520, alignment: isMe ? .trailing : .leading)
            if !isMe { Spacer(minLength: 80) }
        }
        .padding(.horizontal, 12)
    }
}

private struct AttachmentView: View {
    @EnvironmentObject private var state: AppState
    let attachment: ChatAttachment

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if attachment.isPlayableMedia {
                AttachmentMediaPlayer(url: attachment.url, isVideo: attachment.isVideo)
                    .frame(height: attachment.isVideo ? 220 : 56)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            HStack(spacing: 8) {
                if attachment.isImage {
                    AttachmentThumbnail(url: attachment.url)
                } else {
                    Image(systemName: iconName)
                        .frame(width: 28)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(attachment.filename)
                        .font(.callout)
                        .lineLimit(1)
                    Text(attachment.url.path)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Button {
                    state.openAttachment(attachment)
                } label: {
                    Label("Open", systemImage: "arrow.up.right.square")
                }
                .labelStyle(.iconOnly)
            }
        }
        .padding(8)
        .background(.quaternary.opacity(0.35))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private var iconName: String {
        if attachment.isAudio { return "waveform" }
        if attachment.isVideo { return "film" }
        if attachment.isImage { return "photo" }
        return "paperclip"
    }
}

private struct AttachmentThumbnail: View {
    let url: URL
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: "photo")
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: 72, height: 72)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .task(id: url) {
            await loadImage()
        }
    }

    private func loadImage() async {
        let cache = AttachmentImageCache.shared
        let key = url as NSURL
        if let cached = cache.object(forKey: key) {
            image = cached
            return
        }
        let data = await Task.detached(priority: .utility) {
            try? Data(contentsOf: url)
        }.value
        guard !Task.isCancelled, let data, let loaded = NSImage(data: data) else { return }
        cache.setObject(loaded, forKey: key)
        image = loaded
    }
}

private enum AttachmentImageCache {
    static let shared = NSCache<NSURL, NSImage>()
}

private struct SpeakerProfilePanel: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Speakers")
                    .font(.headline)
                Spacer()
                Button {
                    state.refreshVoicebox()
                } label: {
                    Label("Profiles", systemImage: "arrow.clockwise")
                }
            }

            if state.chatParticipants.isEmpty {
                Text("Import a WhatsApp export to map speakers.")
                    .foregroundStyle(.secondary)
            } else {
                List {
                    ForEach(state.chatParticipants, id: \.self) { participant in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(participant)
                                    .font(.body)
                                if participant == state.meParticipant {
                                    Text("me")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                            }
                            Picker("Voice", selection: Binding(
                                get: { state.profileID(for: participant) },
                                set: { state.setProfileID($0, for: participant) }
                            )) {
                                Text("No voice").tag("")
                                ForEach(state.profiles) { profile in
                                    Text(profile.name).tag(profile.id)
                                }
                            }
                            .labelsHidden()
                            Button {
                                state.assignSelectedVoiceProfile(to: participant)
                            } label: {
                                Label("Use Selected Profile", systemImage: "person.crop.circle.badge.checkmark")
                            }
                            .disabled(state.selectedProfileID == nil)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }

            Divider()

            Text("Selected")
                .font(.headline)
            if let message = state.selectedChatMessage {
                Text(message.participant)
                    .foregroundStyle(.secondary)
                Text(message.text)
                    .lineLimit(5)
                    .textSelection(.enabled)
            } else {
                Text("Click a message to speak it through its speaker profile.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
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
                HStack {
                    Button {
                        state.startVoiceboxServer()
                    } label: {
                        Label("Start Server", systemImage: "power")
                    }
                    .disabled(state.isVoiceboxServerStarting || state.isVoiceboxServerRunning)

                    Button {
                        state.stopVoiceboxServer()
                    } label: {
                        Label("Stop Server", systemImage: "stop.fill")
                    }
                    .disabled(!state.isVoiceboxServerManaged && !state.isVoiceboxServerStarting)

                    if state.isVoiceboxServerStarting {
                        ProgressView()
                            .controlSize(.small)
                    }
                }
                Text(state.voiceboxServerMessage.isEmpty ? "Voicebox starts from external/voicebox when using the default local URL." : state.voiceboxServerMessage)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(nil)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
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
                                .allowsHitTesting(false)
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
        HStack(spacing: 0) {
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
            .frame(width: 320)

            Divider()
            VStack(alignment: .leading, spacing: 12) {
                ProfileEditorView()
                Divider()
                ProfileSamplesView()
            }
            .padding()
            .frame(minWidth: 560, maxWidth: .infinity)
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

            SuggestedConversationClipsView()
        }
    }
}

private struct SuggestedConversationClipsView: View {
    @EnvironmentObject private var state: AppState

    var body: some View {
        let suggestions = state.selectedProfileConversationClipSuggestions
        let assignedSpeakers = state.speakersAssignedToSelectedProfile
        VStack(alignment: .leading, spacing: 8) {
            Divider()
            HStack {
                Text("Suggested from Chat")
                    .font(.headline)
                Spacer()
                Text("\(suggestions.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if state.selectedProfileID == nil {
                Text("Select a profile to see assigned speaker clips.")
                    .foregroundStyle(.secondary)
            } else if assignedSpeakers.isEmpty {
                Text("Assign this profile to a chat speaker to see candidate audio clips.")
                    .foregroundStyle(.secondary)
            } else if suggestions.isEmpty {
                Text("No audio attachments found for \(assignedSpeakers.joined(separator: ", ")).")
                    .foregroundStyle(.secondary)
            } else {
                Text("Candidates from \(assignedSpeakers.joined(separator: ", ")). Add only clips that contain the speaker's voice.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(suggestions) { suggestion in
                            SuggestedConversationClipRow(suggestion: suggestion)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(minHeight: 120, maxHeight: 220)
            }
        }
    }
}

private struct SuggestedConversationClipRow: View {
    @EnvironmentObject private var state: AppState
    let suggestion: ConversationAudioClipSuggestion

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            AttachmentMediaPlayer(url: suggestion.audioAttachment.url, isVideo: suggestion.audioAttachment.isVideo)
                .frame(height: 56)
                .clipShape(RoundedRectangle(cornerRadius: 6))

            HStack(spacing: 8) {
                Image(systemName: "waveform")
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(suggestion.filename)
                        .lineLimit(1)
                    Text("\(suggestion.speaker) · \(suggestion.timestamp.formatted(date: .abbreviated, time: .shortened))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    state.openSuggestedClip(suggestion)
                } label: {
                    Label("Open", systemImage: "play.circle")
                }
                Button {
                    state.useSuggestedClipText(suggestion)
                } label: {
                    Label("Use Text", systemImage: "text.quote")
                }
                .disabled(suggestion.referenceText.isEmpty)
                Button {
                    state.addSuggestedClip(suggestion)
                } label: {
                    Label("Add", systemImage: "plus.circle")
                }
                .disabled(state.selectedProfileID == nil || state.isVoiceboxBusy)
            }

            Text(suggestion.referenceText.isEmpty ? "No transcript text detected. Enter reference text before adding." : suggestion.referenceText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .textSelection(.enabled)
            Text(suggestion.url.path)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .textSelection(.enabled)
        }
        .padding(.vertical, 6)
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
