import AppKit
import Foundation
import UniformTypeIdentifiers

@MainActor
final class AppState: ObservableObject {
    @Published var configuration = RunConfiguration()
    @Published var logText = ""
    @Published var isRunning = false
    @Published var lastExitStatus: Int32?
    @Published var profiles: [VoiceProfile] = []
    @Published var selectedProfileID: String?
    @Published var samples: [ProfileSample] = []
    @Published var selectedSampleID: String?
    @Published var profileName = ""
    @Published var profileDescription = ""
    @Published var profileLanguage = "en"
    @Published var profilePersonality = ""
    @Published var sampleReferenceText = ""
    @Published var generationText = ""
    @Published var generatedAudioURL: URL?
    @Published var voiceboxMessage = ""
    @Published var isVoiceboxBusy = false
    @Published var chatMessages: [ChatMessage] = []
    @Published var chatParticipants: [String] = []
    @Published var meParticipant = ""
    @Published var selectedChatMessageID: String?
    @Published var participantProfileIDs: [String: String] = [:]
    @Published var chatMessage = ""
    @Published var isConversationGenerating = false
    @Published var conversationProgress = ""

    private let runner = ChatParserRunner()
    private let parser = WhatsAppExportParser()
    private let defaultGeneratedAudioName = "voicebox-selection.wav"
    private var conversationTask: Task<Void, Never>?

    var canRun: Bool {
        guard configuration.inputDirectory != nil, !isRunning else { return false }
        if configuration.mode == .synthesizeToAudio {
            return !configuration.profileID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !configuration.profileMap.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return true
    }

    var selectedProfile: VoiceProfile? {
        profiles.first { $0.id == selectedProfileID }
    }

    var selectedSample: ProfileSample? {
        samples.first { $0.id == selectedSampleID }
    }

    var selectedChatMessage: ChatMessage? {
        chatMessages.first { $0.id == selectedChatMessageID }
    }

    func chooseInputDirectory() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Choose"
        panel.message = "Choose an unzipped WhatsApp export folder or parent folder."

        if panel.runModal() == .OK {
            configuration.inputDirectory = panel.url
            loadChatExport()
        }
    }

    func loadChatExport() {
        guard let inputDirectory = configuration.inputDirectory else { return }
        do {
            let messages = try parser.parseExport(at: inputDirectory)
            chatMessages = messages
            chatParticipants = Array(Set(messages.compactMap(\.speaker))).sorted()
            if meParticipant.isEmpty || !chatParticipants.contains(meParticipant) {
                meParticipant = chatParticipants.first ?? ""
            }
            selectedChatMessageID = messages.first?.id
            chatMessage = "Loaded \(messages.count) messages"
        } catch {
            chatMessages = []
            chatParticipants = []
            selectedChatMessageID = nil
            chatMessage = error.localizedDescription
        }
    }

    func selectChatMessage(_ message: ChatMessage) {
        selectedChatMessageID = message.id
        if !message.text.isEmpty {
            generationText = message.text
        }
    }

    func openAttachment(_ attachment: ChatAttachment) {
        NSWorkspace.shared.open(attachment.url)
    }

    func profileID(for participant: String) -> String {
        participantProfileIDs[participant] ?? ""
    }

    func setProfileID(_ profileID: String, for participant: String) {
        participantProfileIDs[participant] = profileID
        rebuildProfileMap()
    }

    func assignSelectedVoiceProfile(to participant: String) {
        guard let selectedProfileID else { return }
        setProfileID(selectedProfileID, for: participant)
    }

    func generateSelectedChatMessageAudio() {
        guard let message = selectedChatMessage else { return }
        generationText = message.text
        selectedProfileID = participantProfileIDs[message.participant] ?? selectedProfileID
        generateSelectedText()
    }

    func generateConversationAudio() {
        cancelConversationGeneration(updateMessage: false)
        let playable = chatMessages.filter { message in
            guard let speaker = message.speaker else { return false }
            return !(participantProfileIDs[speaker] ?? "").isEmpty && !message.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        guard !playable.isEmpty else {
            chatMessage = "Assign at least one speaker to a Voicebox profile."
            return
        }

        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose Output Folder"
        if panel.runModal() != .OK { return }
        guard let outputFolder = panel.url else { return }

        isConversationGenerating = true
        conversationProgress = "0 / \(playable.count)"
        conversationTask = Task {
            do {
                let api = try VoiceboxAPI(baseURLString: self.configuration.voiceboxURL)
                for (index, message) in playable.enumerated() {
                    try Task.checkCancellation()
                    guard let speaker = message.speaker, let profileID = self.participantProfileIDs[speaker] else { continue }
                    let safeSpeaker = speaker.replacingOccurrences(of: "[^A-Za-z0-9_-]+", with: "-", options: .regularExpression)
                    let filename = "\(String(format: "%05d", index + 1))-\(safeSpeaker).wav"
                    let destination = outputFolder.appendingPathComponent(filename)
                    _ = try await api.generateSpeech(
                        profileID: profileID,
                        text: message.text,
                        language: self.configuration.language,
                        destination: destination
                    )
                    await MainActor.run {
                        guard !Task.isCancelled else { return }
                        self.conversationProgress = "\(index + 1) / \(playable.count)"
                    }
                }
                try Task.checkCancellation()
                await MainActor.run {
                    self.chatMessage = "Generated \(playable.count) audio clips"
                    self.isConversationGenerating = false
                    self.conversationProgress = ""
                    self.conversationTask = nil
                    NSWorkspace.shared.open(outputFolder)
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.chatMessage = "Conversation generation cancelled."
                    self.isConversationGenerating = false
                    self.conversationProgress = ""
                    self.conversationTask = nil
                }
            } catch {
                await MainActor.run {
                    self.chatMessage = error.localizedDescription
                    self.isConversationGenerating = false
                    self.conversationProgress = ""
                    self.conversationTask = nil
                }
            }
        }
    }

    func cancelConversationGeneration() {
        cancelConversationGeneration(updateMessage: true)
    }

    private func cancelConversationGeneration(updateMessage: Bool) {
        conversationTask?.cancel()
        conversationTask = nil
        isConversationGenerating = false
        conversationProgress = ""
        if updateMessage {
            chatMessage = "Conversation generation cancellation requested."
        }
    }

    func run() {
        guard canRun else { return }
        logText = ""
        lastExitStatus = nil
        isRunning = true
        append("Starting ChatParser with Voicebox at \(configuration.voiceboxURL)\n")

        runner.run(
            configuration: configuration,
            onOutput: { [weak self] text in
                Task { @MainActor in self?.append(text) }
            },
            onTermination: { [weak self] status in
                Task { @MainActor in
                    self?.isRunning = false
                    self?.lastExitStatus = status
                    self?.append("\nProcess exited with status \(status)\n")
                }
            }
        )
    }

    func cancel() {
        runner.cancel()
        append("\nCancellation requested.\n")
    }

    private func append(_ text: String) {
        logText.append(text)
    }

    func refreshVoicebox() {
        Task {
            await performVoiceboxAction("Loaded profiles") { api in
                let loaded = try await api.listProfiles()
                await MainActor.run {
                    self.profiles = loaded
                    if self.selectedProfileID == nil || !loaded.contains(where: { $0.id == self.selectedProfileID }) {
                        self.selectedProfileID = loaded.first?.id
                    }
                    self.syncProfileEditor()
                }
                if let selectedProfileID = await MainActor.run(body: { self.selectedProfileID }) {
                    let loadedSamples = try await api.listSamples(profileID: selectedProfileID)
                    await MainActor.run { self.samples = loadedSamples }
                }
            }
        }
    }

    func selectProfile(_ profileID: String?) {
        selectedProfileID = profileID
        syncProfileEditor()
        refreshSamples()
    }

    func createProfile() {
        Task {
            await performVoiceboxAction("Created profile") { api in
                let created = try await api.createProfile(
                    name: self.profileNameOrDefault,
                    description: self.emptyToNil(self.profileDescription),
                    language: self.profileLanguageOrDefault,
                    personality: self.emptyToNil(self.profilePersonality)
                )
                let loaded = try await api.listProfiles()
                await MainActor.run {
                    self.profiles = loaded
                    self.selectedProfileID = created.id
                    self.syncProfileEditor()
                }
            }
        }
    }

    func updateProfile() {
        guard let selectedProfileID else { return }
        Task {
            await performVoiceboxAction("Updated profile") { api in
                let updated = try await api.updateProfile(
                    selectedProfileID,
                    name: self.profileNameOrDefault,
                    description: self.emptyToNil(self.profileDescription),
                    language: self.profileLanguageOrDefault,
                    personality: self.emptyToNil(self.profilePersonality)
                )
                await MainActor.run {
                    if let index = self.profiles.firstIndex(where: { $0.id == updated.id }) {
                        self.profiles[index] = updated
                    }
                    self.syncProfileEditor()
                }
            }
        }
    }

    func deleteProfile() {
        guard let selectedProfileID else { return }
        Task {
            await performVoiceboxAction("Deleted profile") { api in
                try await api.deleteProfile(selectedProfileID)
                let loaded = try await api.listProfiles()
                await MainActor.run {
                    self.profiles = loaded
                    self.selectedProfileID = loaded.first?.id
                    self.samples = []
                    self.syncProfileEditor()
                }
            }
        }
    }

    func refreshSamples() {
        guard let selectedProfileID else {
            samples = []
            return
        }
        Task {
            await performVoiceboxAction("Loaded samples") { api in
                let loaded = try await api.listSamples(profileID: selectedProfileID)
                await MainActor.run {
                    self.samples = loaded
                    self.selectedSampleID = loaded.first?.id
                    self.sampleReferenceText = loaded.first?.referenceText ?? ""
                }
            }
        }
    }

    func selectSample(_ sampleID: String?) {
        selectedSampleID = sampleID
        sampleReferenceText = selectedSample?.referenceText ?? ""
    }

    func chooseAndUploadSample() {
        guard let selectedProfileID else { return }
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.audio]
        panel.prompt = "Add Clip"
        panel.message = "Choose a local voice clip for the selected Voicebox profile."

        if panel.runModal() == .OK, let url = panel.url {
            Task {
                await performVoiceboxAction("Added sample") { api in
                    let sample = try await api.addSample(
                        profileID: selectedProfileID,
                        fileURL: url,
                        referenceText: self.sampleReferenceText
                    )
                    let loaded = try await api.listSamples(profileID: selectedProfileID)
                    await MainActor.run {
                        self.samples = loaded
                        self.selectedSampleID = sample.id
                        self.sampleReferenceText = sample.referenceText
                    }
                }
            }
        }
    }

    func updateSample() {
        guard let selectedSampleID else { return }
        Task {
            await performVoiceboxAction("Updated sample") { api in
                let sample = try await api.updateSample(selectedSampleID, referenceText: self.sampleReferenceText)
                await MainActor.run {
                    if let index = self.samples.firstIndex(where: { $0.id == sample.id }) {
                        self.samples[index] = sample
                    }
                }
            }
        }
    }

    func deleteSample() {
        guard let selectedSampleID, let selectedProfileID else { return }
        Task {
            await performVoiceboxAction("Deleted sample") { api in
                try await api.deleteSample(selectedSampleID)
                let loaded = try await api.listSamples(profileID: selectedProfileID)
                await MainActor.run {
                    self.samples = loaded
                    self.selectedSampleID = loaded.first?.id
                    self.sampleReferenceText = loaded.first?.referenceText ?? ""
                }
            }
        }
    }

    func generateSelectedText() {
        guard let selectedProfileID, !generationText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = defaultGeneratedAudioName
        panel.prompt = "Generate"

        if panel.runModal() == .OK, let destination = panel.url {
            Task {
                await performVoiceboxAction("Generated audio") { api in
                    let output = try await api.generateSpeech(
                        profileID: selectedProfileID,
                        text: self.generationText,
                        language: self.configuration.language,
                        destination: destination
                    )
                    await MainActor.run { self.generatedAudioURL = output }
                }
            }
        }
    }

    private var profileNameOrDefault: String {
        let trimmed = profileName.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "New Voice Profile" : trimmed
    }

    private func syncProfileEditor() {
        guard let selectedProfile else {
            profileName = ""
            profileDescription = ""
            profileLanguage = "en"
            profilePersonality = ""
            return
        }
        profileName = selectedProfile.name
        profileDescription = selectedProfile.description ?? ""
        profileLanguage = selectedProfile.language
        profilePersonality = selectedProfile.personality ?? ""
        configuration.profileID = selectedProfile.id
    }

    private func rebuildProfileMap() {
        configuration.profileMap = participantProfileIDs
            .filter { !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "\n")
    }

    private var profileLanguageOrDefault: String {
        let trimmed = profileLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "en" : trimmed
    }

    private func emptyToNil(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func performVoiceboxAction(_ successMessage: String, action: @escaping (VoiceboxAPI) async throws -> Void) async {
        await MainActor.run {
            self.isVoiceboxBusy = true
            self.voiceboxMessage = ""
        }
        do {
            let api = try VoiceboxAPI(baseURLString: await MainActor.run { self.configuration.voiceboxURL })
            try await action(api)
            await MainActor.run { self.voiceboxMessage = successMessage }
        } catch {
            await MainActor.run { self.voiceboxMessage = error.localizedDescription }
        }
        await MainActor.run { self.isVoiceboxBusy = false }
    }
}
