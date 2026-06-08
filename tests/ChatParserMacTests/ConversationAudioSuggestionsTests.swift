import Foundation
import Testing
@testable import ChatParserMac

struct ConversationAudioSuggestionsTests {
    @Test func suggestionsUseAssignedSpeakersAndExistingAudioAttachments() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let aliceAudio = folder.appendingPathComponent("alice.opus")
        let bobAudio = folder.appendingPathComponent("bob.opus")
        let aliceImage = folder.appendingPathComponent("alice.jpg")
        try Data("audio".utf8).write(to: aliceAudio)
        try Data("audio".utf8).write(to: bobAudio)
        try Data("image".utf8).write(to: aliceImage)

        let messages = [
            message("1", speaker: "Alice", text: "first voice note", attachment: audio("alice.opus", aliceAudio)),
            message("2", speaker: "Bob", text: "other profile", attachment: audio("bob.opus", bobAudio)),
            message("3", speaker: "Alice", text: "photo", attachment: image("alice.jpg", aliceImage)),
            message("4", speaker: nil, text: "system", attachment: audio("system.opus", folder.appendingPathComponent("system.opus"))),
            message("5", speaker: "Alice", text: "missing audio", attachment: audio("missing.opus", folder.appendingPathComponent("missing.opus")))
        ]

        let suggestions = ConversationAudioClipSuggestion.suggestions(
            forProfileID: "profile-a",
            messages: messages,
            participantProfileIDs: ["Alice": "profile-a", "Bob": "profile-b"]
        )

        #expect(suggestions.map(\.speaker) == ["Alice"])
        #expect(suggestions.map(\.filename) == ["alice.opus"])
        #expect(suggestions.first?.referenceText == "first voice note")
    }

    @Test func suggestionsStripIOSAndAndroidAttachmentMarkers() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let iosAudio = folder.appendingPathComponent("ios.opus")
        let androidAudio = folder.appendingPathComponent("AUD-20260608-WA0001.opus")
        try Data("audio".utf8).write(to: iosAudio)
        try Data("audio".utf8).write(to: androidAudio)

        let messages = [
            message("1", speaker: "Alice", text: "Voice note text\n<attached: ios.opus>", attachment: audio("ios.opus", iosAudio)),
            message("2", speaker: "Alice", text: "Android text AUD-20260608-WA0001.opus (file attached)", attachment: audio("AUD-20260608-WA0001.opus", androidAudio))
        ]

        let suggestions = ConversationAudioClipSuggestion.suggestions(
            forProfileID: "profile-a",
            messages: messages,
            participantProfileIDs: ["Alice": "profile-a"]
        )

        #expect(suggestions.map(\.referenceText) == ["Voice note text", "Android text"])
    }

    @Test func suggestionsReturnEmptyForEmptyProfileID() throws {
        let folder = FileManager.default.temporaryDirectory
        let audioURL = folder.appendingPathComponent("clip.opus")
        try Data("audio".utf8).write(to: audioURL)
        defer { try? FileManager.default.removeItem(at: audioURL) }

        let suggestions = ConversationAudioClipSuggestion.suggestions(
            forProfileID: "",
            messages: [message("1", speaker: "Alice", text: "clip", attachment: audio("clip.opus", audioURL))],
            participantProfileIDs: ["Alice": "profile-a"]
        )

        #expect(suggestions.isEmpty)
    }

    private func message(_ id: String, speaker: String?, text: String, attachment: ChatAttachment?) -> ChatMessage {
        ChatMessage(id: id, timestamp: Date(timeIntervalSince1970: Double(id) ?? 0), speaker: speaker, text: text, attachment: attachment)
    }

    private func audio(_ filename: String, _ url: URL) -> ChatAttachment {
        ChatAttachment(filename: filename, url: url, isAudio: true)
    }

    private func image(_ filename: String, _ url: URL) -> ChatAttachment {
        ChatAttachment(filename: filename, url: url, isAudio: false)
    }
}
