import Foundation

struct ChatAttachment: Identifiable, Equatable {
    let filename: String
    let url: URL
    let isAudio: Bool

    var id: String {
        "\(filename)-\(url.standardizedFileURL.path)"
    }

    var isImage: Bool {
        ["jpg", "jpeg", "png", "gif", "heic", "webp"].contains(url.pathExtension.lowercased())
    }

    var isVideo: Bool {
        ["mp4", "mov", "m4v", "3gp", "webm"].contains(url.pathExtension.lowercased())
    }

    var isPlayableMedia: Bool {
        isAudio || isVideo
    }
}

struct ChatMessage: Identifiable, Equatable {
    let id: String
    let timestamp: Date
    let speaker: String?
    let text: String
    let attachment: ChatAttachment?

    var participant: String {
        speaker ?? "System"
    }
}

struct ConversationAudioClipSuggestion: Identifiable, Equatable {
    let messageID: String
    let speaker: String
    let timestamp: Date
    let filename: String
    let url: URL
    let referenceText: String

    var id: String {
        "\(messageID)-\(url.standardizedFileURL.path)"
    }

    var audioAttachment: ChatAttachment {
        ChatAttachment(filename: filename, url: url, isAudio: true)
    }

    static func suggestions(
        forProfileID profileID: String,
        messages: [ChatMessage],
        participantProfileIDs: [String: String],
        fileManager: FileManager = .default
    ) -> [ConversationAudioClipSuggestion] {
        guard !profileID.isEmpty else { return [] }
        return messages.compactMap { message in
            guard let speaker = message.speaker,
                  participantProfileIDs[speaker] == profileID,
                  let attachment = message.attachment,
                  attachment.isAudio,
                  fileManager.fileExists(atPath: attachment.url.path)
            else { return nil }
            return ConversationAudioClipSuggestion(
                messageID: message.id,
                speaker: speaker,
                timestamp: message.timestamp,
                filename: attachment.filename,
                url: attachment.url,
                referenceText: referenceText(from: message.text, attachmentFilename: attachment.filename)
            )
        }
    }

    private static func referenceText(from text: String, attachmentFilename: String) -> String {
        text.replacingOccurrences(of: #"<attached:\s*\#(NSRegularExpression.escapedPattern(for: attachmentFilename))>"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"\#(NSRegularExpression.escapedPattern(for: attachmentFilename))\s+\(file attached\)"#, with: "", options: [.regularExpression, .caseInsensitive])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
