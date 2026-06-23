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

enum WhatsAppSourceFormat: String, Equatable {
    case ios
    case android
}

struct ChatMessage: Identifiable, Equatable {
    let id: String
    let timestamp: Date
    let speaker: String?
    let text: String
    let attachment: ChatAttachment?
    let sourceFormat: WhatsAppSourceFormat
    let sequenceNumber: Int

    init(
        id: String,
        timestamp: Date,
        speaker: String?,
        text: String,
        attachment: ChatAttachment?,
        sourceFormat: WhatsAppSourceFormat = .ios,
        sequenceNumber: Int = 0
    ) {
        self.id = id
        self.timestamp = timestamp
        self.speaker = speaker
        self.text = text
        self.attachment = attachment
        self.sourceFormat = sourceFormat
        self.sequenceNumber = sequenceNumber
    }

    var participant: String {
        speaker ?? "System"
    }

    var generatedAudioFilename: String {
        switch sourceFormat {
        case .android:
            return "PTT-\(formattedTimestamp("yyyyMMdd"))-WA\(String(format: "%04d", sequenceNumber)).wav"
        case .ios:
            return "\(String(format: "%08d", sequenceNumber + 1))-AUDIO-\(formattedTimestamp("yyyy-MM-dd-HH-mm-ss")).wav"
        }
    }

    private func formattedTimestamp(_ format: String) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = format
        return formatter.string(from: timestamp)
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

struct ConversationAudioRenderJob: Identifiable, Equatable {
    enum Action: Equatable {
        case synthesize(profileID: String, text: String)
        case copyExistingAudio(sourceURL: URL)
    }

    let id: String
    let speaker: String
    let destinationFilename: String
    let action: Action
}

enum ConversationAudioRenderPlan {
    static func jobs(
        messages: [ChatMessage],
        participantProfileIDs: [String: String],
        fileManager: FileManager = .default
    ) -> [ConversationAudioRenderJob] {
        messages.compactMap { message in
            guard let speaker = message.speaker,
                  let profileID = participantProfileIDs[speaker],
                  !profileID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }

            if let attachment = message.attachment, attachment.isAudio {
                guard fileManager.fileExists(atPath: attachment.url.path) else { return nil }
                return ConversationAudioRenderJob(
                    id: message.id,
                    speaker: speaker,
                    destinationFilename: existingAudioFilename(for: message, attachment: attachment),
                    action: .copyExistingAudio(sourceURL: attachment.url)
                )
            }

            let text = message.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            return ConversationAudioRenderJob(
                id: message.id,
                speaker: speaker,
                destinationFilename: message.generatedAudioFilename,
                action: .synthesize(profileID: profileID, text: text)
            )
        }
    }

    static func progressText(
        completed: Int,
        total: Int,
        currentFilename: String?,
        startedAt: Date,
        now: Date = Date()
    ) -> String {
        let safeTotal = max(total, 0)
        let safeCompleted = min(max(completed, 0), safeTotal)
        var parts = ["\(safeCompleted) / \(safeTotal)"]
        if let currentFilename, !currentFilename.isEmpty, safeCompleted < safeTotal {
            parts.append(currentFilename)
        }
        if safeCompleted > 0, safeCompleted < safeTotal {
            let elapsed = max(now.timeIntervalSince(startedAt), 0)
            let rate = elapsed / Double(safeCompleted)
            let remaining = rate * Double(safeTotal - safeCompleted)
            parts.append("ETA \(formatDuration(remaining))")
        }
        return parts.joined(separator: " - ")
    }

    private static func existingAudioFilename(for message: ChatMessage, attachment: ChatAttachment) -> String {
        let generated = URL(fileURLWithPath: message.generatedAudioFilename)
        let attachmentExtension = attachment.url.pathExtension
        guard !attachmentExtension.isEmpty else {
            return attachment.filename
        }
        return generated.deletingPathExtension().appendingPathExtension(attachmentExtension).lastPathComponent
    }

    private static func formatDuration(_ seconds: TimeInterval) -> String {
        let rounded = max(Int(seconds.rounded()), 0)
        if rounded < 60 {
            return "\(rounded)s"
        }
        let minutes = rounded / 60
        let remainingSeconds = rounded % 60
        if minutes < 60 {
            return "\(minutes)m \(remainingSeconds)s"
        }
        let hours = minutes / 60
        let remainingMinutes = minutes % 60
        return "\(hours)h \(remainingMinutes)m"
    }
}
