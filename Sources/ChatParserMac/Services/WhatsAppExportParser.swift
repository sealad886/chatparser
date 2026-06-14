import Foundation

struct WhatsAppExportParser: Sendable {
    private static let timestampFormats = [
        "dd/MM/yyyy, HH:mm:ss",
        "dd/MM/yyyy, HH:mm",
        "dd/MM/yy, HH:mm:ss",
        "dd/MM/yy, HH:mm",
        "dd/MM/yyyy, h:mm:ss a",
        "dd/MM/yyyy, h:mm a",
        "dd/MM/yy, h:mm:ss a",
        "dd/MM/yy, h:mm a",
        "MM/dd/yyyy, h:mm:ss a",
        "MM/dd/yyyy, h:mm a",
        "MM/dd/yy, h:mm:ss a",
        "MM/dd/yy, h:mm a",
        "MM/dd/yyyy, HH:mm:ss",
        "MM/dd/yyyy, HH:mm",
        "MM/dd/yy, HH:mm:ss",
        "MM/dd/yy, HH:mm"
    ]

    func parseExport(at folder: URL) throws -> [ChatMessage] {
        let chatFile = try findChatFile(in: folder)
        let raw = try String(contentsOf: chatFile, encoding: .utf8)
        return parse(raw, mediaRoot: chatFile.deletingLastPathComponent())
    }

    func parse(_ raw: String, mediaRoot: URL) -> [ChatMessage] {
        var messages: [ChatMessage] = []
        var current: ChatMessage?
        var sequence = 0
        let formatters = Self.makeTimestampFormatters()

        for line in raw.components(separatedBy: .newlines) {
            guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            if let parsed = parseHeader(line, mediaRoot: mediaRoot, sequence: sequence, formatters: formatters) {
                if let current {
                    messages.append(current)
                }
                current = parsed
                sequence += 1
            } else if let existing = current {
                let text = existing.text.isEmpty ? line : existing.text + "\n" + line
                current = ChatMessage(
                    id: existing.id,
                    timestamp: existing.timestamp,
                    speaker: existing.speaker,
                    text: text,
                    attachment: existing.attachment,
                    sourceFormat: existing.sourceFormat,
                    sequenceNumber: existing.sequenceNumber
                )
            }
        }

        if let current {
            messages.append(current)
        }
        return messages
    }

    private func findChatFile(in folder: URL) throws -> URL {
        let direct = folder.appendingPathComponent("_chat.txt")
        if FileManager.default.fileExists(atPath: direct.path) {
            return direct
        }
        if let androidDirect = try androidNamedChatFile(in: folder) {
            return androidDirect
        }
        let enumerator = FileManager.default.enumerator(
            at: folder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        )
        while let file = enumerator?.nextObject() as? URL {
            if isWhatsAppChatFile(file) {
                return file
            }
        }
        throw CocoaError(.fileNoSuchFile)
    }

    private func androidNamedChatFile(in folder: URL) throws -> URL? {
        let contents = try FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        return contents.first { isWhatsAppChatFile($0) }
    }

    private func isWhatsAppChatFile(_ file: URL) -> Bool {
        let name = file.lastPathComponent
        if name == "_chat.txt" {
            return true
        }
        let lower = name.lowercased()
        return lower.hasSuffix(".txt")
            && !lower.contains("-aud2txt")
            && lower.hasPrefix("whatsapp chat")
    }

    private static func makeTimestampFormatters() -> [DateFormatter] {
        timestampFormats.map { format in
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = format
            return formatter
        }
    }

    private func parseHeader(_ line: String, mediaRoot: URL, sequence: Int, formatters: [DateFormatter]) -> ChatMessage? {
        let cleaned = line.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "\u{200e}", with: "")

        if let match = match(cleaned, pattern: #"^\[(.+?)\]\s*(.*)$"#),
           let timestamp = parseTimestamp(match[1], formatters: formatters) {
            let split = splitSpeaker(match[2])
            return makeMessage(sequence: sequence, sourceFormat: .ios, timestamp: timestamp, speaker: split.speaker, text: split.text, mediaRoot: mediaRoot)
        }

        if let match = match(cleaned, pattern: #"^(.+?)\s+-\s+(.*)$"#),
           let timestamp = parseTimestamp(match[1], formatters: formatters) {
            let split = splitSpeaker(match[2])
            return makeMessage(sequence: sequence, sourceFormat: .android, timestamp: timestamp, speaker: split.speaker, text: split.text, mediaRoot: mediaRoot)
        }

        return nil
    }

    private func makeMessage(sequence: Int, sourceFormat: WhatsAppSourceFormat, timestamp: Date, speaker: String?, text: String, mediaRoot: URL) -> ChatMessage {
        ChatMessage(
            id: "\(mediaRoot.standardizedFileURL.path)#\(sequence)",
            timestamp: timestamp,
            speaker: speaker,
            text: text,
            attachment: attachment(in: text, mediaRoot: mediaRoot),
            sourceFormat: sourceFormat,
            sequenceNumber: sequence
        )
    }

    private func splitSpeaker(_ body: String) -> (speaker: String?, text: String) {
        guard let range = body.range(of: ":") else {
            return (nil, body.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let speaker = String(body[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let remainder = String(body[range.upperBound...])
        if remainder.hasPrefix("//") {
            return (nil, body.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let text = remainder.trimmingCharacters(in: .whitespacesAndNewlines)
        return (speaker.isEmpty ? nil : speaker, text)
    }

    private func attachment(in text: String, mediaRoot: URL) -> ChatAttachment? {
        if let match = match(text, pattern: #"<attached:\s*(.+?)>"#) {
            return makeAttachment(filename: match[1], mediaRoot: mediaRoot)
        }
        if let match = match(text, pattern: #"(\S+\.[A-Za-z0-9]+)\s+\(file attached\)"#, options: [.caseInsensitive]) {
            return makeAttachment(filename: match[1], mediaRoot: mediaRoot)
        }
        return nil
    }

    private func makeAttachment(filename: String, mediaRoot: URL) -> ChatAttachment {
        let trimmed = filename.trimmingCharacters(in: .whitespacesAndNewlines)
        let url = mediaRoot.appendingPathComponent(trimmed)
        let ext = url.pathExtension.lowercased()
        let upper = trimmed.uppercased()
        let isAudio = upper.contains("AUDIO") || upper.hasPrefix("AUD-") || upper.hasPrefix("PTT-") || ["opus", "ogg", "m4a", "mp3", "wav", "aac", "flac", "webm"].contains(ext)
        return ChatAttachment(filename: trimmed, url: url, isAudio: isAudio)
    }

    private func parseTimestamp(_ value: String, formatters: [DateFormatter]) -> Date? {
        let normalized = value.replacingOccurrences(of: "\u{202f}", with: " ")
            .replacingOccurrences(of: "\u{00a0}", with: " ")
        for formatter in formatters {
            if let date = formatter.date(from: normalized) {
                return date
            }
        }
        return nil
    }

    private func match(_ text: String, pattern: String, options: NSRegularExpression.Options = []) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let result = regex.firstMatch(in: text, range: range) else { return nil }
        return (0..<result.numberOfRanges).compactMap { index in
            guard let range = Range(result.range(at: index), in: text) else { return nil }
            return String(text[range])
        }
    }
}
