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
}

struct ChatMessage: Identifiable, Equatable {
    let timestamp: Date
    let speaker: String?
    let text: String
    let attachment: ChatAttachment?

    var id: String {
        [
            timestamp.ISO8601Format(),
            speaker ?? "System",
            text,
            attachment?.id ?? ""
        ].joined(separator: "|")
    }

    var participant: String {
        speaker ?? "System"
    }
}
