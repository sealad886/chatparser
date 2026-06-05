import Foundation

struct ChatAttachment: Identifiable, Equatable {
    let id = UUID()
    let filename: String
    let url: URL
    let isAudio: Bool

    var isImage: Bool {
        ["jpg", "jpeg", "png", "gif", "heic", "webp"].contains(url.pathExtension.lowercased())
    }

    var isVideo: Bool {
        ["mp4", "mov", "m4v", "3gp", "webm"].contains(url.pathExtension.lowercased())
    }
}

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let timestamp: Date
    let speaker: String?
    let text: String
    let attachment: ChatAttachment?

    var participant: String {
        speaker ?? "System"
    }
}
