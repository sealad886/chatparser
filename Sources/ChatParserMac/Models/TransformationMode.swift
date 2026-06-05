import Foundation

enum TransformationMode: String, CaseIterable, Identifiable {
    case transcribeToText
    case synthesizeToAudio

    var id: String { rawValue }

    var title: String {
        switch self {
        case .transcribeToText:
            "Transcribe Audio"
        case .synthesizeToAudio:
            "Generate Audio"
        }
    }

    var cliValue: String {
        switch self {
        case .transcribeToText:
            "text"
        case .synthesizeToAudio:
            "audio"
        }
    }
}
