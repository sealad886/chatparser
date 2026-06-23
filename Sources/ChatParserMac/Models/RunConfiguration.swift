import Foundation

struct RunConfiguration: Equatable {
    var inputDirectory: URL?
    var mode: TransformationMode = .transcribeToText
    var voiceboxURL: String = "http://127.0.0.1:17493"
    var model: String = "turbo"
    var profileID: String = ""
    var profileMap: String = ""
    var language: String = "en"
    var forceRedo: Bool = false
    var showProgress: Bool = true

    var isVoiceboxURLValid: Bool {
        Self.isValidVoiceboxURL(voiceboxURL)
    }

    var isLoopbackVoiceboxURL: Bool {
        guard let url = URL(string: voiceboxURL.trimmingCharacters(in: .whitespacesAndNewlines)),
              let host = url.host
        else { return false }
        return ["127.0.0.1", "localhost", "::1"].contains(host)
    }

    static func isValidVoiceboxURL(_ value: String) -> Bool {
        guard let url = URL(string: value.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = url.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              url.host != nil
        else {
            return false
        }
        return true
    }

    var commandArguments: [String] {
        guard let inputDirectory else { return [] }
        var args = [
            "chatparser.py",
            "--input-directory", inputDirectory.path,
            "--to-type", mode.cliValue,
            "--model", model,
            "--voicebox-url", voiceboxURL,
            "--voicebox-language", language
        ]
        if !profileID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--voicebox-profile", profileID]
        }
        profileMap
            .split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .forEach { args += ["--voicebox-profile-map", $0] }
        if forceRedo {
            args.append("--force-redo")
        }
        if showProgress {
            args.append("--progress-bar")
        } else {
            args.append("--no-progress-bar")
        }
        return args
    }
}
