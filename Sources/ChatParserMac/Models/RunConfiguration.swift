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
        }
        return args
    }
}
