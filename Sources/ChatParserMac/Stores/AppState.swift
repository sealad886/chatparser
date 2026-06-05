import AppKit
import Foundation

@MainActor
final class AppState: ObservableObject {
    @Published var configuration = RunConfiguration()
    @Published var logText = ""
    @Published var isRunning = false
    @Published var lastExitStatus: Int32?

    private let runner = ChatParserRunner()

    var canRun: Bool {
        guard configuration.inputDirectory != nil, !isRunning else { return false }
        if configuration.mode == .synthesizeToAudio {
            return !configuration.profileID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !configuration.profileMap.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return true
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
}
