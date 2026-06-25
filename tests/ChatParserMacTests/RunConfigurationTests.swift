import Foundation
import Testing
@testable import ChatParserMac

struct RunConfigurationTests {
    @Test func commandArgumentsDisableProgressWhenShowProgressIsFalse() {
        var configuration = RunConfiguration()
        configuration.inputDirectory = URL(fileURLWithPath: "/tmp/chat-export", isDirectory: true)
        configuration.showProgress = false

        let arguments = configuration.commandArguments

        #expect(arguments.contains("--no-progress-bar"))
        #expect(!arguments.contains("--progress-bar"))
    }

    @Test func commandArgumentsUseSelectedModeForToType() {
        var configuration = RunConfiguration()
        configuration.inputDirectory = URL(fileURLWithPath: "/tmp/chat-export", isDirectory: true)

        configuration.mode = .transcribeToText
        #expect(argumentValue(after: "--to-type", in: configuration.commandArguments) == "text")

        configuration.mode = .synthesizeToAudio
        #expect(argumentValue(after: "--to-type", in: configuration.commandArguments) == "audio")
    }

    @Test func transcriptionModelValidationMatchesPythonCLIChoices() {
        var configuration = RunConfiguration()

        configuration.model = "turbo"
        #expect(configuration.isTranscriptionModelValid)

        configuration.model = "whisper-medium"
        #expect(configuration.isTranscriptionModelValid)

        configuration.model = "large-v3"
        #expect(!configuration.isTranscriptionModelValid)
    }

    private func argumentValue(after flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag),
              arguments.indices.contains(index + 1)
        else { return nil }
        return arguments[index + 1]
    }
}
