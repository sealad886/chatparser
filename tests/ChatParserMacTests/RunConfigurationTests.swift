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
}
