import Foundation
import Testing
@testable import ChatParserMac

@MainActor
struct AppStateRunTests {
    @Test func runDoesNotLaunchParserWhenVoiceboxStartupFails() async throws {
        let runner = FakeChatParserRunner()
        let server = FakeVoiceboxServer()
        server.startError = TestRunError.startFailed
        let state = AppState(
            runner: runner,
            voiceboxServer: server,
            voiceboxHealthChecker: FakeVoiceboxHealthChecker(reachable: false)
        )
        state.configuration.inputDirectory = URL(fileURLWithPath: "/tmp/chat-export", isDirectory: true)

        state.run()
        try await Task.sleep(nanoseconds: 20_000_000)

        #expect(runner.runCount == 0)
        #expect(state.isRunning == false)
        #expect(state.lastExitStatus == 1)
        #expect(state.logText.contains("Voicebox is not ready"))
    }

    @Test func cancelDuringVoiceboxStartupPreventsParserLaunch() async throws {
        let runner = FakeChatParserRunner()
        let server = FakeVoiceboxServer()
        server.shouldSuspendStart = true
        let state = AppState(
            runner: runner,
            voiceboxServer: server,
            voiceboxHealthChecker: FakeVoiceboxHealthChecker(reachable: false)
        )
        state.configuration.inputDirectory = URL(fileURLWithPath: "/tmp/chat-export", isDirectory: true)

        state.run()
        while server.startContinuation == nil {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        state.cancel()
        server.finishStart()
        try await Task.sleep(nanoseconds: 20_000_000)

        #expect(runner.runCount == 0)
        #expect(state.isRunning == false)
        #expect(state.logText.contains("Cancellation requested."))
    }

    @Test func transcribeModeDisablesDirectAudioGenerationControls() {
        let state = makeState()
        state.configuration.mode = .transcribeToText
        state.selectedProfileID = "profile-a"
        state.generationText = "hello"
        state.chatMessages = [sampleTextMessage()]
        state.selectedChatMessageID = "message-1"

        #expect(!state.canGenerateSelectedTextAudio)
        #expect(!state.canGenerateSelectedChatMessageAudio)
        #expect(!state.canGenerateConversationAudio)
    }

    @Test func transcribeModeRejectsDirectAudioGenerationActionsBeforePanelsOpen() {
        let state = makeState()
        state.configuration.mode = .transcribeToText
        state.selectedProfileID = "profile-a"
        state.generationText = "manual text"
        state.chatMessages = [sampleTextMessage()]
        state.selectedChatMessageID = "message-1"

        state.generateSelectedText()
        #expect(state.voiceboxMessage == "Switch Mode to Generate Audio before creating audio clips.")

        state.generateConversationAudio()
        #expect(state.chatMessage == "Switch Mode to Generate Audio before creating audio clips.")

        state.generateSelectedChatMessageAudio()
        #expect(state.chatMessage == "Switch Mode to Generate Audio before creating audio clips.")
        #expect(state.generationText == "manual text")
    }

    private func makeState() -> AppState {
        AppState(
            runner: FakeChatParserRunner(),
            voiceboxServer: FakeVoiceboxServer(),
            voiceboxHealthChecker: FakeVoiceboxHealthChecker(reachable: false)
        )
    }

    private func sampleTextMessage() -> ChatMessage {
        ChatMessage(
            id: "message-1",
            timestamp: Date(timeIntervalSince1970: 1_782_172_800),
            speaker: "Alice",
            text: "hello from chat",
            attachment: nil,
            sourceFormat: .android,
            sequenceNumber: 0
        )
    }
}

private enum TestRunError: LocalizedError {
    case startFailed

    var errorDescription: String? {
        "test startup failure"
    }
}

private final class FakeChatParserRunner: ChatParserRunning {
    var runCount = 0
    var cancelCount = 0

    func run(
        configuration: RunConfiguration,
        onOutput: @escaping @Sendable (String) -> Void,
        onTermination: @escaping @Sendable (Int32) -> Void
    ) {
        runCount += 1
    }

    func cancel() {
        cancelCount += 1
    }
}

@MainActor
private final class FakeVoiceboxServer: VoiceboxServerControlling {
    var startCalls = 0
    var stopCalls = 0
    var managed = false
    var startError: Error?
    var shouldSuspendStart = false
    var startContinuation: CheckedContinuation<Void, Error>?

    func start(baseURLString: String, onOutput: @escaping @Sendable (String) -> Void) async throws {
        startCalls += 1
        if let startError {
            throw startError
        }
        if shouldSuspendStart {
            try await withCheckedThrowingContinuation { continuation in
                startContinuation = continuation
            }
        }
    }

    func finishStart() {
        startContinuation?.resume()
        startContinuation = nil
    }

    func stop(baseURLString: String?) {
        stopCalls += 1
    }

    func isManagedServerRunning(baseURLString: String) -> Bool {
        managed
    }
}

private struct FakeVoiceboxHealthChecker: VoiceboxHealthChecking {
    let reachable: Bool

    func isReachable(baseURLString: String) async -> Bool {
        reachable
    }
}
