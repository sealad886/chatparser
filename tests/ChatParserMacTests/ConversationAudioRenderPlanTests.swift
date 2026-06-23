import Foundation
import Testing
@testable import ChatParserMac

struct ConversationAudioRenderPlanTests {
    @Test func jobsCopyExistingAudioAttachmentsInsteadOfSynthesizingMarkerText() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let audioURL = folder.appendingPathComponent("PTT-20260623-WA0000.opus")
        try Data("opus-bytes".utf8).write(to: audioURL)
        let timestamp = Date(timeIntervalSince1970: 1_782_172_800)
        let messages = [
            ChatMessage(
                id: "audio",
                timestamp: timestamp,
                speaker: "Alice",
                text: "PTT-20260623-WA0000.opus (file attached)",
                attachment: ChatAttachment(filename: "PTT-20260623-WA0000.opus", url: audioURL, isAudio: true),
                sourceFormat: .android,
                sequenceNumber: 0
            ),
            ChatMessage(
                id: "text",
                timestamp: timestamp,
                speaker: "Alice",
                text: "hello there",
                attachment: nil,
                sourceFormat: .android,
                sequenceNumber: 1
            )
        ]

        let jobs = ConversationAudioRenderPlan.jobs(
            messages: messages,
            participantProfileIDs: ["Alice": "profile-a"]
        )

        #expect(jobs.count == 2)
        #expect(jobs[0].destinationFilename == "PTT-20260623-WA0000.opus")
        #expect(jobs[0].action == .copyExistingAudio(sourceURL: audioURL))
        #expect(jobs[1].destinationFilename == "PTT-20260623-WA0001.wav")
        #expect(jobs[1].action == .synthesize(profileID: "profile-a", text: "hello there"))
    }

    @Test func jobsSkipMissingAudioAttachmentsInsteadOfSynthesizingMarkerText() {
        let missingURL = FileManager.default.temporaryDirectory.appendingPathComponent("missing.opus")
        let message = ChatMessage(
            id: "missing",
            timestamp: Date(timeIntervalSince1970: 1_782_172_800),
            speaker: "Alice",
            text: "missing.opus (file attached)",
            attachment: ChatAttachment(filename: "missing.opus", url: missingURL, isAudio: true),
            sourceFormat: .android,
            sequenceNumber: 0
        )

        let jobs = ConversationAudioRenderPlan.jobs(
            messages: [message],
            participantProfileIDs: ["Alice": "profile-a"]
        )

        #expect(jobs.isEmpty)
    }

    @Test func progressTextIncludesCurrentFileAndETAAfterProgressStarts() {
        let startedAt = Date(timeIntervalSince1970: 0)
        let now = Date(timeIntervalSince1970: 20)

        let text = ConversationAudioRenderPlan.progressText(
            completed: 2,
            total: 5,
            currentFilename: "PTT-20260623-WA0002.wav",
            startedAt: startedAt,
            now: now
        )

        #expect(text == "2 / 5 - PTT-20260623-WA0002.wav - ETA 30s")
    }
}
