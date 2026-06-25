import Foundation
import Testing
@testable import ChatParserMac

struct WhatsAppExportParserTests {
    @Test func parseExportFindsAndroidNamedChatFile() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let chatFile = folder.appendingPathComponent("WhatsApp Chat with Alice.txt")
        let audioFile = folder.appendingPathComponent("PTT-20260530-WA0000.opus")
        try """
        30/05/2026, 00:32 - Alice: Android text message
        30/05/2026, 00:33 - Bob: PTT-20260530-WA0000.opus (file attached)
        continuation text
        """.write(to: chatFile, atomically: true, encoding: .utf8)
        try Data("audio".utf8).write(to: audioFile)

        let messages = try WhatsAppExportParser().parseExport(at: folder)

        #expect(messages.count == 2)
        #expect(messages[0].speaker == "Alice")
        #expect(messages[0].text == "Android text message")
        #expect(messages[0].sourceFormat == .android)
        #expect(messages[0].sequenceNumber == 0)
        #expect(messages[0].generatedAudioFilename == "PTT-20260530-WA0000.wav")
        #expect(messages[1].speaker == "Bob")
        #expect(messages[1].text == "PTT-20260530-WA0000.opus (file attached)\ncontinuation text")
        #expect(messages[1].sourceFormat == .android)
        #expect(messages[1].sequenceNumber == 1)
        #expect(messages[1].generatedAudioFilename == "PTT-20260530-WA0001.wav")
        #expect(messages[1].attachment?.filename == "PTT-20260530-WA0000.opus")
        #expect(messages[1].attachment?.isAudio == true)
    }

    @Test func parsePreservesAndroidAttachmentFilenameWithSpaces() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let filename = "WhatsApp Audio 2024-12-31 at 23.05.00.opus"

        let messages = WhatsAppExportParser().parse(
            "31/12/2024, 23:05 - Alice: \(filename) (file attached)\n",
            mediaRoot: folder
        )

        #expect(messages.count == 1)
        #expect(messages[0].attachment?.filename == filename)
        #expect(messages[0].attachment?.url.lastPathComponent == filename)
        #expect(messages[0].attachment?.isAudio == true)
    }

    @Test func parsePreservesAndroidImageAndDocumentFilenamesWithSpaces() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let imageFilename = "WhatsApp Image 2024-12-31 at 23.05.00.jpeg"
        let documentFilename = "WhatsApp Document 2024-12-31 at 23.05.01.pdf"

        let messages = WhatsAppExportParser().parse(
            """
            31/12/2024, 23:05 - Alice: \(imageFilename) (file attached)
            31/12/2024, 23:06 - Alice: \(documentFilename) (file attached)
            """,
            mediaRoot: folder
        )

        #expect(messages.count == 2)
        #expect(messages[0].attachment?.filename == imageFilename)
        #expect(messages[0].attachment?.url.lastPathComponent == imageFilename)
        #expect(messages[0].attachment?.isAudio == false)
        #expect(messages[1].attachment?.filename == documentFilename)
        #expect(messages[1].attachment?.url.lastPathComponent == documentFilename)
        #expect(messages[1].attachment?.isAudio == false)
    }

    @Test func parseContinuationAttachmentMarkerBelongsToCurrentMessage() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let messages = WhatsAppExportParser().parse(
            "[31/12/2024, 23:05:07] Alice: voice note caption\n"
                + "<attached: clip.ogg>\n",
            mediaRoot: folder
        )

        #expect(messages.count == 1)
        #expect(messages[0].text == "voice note caption\n<attached: clip.ogg>")
        #expect(messages[0].attachment?.filename == "clip.ogg")
        #expect(messages[0].attachment?.isAudio == true)
    }

    @Test func generatedAudioFilenameMatchesIOSExportNaming() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let messages = WhatsAppExportParser().parse(
            "[30/05/2026, 00:32:00] Alice: iOS text message\n",
            mediaRoot: folder
        )

        #expect(messages.count == 1)
        #expect(messages[0].sourceFormat == .ios)
        #expect(messages[0].sequenceNumber == 0)
        #expect(messages[0].generatedAudioFilename == "00000001-AUDIO-2026-05-30-00-32-00.wav")
    }

    @Test func parseExportFindsNestedAndroidNamedChatFile() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let folder = root.appendingPathComponent("00000179-WhatsApp Chat with Alice", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let chatFile = folder.appendingPathComponent("WhatsApp Chat with Alice.txt")
        try "30/05/2026, 00:32 - Alice: Android text message\n"
            .write(to: chatFile, atomically: true, encoding: .utf8)

        let messages = try WhatsAppExportParser().parseExport(at: root)

        #expect(messages.count == 1)
        #expect(messages[0].speaker == "Alice")
        #expect(messages[0].text == "Android text message")
    }

    @Test func parseExportPreservesAndroidSpeakerWhenMessageIsEmpty() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let chatFile = folder.appendingPathComponent("WhatsApp Chat with Alice.txt")
        try """
        30/05/2026, 00:32 - M:
        30/05/2026, 00:33 - Alice:No leading space
        30/05/2026, 00:34 - https://example.invalid
        """.write(to: chatFile, atomically: true, encoding: .utf8)

        let messages = try WhatsAppExportParser().parseExport(at: folder)

        #expect(messages.count == 3)
        #expect(messages[0].speaker == "M")
        #expect(messages[0].text == "")
        #expect(messages[1].speaker == "Alice")
        #expect(messages[1].text == "No leading space")
        #expect(messages[2].speaker == nil)
        #expect(messages[2].text == "https://example.invalid")
    }

    @Test func parseExportAppendsContinuationWithoutLeadingNewlineWhenHeaderTextIsEmpty() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let chatFile = folder.appendingPathComponent("WhatsApp Chat with Alice.txt")
        try """
        30/05/2026, 00:32 - M:
        continued message text
        """.write(to: chatFile, atomically: true, encoding: .utf8)

        let messages = try WhatsAppExportParser().parseExport(at: folder)

        #expect(messages.count == 1)
        #expect(messages[0].speaker == "M")
        #expect(messages[0].text == "continued message text")
    }
}
