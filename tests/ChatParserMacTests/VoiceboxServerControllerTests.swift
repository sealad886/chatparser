import Foundation
import Testing
@testable import ChatParserMac

@MainActor
struct VoiceboxServerControllerTests {
    @Test func recoversManagedServerFromExistingLog() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeServerLog(root: root, pid: 12345)
        let inspector = FakeProcessInspector(
            existingPIDs: [12345],
            commands: [12345: "/opt/homebrew/bin/python -m backend.main --host 127.0.0.1 --port 17493"]
        )
        let controller = VoiceboxServerController(rootURL: root, processInspector: inspector)

        #expect(controller.isManagedServerRunning(baseURLString: "http://127.0.0.1:17493"))
        #expect(FileManager.default.fileExists(atPath: controller.managedProcessRecordURL.path))
    }

    @Test func rejectsRecoveredLogPIDWhenEndpointDoesNotMatch() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeServerLog(root: root, pid: 12345)
        let inspector = FakeProcessInspector(
            existingPIDs: [12345],
            commands: [12345: "/opt/homebrew/bin/python -m backend.main --host 127.0.0.1 --port 17493"]
        )
        let controller = VoiceboxServerController(rootURL: root, processInspector: inspector)

        #expect(!controller.isManagedServerRunning(baseURLString: "http://127.0.0.1:17494"))
    }

    @Test func endpointMismatchDoesNotDeleteExistingManagedRecord() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let logs = root.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let recordURL = logs.appendingPathComponent("voicebox-server.json")
        let record = VoiceboxManagedProcessRecord(
            pid: 24680,
            host: "127.0.0.1",
            port: 17493,
            startedAt: Date(timeIntervalSince1970: 0)
        )
        try JSONEncoder().encode(record).write(to: recordURL)
        let inspector = FakeProcessInspector(
            existingPIDs: [24680],
            commands: [24680: "/opt/homebrew/bin/python -m backend.main --host 127.0.0.1 --port 17493"]
        )
        let controller = VoiceboxServerController(rootURL: root, processInspector: inspector)

        #expect(!controller.isManagedServerRunning(baseURLString: "http://127.0.0.1:17494"))
        #expect(FileManager.default.fileExists(atPath: recordURL.path))
    }

    @Test func stopTerminatesAdoptedManagedServer() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let logs = root.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let record = VoiceboxManagedProcessRecord(
            pid: 24680,
            host: "127.0.0.1",
            port: 17493,
            startedAt: Date(timeIntervalSince1970: 0)
        )
        let recordURL = logs.appendingPathComponent("voicebox-server.json")
        try JSONEncoder().encode(record).write(to: recordURL)
        let inspector = FakeProcessInspector(
            existingPIDs: [24680],
            commands: [24680: "/opt/homebrew/bin/python -m backend.main --host 127.0.0.1 --port 17493"]
        )
        let controller = VoiceboxServerController(rootURL: root, processInspector: inspector)

        controller.stop(baseURLString: "http://127.0.0.1:17493")

        #expect(inspector.terminatedPIDs == [24680])
        #expect(!FileManager.default.fileExists(atPath: recordURL.path))
    }

    private func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func writeServerLog(root: URL, pid: Int32) throws {
        let logs = root.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        try """
        --- Voicebox server start 2026-06-23 13:08:19 +0000 ---
        INFO:     Started server process [\(pid)]
        INFO:     Uvicorn running on http://127.0.0.1:17493 (Press CTRL+C to quit)
        """.write(to: logs.appendingPathComponent("voicebox-server.log"), atomically: true, encoding: .utf8)
    }
}

private final class FakeProcessInspector: VoiceboxProcessInspecting, @unchecked Sendable {
    private let existingPIDs: Set<Int32>
    private let commands: [Int32: String]
    private(set) var terminatedPIDs: [Int32] = []

    init(existingPIDs: Set<Int32>, commands: [Int32: String]) {
        self.existingPIDs = existingPIDs
        self.commands = commands
    }

    func processExists(pid: Int32) -> Bool {
        existingPIDs.contains(pid)
    }

    func commandLine(pid: Int32) -> String? {
        commands[pid]
    }

    func terminate(pid: Int32) {
        terminatedPIDs.append(pid)
    }
}
