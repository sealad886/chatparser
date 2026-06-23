import Foundation
import Darwin

enum VoiceboxServerError: LocalizedError {
    case nonLoopbackURL(String)
    case missingSubmodule(URL)
    case missingPython(URL)
    case startFailed(String)
    case timedOut

    var errorDescription: String? {
        switch self {
        case .nonLoopbackURL(let value):
            "ChatParser can only auto-start Voicebox for loopback URLs. Current URL: \(value)"
        case .missingSubmodule(let url):
            "Voicebox submodule is missing at \(url.path). Run: git submodule update --init --recursive external/voicebox"
        case .missingPython(let url):
            "Voicebox backend venv is missing at \(url.path). Run: ./script/setup_voicebox.sh"
        case .startFailed(let detail):
            "Voicebox server failed to start: \(detail)"
        case .timedOut:
            "Voicebox server did not become ready before timeout."
        }
    }
}

struct VoiceboxManagedProcessRecord: Codable, Equatable {
    let pid: Int32
    let host: String
    let port: Int
    let startedAt: Date

    func matches(host: String, port: Int) -> Bool {
        self.host == host && self.port == port
    }
}

protocol VoiceboxProcessInspecting: Sendable {
    func processExists(pid: Int32) -> Bool
    func commandLine(pid: Int32) -> String?
    func terminate(pid: Int32)
}

struct SystemVoiceboxProcessInspector: VoiceboxProcessInspecting {
    func processExists(pid: Int32) -> Bool {
        kill(pid, 0) == 0 || errno == EPERM
    }

    func commandLine(pid: Int32) -> String? {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-p", String(pid), "-o", "command="]
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }

    func terminate(pid: Int32) {
        kill(pid, SIGTERM)
    }
}

@MainActor
final class VoiceboxServerController {
    private let rootURL: URL
    private let processInspector: any VoiceboxProcessInspecting
    private var process: Process?
    private var processEndpoint: (host: String, port: Int)?
    private var outputPipe: Pipe?
    private var logFileHandle: FileHandle?

    init(
        rootURL: URL = VoiceboxServerController.resolveRepositoryRoot(),
        processInspector: any VoiceboxProcessInspecting = SystemVoiceboxProcessInspector()
    ) {
        self.rootURL = rootURL
        self.processInspector = processInspector
    }

    var isRunning: Bool {
        process?.isRunning ?? false
    }

    func start(baseURLString: String, onOutput: @escaping @Sendable (String) -> Void) async throws {
        guard !isRunning else { return }
        let endpoint = try endpointInfo(from: baseURLString)
        guard managedProcessRecordIfRunning(matching: endpoint) == nil else { return }
        let voiceboxURL = rootURL.appendingPathComponent("external/voicebox")
        let backendPython = voiceboxURL.appendingPathComponent("backend/venv/bin/python")
        guard FileManager.default.fileExists(atPath: voiceboxURL.appendingPathComponent("backend/main.py").path) else {
            throw VoiceboxServerError.missingSubmodule(voiceboxURL)
        }
        guard FileManager.default.isExecutableFile(atPath: backendPython.path) else {
            throw VoiceboxServerError.missingPython(backendPython)
        }

        let process = Process()
        process.currentDirectoryURL = voiceboxURL
        process.executableURL = backendPython
        process.arguments = [
            "-m", "backend.main",
            "--host", endpoint.host,
            "--port", String(endpoint.port)
        ]

        let pipe = Pipe()
        let logFile = try prepareLogFile()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            logFile.write(data)
            onOutput(text)
        }
        process.terminationHandler = { [weak self] process in
            pipe.fileHandleForReading.readabilityHandler = nil
            let message = "Voicebox server exited with status \(process.terminationStatus)\n"
            if let data = message.data(using: .utf8) {
                logFile.write(data)
            }
            try? logFile.close()
            onOutput("Voicebox server exited with status \(process.terminationStatus)\n")
            Task { @MainActor in
                self?.process = nil
                self?.processEndpoint = nil
                self?.outputPipe = nil
                self?.logFileHandle = nil
            }
        }

        do {
            try process.run()
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            try? logFile.close()
            throw VoiceboxServerError.startFailed(error.localizedDescription)
        }

        self.process = process
        self.processEndpoint = endpoint
        self.outputPipe = pipe
        self.logFileHandle = logFile
        do {
            try writeManagedProcessRecord(
                VoiceboxManagedProcessRecord(
                    pid: process.processIdentifier,
                    host: endpoint.host,
                    port: endpoint.port,
                    startedAt: Date()
                )
            )
        } catch {
            onOutput("Could not write Voicebox server ownership record: \(error.localizedDescription)\n")
        }
        onOutput("Voicebox server log: \(logFileURL.path)\n")
        try await waitUntilReady(baseURLString: baseURLString)
    }

    func stop(baseURLString: String? = nil) {
        if let process {
            let processID = process.processIdentifier
            process.terminate()
            clearManagedProcessRecord(processID: processID)
        } else if let baseURLString,
                  let endpoint = try? endpointInfo(from: baseURLString),
                  let record = managedProcessRecordIfRunning(matching: endpoint) {
            processInspector.terminate(pid: record.pid)
            clearManagedProcessRecord(processID: record.pid)
        }
        process = nil
        processEndpoint = nil
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        outputPipe = nil
        try? logFileHandle?.close()
        logFileHandle = nil
    }

    var logFileURL: URL {
        rootURL.appendingPathComponent("logs/voicebox-server.log")
    }

    var managedProcessRecordURL: URL {
        rootURL.appendingPathComponent("logs/voicebox-server.json")
    }

    func isManagedServerRunning(baseURLString: String) -> Bool {
        guard let endpoint = try? endpointInfo(from: baseURLString) else { return false }
        return managedProcessRecordIfRunning(matching: endpoint) != nil
    }

    private func waitUntilReady(baseURLString: String) async throws {
        let deadline = Date().addingTimeInterval(45)
        while Date() < deadline {
            do {
                try await VoiceboxAPI(baseURLString: baseURLString).health()
                return
            } catch {
                try await Task.sleep(nanoseconds: 500_000_000)
            }
        }
        throw VoiceboxServerError.timedOut
    }

    private func endpointInfo(from value: String) throws -> (host: String, port: Int) {
        guard let url = URL(string: value.trimmingCharacters(in: .whitespacesAndNewlines)),
              let host = url.host,
              ["127.0.0.1", "localhost", "::1"].contains(host)
        else {
            throw VoiceboxServerError.nonLoopbackURL(value)
        }
        return (host == "localhost" ? "127.0.0.1" : host, url.port ?? 17493)
    }

    private func prepareLogFile() throws -> FileHandle {
        let fileManager = FileManager.default
        let logURL = logFileURL
        try fileManager.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fileManager.fileExists(atPath: logURL.path) {
            fileManager.createFile(atPath: logURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: logURL)
        try handle.seekToEnd()
        let marker = "\n--- Voicebox server start \(Date()) ---\n"
        if let data = marker.data(using: .utf8) {
            handle.write(data)
        }
        return handle
    }

    private func managedProcessRecordIfRunning(matching endpoint: (host: String, port: Int)) -> VoiceboxManagedProcessRecord? {
        if let process,
           process.isRunning,
           let processEndpoint,
           processEndpoint.host == endpoint.host,
           processEndpoint.port == endpoint.port {
            return VoiceboxManagedProcessRecord(
                pid: process.processIdentifier,
                host: endpoint.host,
                port: endpoint.port,
                startedAt: Date()
            )
        }

        if let record = readManagedProcessRecord(),
           record.matches(host: endpoint.host, port: endpoint.port) {
            if isManagedBackendProcessRunning(record, endpoint: endpoint) {
                return record
            }
            clearManagedProcessRecord(processID: record.pid)
        }

        if let record = recoverManagedProcessRecordFromLog(endpoint: endpoint),
           isManagedBackendProcessRunning(record, endpoint: endpoint) {
            try? writeManagedProcessRecord(record)
            return record
        }

        return nil
    }

    private func isManagedBackendProcessRunning(
        _ record: VoiceboxManagedProcessRecord,
        endpoint: (host: String, port: Int)
    ) -> Bool {
        guard processInspector.processExists(pid: record.pid),
              let command = processInspector.commandLine(pid: record.pid)
        else { return false }

        return command.contains("backend.main")
            && command.contains("--host \(endpoint.host)")
            && command.contains("--port \(endpoint.port)")
    }

    private func readManagedProcessRecord() -> VoiceboxManagedProcessRecord? {
        guard let data = try? Data(contentsOf: managedProcessRecordURL) else { return nil }
        return try? JSONDecoder().decode(VoiceboxManagedProcessRecord.self, from: data)
    }

    private func writeManagedProcessRecord(_ record: VoiceboxManagedProcessRecord) throws {
        try FileManager.default.createDirectory(
            at: managedProcessRecordURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONEncoder().encode(record)
        try data.write(to: managedProcessRecordURL, options: .atomic)
    }

    private func clearManagedProcessRecord(processID: Int32?) {
        guard let record = readManagedProcessRecord() else { return }
        guard processID == nil || record.pid == processID else { return }
        try? FileManager.default.removeItem(at: managedProcessRecordURL)
    }

    private func recoverManagedProcessRecordFromLog(endpoint: (host: String, port: Int)) -> VoiceboxManagedProcessRecord? {
        guard let text = try? String(contentsOf: logFileURL, encoding: .utf8),
              let regex = try? NSRegularExpression(pattern: #"Started server process \[(\d+)\]"#)
        else { return nil }

        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.matches(in: text, range: range).last,
              let pidRange = Range(match.range(at: 1), in: text),
              let pid = Int32(text[pidRange])
        else { return nil }

        return VoiceboxManagedProcessRecord(
            pid: pid,
            host: endpoint.host,
            port: endpoint.port,
            startedAt: Date()
        )
    }

    nonisolated private static func resolveRepositoryRoot() -> URL {
        let fileManager = FileManager.default
        let current = URL(fileURLWithPath: fileManager.currentDirectoryPath)
        if fileManager.fileExists(atPath: current.appendingPathComponent("chatparser.py").path) {
            return current
        }

        var candidate = Bundle.main.bundleURL
        for _ in 0..<8 {
            if fileManager.fileExists(atPath: candidate.appendingPathComponent("chatparser.py").path) {
                return candidate
            }
            let parent = candidate.deletingLastPathComponent()
            if parent.path == candidate.path { break }
            candidate = parent
        }

        return current
    }
}
