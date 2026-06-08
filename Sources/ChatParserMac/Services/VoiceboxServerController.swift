import Foundation

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

@MainActor
final class VoiceboxServerController {
    private let rootURL: URL
    private var process: Process?
    private var outputPipe: Pipe?

    init(rootURL: URL = VoiceboxServerController.resolveRepositoryRoot()) {
        self.rootURL = rootURL
    }

    var isRunning: Bool {
        process?.isRunning ?? false
    }

    func start(baseURLString: String, onOutput: @escaping @Sendable (String) -> Void) async throws {
        guard !isRunning else { return }
        let endpoint = try endpointInfo(from: baseURLString)
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
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            onOutput(text)
        }
        process.terminationHandler = { [weak self] process in
            pipe.fileHandleForReading.readabilityHandler = nil
            onOutput("Voicebox server exited with status \(process.terminationStatus)\n")
            Task { @MainActor in
                self?.process = nil
                self?.outputPipe = nil
            }
        }

        do {
            try process.run()
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            throw VoiceboxServerError.startFailed(error.localizedDescription)
        }

        self.process = process
        self.outputPipe = pipe
        try await waitUntilReady(baseURLString: baseURLString)
    }

    func stop() {
        process?.terminate()
        process = nil
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        outputPipe = nil
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
