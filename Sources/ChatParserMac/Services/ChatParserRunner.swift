import Foundation
import Darwin

protocol ChatParserRunning: AnyObject {
    func run(
        configuration: RunConfiguration,
        onOutput: @escaping @Sendable (String) -> Void,
        onTermination: @escaping @Sendable (Int32) -> Void
    )
    func cancel()
}

final class ChatParserRunner: ChatParserRunning {
    private let rootURL: URL
    private var process: Process?

    init(rootURL: URL = ChatParserRunner.resolveRepositoryRoot()) {
        self.rootURL = rootURL
    }

    private static func resolveRepositoryRoot() -> URL {
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

    func run(
        configuration: RunConfiguration,
        onOutput: @escaping @Sendable (String) -> Void,
        onTermination: @escaping @Sendable (Int32) -> Void
    ) {
        let python = rootURL.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            onOutput("Missing repo virtual environment: \(python.path)\nRun: python3 -m venv .venv && . .venv/bin/activate && python -m pip install -r requirements.txt\n")
            onTermination(127)
            return
        }

        let process = Process()
        process.currentDirectoryURL = rootURL
        process.executableURL = python
        process.arguments = configuration.commandArguments

        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        outputPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            onOutput(text)
        }

        process.terminationHandler = { process in
            outputPipe.fileHandleForReading.readabilityHandler = nil
            onTermination(process.terminationStatus)
        }

        self.process = process

        do {
            try process.run()
        } catch {
            outputPipe.fileHandleForReading.readabilityHandler = nil
            onOutput("Failed to start ChatParser: \(error.localizedDescription)\n")
            onTermination(127)
        }
    }

    func cancel() {
        guard let process, process.isRunning else { return }
        if kill(process.processIdentifier, SIGINT) != 0 {
            process.terminate()
        }
    }
}
