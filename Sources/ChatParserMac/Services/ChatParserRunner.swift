import Foundation

final class ChatParserRunner {
    private let rootURL: URL
    private var process: Process?

    init(rootURL: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)) {
        self.rootURL = rootURL
    }

    func run(
        configuration: RunConfiguration,
        onOutput: @escaping @Sendable (String) -> Void,
        onTermination: @escaping @Sendable (Int32) -> Void
    ) {
        let python = rootURL.appendingPathComponent(".venv/bin/python")
        let executable = FileManager.default.isExecutableFile(atPath: python.path)
            ? python.path
            : "/usr/bin/python3"

        let process = Process()
        process.currentDirectoryURL = rootURL
        process.executableURL = URL(fileURLWithPath: executable)
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
        process?.terminate()
    }
}
