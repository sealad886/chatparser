import Foundation
import Testing
@testable import ChatParserMac

@Suite(.serialized)
struct VoiceboxAPITests {
    @Test func generateSpeechPollsAgainWhenStatusStreamEndsBeforeTerminalEvent() async throws {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let session = URLSession(configuration: Self.urlSessionConfiguration { request in
            let path = request.url?.path ?? ""
            switch (request.httpMethod, path) {
            case ("POST", "/generate"):
                return Self.jsonResponse(
                    path: path,
                    body: [
                        "id": "gen-123",
                        "profile_id": "profile-123",
                        "text": "hello",
                        "language": "en",
                        "status": "generating",
                        "audio_path": ""
                    ]
                )
            case ("GET", "/generate/gen-123/status"):
                Self.statusRequests += 1
                if Self.statusRequests == 1 {
                    return Self.eventStream(path: path, body: #"data: {"id":"gen-123","status":"generating"}"# + "\n\n")
                }
                return Self.eventStream(path: path, body: #"data: {"id":"gen-123","status":"completed"}"# + "\n\n")
            case ("GET", "/audio/gen-123"):
                return Self.response(path: path, contentType: "audio/wav", body: Data("wav-bytes".utf8))
            default:
                throw TestHTTPError.unexpectedRequest("\(request.httpMethod ?? "") \(path)")
            }
        })

        Self.statusRequests = 0
        let api = try VoiceboxAPI(
            baseURLString: "http://127.0.0.1:17493",
            session: session,
            generationPollInterval: 0,
            maxGenerationWaitSeconds: 1
        )

        let written = try await api.generateSpeech(
            profileID: "profile-123",
            text: "hello",
            language: "en",
            destination: outputURL
        )

        #expect(written == outputURL)
        #expect(try Data(contentsOf: outputURL) == Data("wav-bytes".utf8))
        #expect(Self.statusRequests == 2)
    }

    @Test func generateSpeechSurfacesTerminalFailureFromStatusStream() async throws {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let session = URLSession(configuration: Self.urlSessionConfiguration { request in
            let path = request.url?.path ?? ""
            switch (request.httpMethod, path) {
            case ("POST", "/generate"):
                return Self.jsonResponse(
                    path: path,
                    body: [
                        "id": "gen-failed",
                        "profile_id": "profile-123",
                        "text": "hello",
                        "language": "en",
                        "status": "generating",
                        "audio_path": ""
                    ]
                )
            case ("GET", "/generate/gen-failed/status"):
                return Self.eventStream(
                    path: path,
                    body: #"data: {"id":"gen-failed","status":"failed","error":"model unavailable"}"# + "\n\n"
                )
            default:
                throw TestHTTPError.unexpectedRequest("\(request.httpMethod ?? "") \(path)")
            }
        })

        let api = try VoiceboxAPI(
            baseURLString: "http://127.0.0.1:17493",
            session: session,
            generationPollInterval: 0,
            maxGenerationWaitSeconds: 1
        )

        do {
            _ = try await api.generateSpeech(
                profileID: "profile-123",
                text: "hello",
                language: "en",
                destination: outputURL
            )
            Issue.record("Expected generateSpeech to throw")
        } catch {
            #expect(error.localizedDescription == "Voicebox generation failed: model unavailable")
        }
        #expect(!FileManager.default.fileExists(atPath: outputURL.path))
    }

    @Test func generateSpeechRejectsEmptyAudioResponse() async throws {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let session = URLSession(configuration: Self.urlSessionConfiguration { request in
            let path = request.url?.path ?? ""
            switch (request.httpMethod, path) {
            case ("POST", "/generate"):
                return Self.jsonResponse(
                    path: path,
                    body: [
                        "id": "gen-empty",
                        "profile_id": "profile-123",
                        "text": "hello",
                        "language": "en",
                        "status": "completed",
                        "audio_path": ""
                    ]
                )
            case ("GET", "/audio/gen-empty"):
                return Self.response(path: path, contentType: "audio/wav", body: Data())
            default:
                throw TestHTTPError.unexpectedRequest("\(request.httpMethod ?? "") \(path)")
            }
        })

        let api = try VoiceboxAPI(baseURLString: "http://127.0.0.1:17493", session: session)

        do {
            _ = try await api.generateSpeech(
                profileID: "profile-123",
                text: "hello",
                language: "en",
                destination: outputURL
            )
            Issue.record("Expected generateSpeech to reject empty audio")
        } catch {
            #expect(error.localizedDescription == "Voicebox returned an empty audio response.")
        }
        #expect(!FileManager.default.fileExists(atPath: outputURL.path))
    }

    @Test func generateSpeechRejectsNonAudioResponse() async throws {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        let session = URLSession(configuration: Self.urlSessionConfiguration { request in
            let path = request.url?.path ?? ""
            switch (request.httpMethod, path) {
            case ("POST", "/generate"):
                return Self.jsonResponse(
                    path: path,
                    body: [
                        "id": "gen-json",
                        "profile_id": "profile-123",
                        "text": "hello",
                        "language": "en",
                        "status": "completed",
                        "audio_path": ""
                    ]
                )
            case ("GET", "/audio/gen-json"):
                return Self.response(
                    path: path,
                    contentType: "application/json",
                    body: Data(#"{"detail":"not ready"}"#.utf8)
                )
            default:
                throw TestHTTPError.unexpectedRequest("\(request.httpMethod ?? "") \(path)")
            }
        })

        let api = try VoiceboxAPI(baseURLString: "http://127.0.0.1:17493", session: session)

        do {
            _ = try await api.generateSpeech(
                profileID: "profile-123",
                text: "hello",
                language: "en",
                destination: outputURL
            )
            Issue.record("Expected generateSpeech to reject non-audio response")
        } catch {
            #expect(error.localizedDescription == "Voicebox returned non-audio content from /audio: application/json")
        }
        #expect(!FileManager.default.fileExists(atPath: outputURL.path))
    }

    private static var statusRequests = 0

    private static func urlSessionConfiguration(
        handler: @escaping @Sendable (URLRequest) throws -> (HTTPURLResponse, Data)
    ) -> URLSessionConfiguration {
        MockURLProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return configuration
    }

    private static func jsonResponse(path: String, body: [String: String]) -> (HTTPURLResponse, Data) {
        response(path: path, contentType: "application/json", body: try! JSONEncoder().encode(body))
    }

    private static func eventStream(path: String, body: String) -> (HTTPURLResponse, Data) {
        response(path: path, contentType: "text/event-stream", body: Data(body.utf8))
    }

    private static func response(path: String, contentType: String, body: Data) -> (HTTPURLResponse, Data) {
        let url = URL(string: "http://127.0.0.1:17493\(path)")!
        let response = HTTPURLResponse(
            url: url,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": contentType]
        )!
        return (response, body)
    }
}

private enum TestHTTPError: Error {
    case unexpectedRequest(String)
}

private final class MockURLProtocol: URLProtocol, @unchecked Sendable {
    static var handler: (@Sendable (URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: TestHTTPError.unexpectedRequest("missing handler"))
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
