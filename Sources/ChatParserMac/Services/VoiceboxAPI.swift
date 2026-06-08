import Foundation

enum VoiceboxAPIError: LocalizedError {
    case invalidBaseURL(String)
    case invalidResponse
    case http(Int, String)
    case missingGenerationID
    case generationFailed(String)
    case generationTimedOut

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL(let value):
            "Invalid Voicebox URL: \(value)"
        case .invalidResponse:
            "Voicebox returned an invalid response."
        case .http(let status, let body):
            "Voicebox HTTP \(status): \(body)"
        case .missingGenerationID:
            "Voicebox did not return a generation id."
        case .generationFailed(let detail):
            "Voicebox generation failed: \(detail)"
        case .generationTimedOut:
            "Voicebox generation timed out."
        }
    }
}

final class VoiceboxAPI {
    private let baseURL: URL
    private let session: URLSession
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    init(baseURLString: String, session: URLSession = .shared) throws {
        guard let url = URL(string: baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = url.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              url.host != nil
        else {
            throw VoiceboxAPIError.invalidBaseURL(baseURLString)
        }
        self.baseURL = url
        self.session = session
    }

    func health() async throws {
        let _: EmptyResponse = try await jsonRequest("GET", path: "/health")
    }

    func listProfiles() async throws -> [VoiceProfile] {
        try await jsonRequest("GET", path: "/profiles")
    }

    func createProfile(name: String, description: String?, language: String, personality: String?) async throws -> VoiceProfile {
        let payload = profilePayload(name: name, description: description, language: language, personality: personality)
        return try await jsonRequest("POST", path: "/profiles", jsonBody: payload)
    }

    func updateProfile(_ profileID: String, name: String, description: String?, language: String, personality: String?) async throws -> VoiceProfile {
        let payload = profilePayload(name: name, description: description, language: language, personality: personality)
        return try await jsonRequest("PUT", path: "/profiles/\(profileID)", jsonBody: payload)
    }

    func deleteProfile(_ profileID: String) async throws {
        let _: EmptyResponse = try await jsonRequest("DELETE", path: "/profiles/\(profileID)")
    }

    func listSamples(profileID: String) async throws -> [ProfileSample] {
        try await jsonRequest("GET", path: "/profiles/\(profileID)/samples")
    }

    func addSample(profileID: String, fileURL: URL, referenceText: String) async throws -> ProfileSample {
        let body = try await multipartBody(
            fields: ["reference_text": referenceText],
            files: [MultipartFile(fieldName: "file", url: fileURL)]
        )
        return try await multipartRequest("POST", path: "/profiles/\(profileID)/samples", multipart: body)
    }

    func updateSample(_ sampleID: String, referenceText: String) async throws -> ProfileSample {
        try await jsonRequest("PUT", path: "/profiles/samples/\(sampleID)", jsonBody: ["reference_text": referenceText])
    }

    func deleteSample(_ sampleID: String) async throws {
        let _: EmptyResponse = try await jsonRequest("DELETE", path: "/profiles/samples/\(sampleID)")
    }

    func generateSpeech(
        profileID: String,
        text: String,
        language: String,
        destination: URL,
        onGenerationID: ((String) async -> Void)? = nil
    ) async throws -> URL {
        let response: GenerationResponse = try await jsonRequest(
            "POST",
            path: "/generate",
            jsonBody: ["profile_id": profileID, "text": text, "language": language]
        )
        await onGenerationID?(response.id)
        if (response.status ?? "").lowercased() != "completed" {
            try await waitForGeneration(response.id)
        }
        let audio = try await rawRequest("GET", path: "/audio/\(response.id)")
        try audio.write(to: destination, options: .atomic)
        return destination
    }

    func cancelGeneration(_ generationID: String) async throws {
        let _: EmptyResponse = try await jsonRequest("POST", path: "/generate/\(generationID)/cancel")
    }

    private func profilePayload(name: String, description: String?, language: String, personality: String?) -> [String: String?] {
        [
            "name": name,
            "description": description,
            "language": language,
            "voice_type": "cloned",
            "personality": personality
        ]
    }

    private func waitForGeneration(_ generationID: String) async throws {
        let status = try await generationStatus(generationID)
        switch status.status.lowercased() {
        case "completed":
            return
        case "failed", "cancelled", "canceled", "error", "not_found":
            throw VoiceboxAPIError.generationFailed(status.error ?? status.status)
        default:
            throw VoiceboxAPIError.generationTimedOut
        }
    }

    private func generationStatus(_ generationID: String) async throws -> GenerationStatus {
        var request = URLRequest(url: endpointURL(path: "/generate/\(generationID)/status"))
        request.httpMethod = "GET"
        request.timeoutInterval = 600
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw VoiceboxAPIError.invalidResponse
        }
        guard 200..<300 ~= http.statusCode else {
            throw VoiceboxAPIError.http(http.statusCode, "")
        }

        for try await line in bytes.lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("data:") else { continue }
            let payload = String(trimmed.dropFirst(5)).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let data = payload.data(using: .utf8),
                  let status = try? decoder.decode(GenerationStatus.self, from: data)
            else {
                throw VoiceboxAPIError.invalidResponse
            }
            if ["completed", "failed", "cancelled", "canceled", "error", "not_found"].contains(status.status.lowercased()) {
                return status
            }
        }
        throw VoiceboxAPIError.invalidResponse
    }

    private func jsonRequest<Response: Decodable, Body: Encodable>(_ method: String, path: String, jsonBody: Body? = Optional<Data>.none) async throws -> Response {
        var request = URLRequest(url: endpointURL(path: path))
        request.httpMethod = method
        if let jsonBody {
            request.httpBody = try encoder.encode(jsonBody)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let data = try await perform(request)
        if Response.self == EmptyResponse.self {
            let text = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            guard data.isEmpty || text.isEmpty || text == "{}" || text == "null" else {
                throw VoiceboxAPIError.invalidResponse
            }
            return EmptyResponse() as! Response
        }
        return try decoder.decode(Response.self, from: data)
    }

    private func multipartRequest<Response: Decodable>(_ method: String, path: String, multipart: MultipartBody) async throws -> Response {
        var request = URLRequest(url: endpointURL(path: path))
        request.httpMethod = method
        request.httpBody = multipart.data
        request.setValue("multipart/form-data; boundary=\(multipart.boundary)", forHTTPHeaderField: "Content-Type")
        let data = try await perform(request)
        return try decoder.decode(Response.self, from: data)
    }

    private func rawRequest(_ method: String, path: String) async throws -> Data {
        var request = URLRequest(url: endpointURL(path: path))
        request.httpMethod = method
        return try await perform(request)
    }

    private func endpointURL(path: String) -> URL {
        baseURL.appendingPathComponent(path.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
    }

    private func perform(_ request: URLRequest) async throws -> Data {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw VoiceboxAPIError.invalidResponse
        }
        guard 200..<300 ~= http.statusCode else {
            throw VoiceboxAPIError.http(http.statusCode, String(decoding: data, as: UTF8.self))
        }
        return data
    }
}

private struct EmptyResponse: Decodable {}

private struct MultipartFile {
    let fieldName: String
    let url: URL
}

private struct MultipartBody {
    let boundary: String
    let data: Data
}

private func multipartBody(fields: [String: String], files: [MultipartFile]) async throws -> MultipartBody {
    let boundary = "----ChatParserMac-\(UUID().uuidString)"
    var data = Data()

    for (name, value) in fields {
        data.append("--\(boundary)\r\n")
        data.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        data.append("\(value)\r\n")
    }

    for file in files {
        let filename = file.url.lastPathComponent
        let contentType = contentType(for: file.url)
        let fileData = try await Task.detached(priority: .userInitiated) {
            try Data(contentsOf: file.url)
        }.value
        data.append("--\(boundary)\r\n")
        data.append("Content-Disposition: form-data; name=\"\(file.fieldName)\"; filename=\"\(filename)\"\r\n")
        data.append("Content-Type: \(contentType)\r\n\r\n")
        data.append(fileData)
        data.append("\r\n")
    }

    data.append("--\(boundary)--\r\n")
    return MultipartBody(boundary: boundary, data: data)
}

private func contentType(for url: URL) -> String {
    switch url.pathExtension.lowercased() {
    case "wav": "audio/wav"
    case "mp3": "audio/mpeg"
    case "m4a": "audio/mp4"
    case "ogg", "opus": "audio/ogg"
    case "flac": "audio/flac"
    case "aac": "audio/aac"
    case "webm": "audio/webm"
    default: "application/octet-stream"
    }
}

private extension Data {
    mutating func append(_ string: String) {
        append(Data(string.utf8))
    }
}
