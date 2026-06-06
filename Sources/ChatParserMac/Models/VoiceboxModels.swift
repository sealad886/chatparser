import Foundation

struct VoiceProfile: Codable, Identifiable, Equatable {
    let id: String
    var name: String
    var description: String?
    var language: String
    var voiceType: String
    var defaultEngine: String?
    var personality: String?
    var generationCount: Int?
    var sampleCount: Int?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case language
        case voiceType = "voice_type"
        case defaultEngine = "default_engine"
        case personality
        case generationCount = "generation_count"
        case sampleCount = "sample_count"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        description = try container.decodeIfPresent(String.self, forKey: .description)
        language = try container.decode(String.self, forKey: .language)
        voiceType = try container.decodeIfPresent(String.self, forKey: .voiceType) ?? "cloned"
        defaultEngine = try container.decodeIfPresent(String.self, forKey: .defaultEngine)
        personality = try container.decodeIfPresent(String.self, forKey: .personality)
        generationCount = try container.decodeIfPresent(Int.self, forKey: .generationCount)
        sampleCount = try container.decodeIfPresent(Int.self, forKey: .sampleCount)
    }
}

struct ProfileSample: Codable, Identifiable, Equatable {
    let id: String
    let profileID: String
    let audioPath: String
    var referenceText: String

    enum CodingKeys: String, CodingKey {
        case id
        case profileID = "profile_id"
        case audioPath = "audio_path"
        case referenceText = "reference_text"
    }
}

struct GenerationResponse: Codable, Equatable {
    let id: String
    let profileID: String
    let text: String
    let language: String
    let status: String?
    let audioPath: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case id
        case profileID = "profile_id"
        case text
        case language
        case status
        case audioPath = "audio_path"
        case error
    }
}

struct GenerationStatus: Codable, Equatable {
    let id: String?
    let status: String
    let audioPath: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case id
        case status
        case audioPath = "audio_path"
        case error
    }
}
