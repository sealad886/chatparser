import io
import json
import urllib.error

import pytest

from voicebox_client import VoiceboxClient, VoiceboxError


class FakeResponse:
    def __init__(self, status: int, payload: dict | bytes, headers: dict | None = None):
        self.status = status
        self._payload = payload
        self.headers = headers or {}

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_transcribe_audio_posts_multipart_audio_and_model(tmp_path):
    audio_file = tmp_path / "clip.ogg"
    audio_file.write_bytes(b"audio-bytes")
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse(200, {"text": "hello from voicebox", "language": "en"})

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    result = client.transcribe_audio(audio_file, model="whisper-turbo")

    assert result.text == "hello from voicebox"
    assert result.language == "en"
    assert captured["url"] == "http://127.0.0.1:17493/transcribe"
    assert captured["timeout"] == 120
    assert "multipart/form-data" in captured["headers"]["Content-type"]
    assert b'name="model"' in captured["body"]
    assert b"whisper-turbo" in captured["body"]
    assert b'name="audio"; filename="clip.ogg"' in captured["body"]
    assert b"audio-bytes" in captured["body"]


def test_create_voice_profile_posts_cloned_profile_payload():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(200, {"id": "profile-123", "name": "Alice", "language": "en"})

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    profile = client.create_voice_profile("Alice", language="en", description="WhatsApp samples")

    assert profile["id"] == "profile-123"
    assert captured["url"] == "http://127.0.0.1:17493/profiles"
    assert captured["payload"] == {
        "name": "Alice",
        "description": "WhatsApp samples",
        "language": "en",
        "voice_type": "cloned",
    }


def test_add_profile_sample_posts_local_audio_and_reference_text(tmp_path):
    sample_file = tmp_path / "alice.m4a"
    sample_file.write_bytes(b"sample-audio")
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["headers"] = dict(request.header_items())
        return FakeResponse(
            200,
            {
                "id": "sample-123",
                "profile_id": "profile-123",
                "audio_path": "profiles/profile-123/sample.wav",
                "reference_text": "hello there",
            },
        )

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    sample = client.add_profile_sample("profile-123", sample_file, "hello there")

    assert sample["id"] == "sample-123"
    assert captured["url"] == "http://127.0.0.1:17493/profiles/profile-123/samples"
    assert "multipart/form-data" in captured["headers"]["Content-type"]
    assert b'name="reference_text"' in captured["body"]
    assert b"hello there" in captured["body"]
    assert b'name="file"; filename="alice.m4a"' in captured["body"]
    assert b"sample-audio" in captured["body"]


def test_generate_speech_polls_and_exports_audio_to_output_path(tmp_path):
    output_file = tmp_path / "speech.mp3"
    captured = []

    def opener(request, timeout):
        captured.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "payload": json.loads(request.data.decode("utf-8")) if request.data else None,
            }
        )
        if request.full_url.endswith("/generate"):
            return FakeResponse(
                200,
                {
                    "id": "gen-123",
                    "profile_id": "voice-123",
                    "text": "Local WhatsApp message",
                    "language": "en",
                    "status": "generating",
                    "created_at": "2026-06-05T00:00:00Z",
                },
            )
        if request.full_url.endswith("/generate/gen-123/status"):
            return FakeResponse(200, {"status": "completed"})
        if request.full_url.endswith("/history/gen-123/export-audio"):
            return FakeResponse(200, b"wav-bytes", {"Content-Type": "audio/wav"})
        raise AssertionError(request.full_url)

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    written = client.generate_speech(
        text="Local WhatsApp message",
        output_path=output_file,
        profile_id="voice-123",
        language="en",
    )

    assert written == output_file
    assert output_file.read_bytes() == b"wav-bytes"
    assert captured[0]["url"] == "http://127.0.0.1:17493/generate"
    assert captured[0]["payload"] == {
        "text": "Local WhatsApp message",
        "language": "en",
        "profile_id": "voice-123",
    }
    assert captured[1]["url"] == "http://127.0.0.1:17493/generate/gen-123/status"
    assert captured[2]["url"] == "http://127.0.0.1:17493/history/gen-123/export-audio"


def test_generate_speech_requires_profile_id(tmp_path):
    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=lambda request, timeout: None)

    with pytest.raises(VoiceboxError, match="profile_id"):
        client.generate_speech("hello", tmp_path / "out.wav")


def test_voicebox_errors_include_endpoint_and_status():
    def opener(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b'{"detail":"model unavailable"}'),
        )

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    with pytest.raises(VoiceboxError) as err:
        client.list_profiles()

    assert "/profiles" in str(err.value)
    assert "503" in str(err.value)
    assert "model unavailable" in str(err.value)


def test_list_profiles_accepts_top_level_list():
    def opener(request, timeout):
        return FakeResponse(200, [{"id": "voice-123", "name": "Alice"}])

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    assert client.list_profiles() == [{"id": "voice-123", "name": "Alice"}]
