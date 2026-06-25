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


class StreamingFakeResponse(FakeResponse):
    def __init__(self, payload: bytes):
        super().__init__(200, b"", {"Content-Type": "text/event-stream"})
        self._lines = iter(payload.splitlines(keepends=True))

    def readline(self) -> bytes:
        return next(self._lines, b"")

    def read(self) -> bytes:
        raise AssertionError("streaming status should be read incrementally")


def test_transcribe_audio_posts_multipart_file_and_model(tmp_path):
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

    result = client.transcribe_audio(audio_file, model="turbo")

    assert result.text == "hello from voicebox"
    assert result.language == "en"
    assert captured["url"] == "http://127.0.0.1:17493/transcribe"
    assert captured["timeout"] == 120
    assert "multipart/form-data" in captured["headers"]["Content-type"]
    assert b'name="model"' in captured["body"]
    assert b"turbo" in captured["body"]
    assert b'name="file"; filename="clip.ogg"' in captured["body"]
    assert b"audio-bytes" in captured["body"]


def test_transcribe_audio_posts_optional_language_hint(tmp_path):
    audio_file = tmp_path / "clip.ogg"
    audio_file.write_bytes(b"audio-bytes")
    captured = {}

    def opener(request, timeout):
        captured["body"] = request.data
        return FakeResponse(200, {"text": "bonjour", "language": "fr"})

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    result = client.transcribe_audio(audio_file, model="turbo", language="fr")

    assert result.text == "bonjour"
    assert b'name="language"' in captured["body"]
    assert b"fr" in captured["body"]


def test_transcribe_audio_retries_accepted_model_download(tmp_path, monkeypatch):
    audio_file = tmp_path / "clip.ogg"
    audio_file.write_bytes(b"audio-bytes")
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse(202, {"detail": "Model download started"})
        return FakeResponse(200, {"text": "download finished", "language": "en"})

    monkeypatch.setattr("voicebox_client.time.sleep", lambda _seconds: None)
    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    result = client.transcribe_audio(audio_file, model="turbo", poll_interval=0.01)

    assert calls == 2
    assert result.text == "download finished"


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


def test_generate_speech_fetches_audio_from_generation_endpoint(tmp_path):
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
                    "status": "completed",
                    "audio_path": "/audio/gen-123",
                    "created_at": "2026-06-05T00:00:00Z",
                },
            )
        if request.full_url.endswith("/audio/gen-123"):
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
    assert captured[1]["url"] == "http://127.0.0.1:17493/audio/gen-123"


def test_generate_speech_waits_for_queued_generation_before_audio_fetch(tmp_path, monkeypatch):
    output_file = tmp_path / "speech.wav"
    captured = []
    status_calls = 0

    def opener(request, timeout):
        nonlocal status_calls
        captured.append(request.full_url)
        if request.full_url.endswith("/generate"):
            return FakeResponse(200, {"id": "gen-123", "status": "generating"})
        if request.full_url.endswith("/generate/gen-123/status"):
            status_calls += 1
            status = "generating" if status_calls == 1 else "completed"
            return StreamingFakeResponse(f'data: {{"id":"gen-123","status":"{status}"}}\n\n'.encode())
        if request.full_url.endswith("/audio/gen-123"):
            return FakeResponse(200, b"wav-bytes", {"Content-Type": "audio/wav"})
        raise AssertionError(request.full_url)

    monkeypatch.setattr("voicebox_client.time.sleep", lambda _seconds: None)
    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    written = client.generate_speech(
        text="Local WhatsApp message",
        output_path=output_file,
        profile_id="voice-123",
        poll_interval=0.01,
    )

    assert written == output_file
    assert status_calls == 2
    assert captured == [
        "http://127.0.0.1:17493/generate",
        "http://127.0.0.1:17493/generate/gen-123/status",
        "http://127.0.0.1:17493/generate/gen-123/status",
        "http://127.0.0.1:17493/audio/gen-123",
    ]
    assert output_file.read_bytes() == b"wav-bytes"


def test_generate_speech_consumes_streaming_status_until_terminal_event(tmp_path, monkeypatch):
    output_file = tmp_path / "speech.wav"
    captured = []

    def opener(request, timeout):
        captured.append(request.full_url)
        if request.full_url.endswith("/generate"):
            return FakeResponse(200, {"id": "gen-123", "status": "generating"})
        if request.full_url.endswith("/generate/gen-123/status"):
            return StreamingFakeResponse(
                b'data: {"id":"gen-123","status":"generating"}\n\n'
                b'data: {"id":"gen-123","status":"completed"}\n\n'
            )
        if request.full_url.endswith("/audio/gen-123"):
            return FakeResponse(200, b"wav-bytes", {"Content-Type": "audio/wav"})
        raise AssertionError(request.full_url)

    monkeypatch.setattr("voicebox_client.time.sleep", lambda _seconds: None)
    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    written = client.generate_speech(
        text="Local WhatsApp message",
        output_path=output_file,
        profile_id="voice-123",
        poll_interval=0.01,
    )

    assert written == output_file
    assert captured == [
        "http://127.0.0.1:17493/generate",
        "http://127.0.0.1:17493/generate/gen-123/status",
        "http://127.0.0.1:17493/audio/gen-123",
    ]
    assert output_file.read_bytes() == b"wav-bytes"


def test_generate_speech_requires_profile_id(tmp_path):
    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=lambda request, timeout: None)

    with pytest.raises(VoiceboxError, match="profile_id"):
        client.generate_speech("hello", tmp_path / "out.wav")


def test_generate_speech_rejects_empty_audio_response(tmp_path):
    output_file = tmp_path / "speech.wav"

    def opener(request, timeout):
        if request.full_url.endswith("/generate"):
            return FakeResponse(200, {"id": "gen-123", "status": "completed"})
        if request.full_url.endswith("/audio/gen-123"):
            return FakeResponse(200, b"", {"Content-Type": "audio/wav"})
        raise AssertionError(request.full_url)

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    with pytest.raises(VoiceboxError, match="empty audio response"):
        client.generate_speech("hello", output_file, profile_id="voice-123")
    assert not output_file.exists()


def test_generate_speech_rejects_non_audio_response(tmp_path):
    output_file = tmp_path / "speech.wav"

    def opener(request, timeout):
        if request.full_url.endswith("/generate"):
            return FakeResponse(200, {"id": "gen-123", "status": "completed"})
        if request.full_url.endswith("/audio/gen-123"):
            return FakeResponse(200, {"detail": "not ready"}, {"Content-Type": "application/json"})
        raise AssertionError(request.full_url)

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    with pytest.raises(VoiceboxError, match="non-audio content"):
        client.generate_speech("hello", output_file, profile_id="voice-123")
    assert not output_file.exists()


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


def test_list_profiles_accepts_profiles_wrapper():
    def opener(request, timeout):
        return FakeResponse(200, {"profiles": [{"id": "voice-123", "name": "Alice"}]})

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    assert client.list_profiles() == [{"id": "voice-123", "name": "Alice"}]


@pytest.mark.parametrize("payload", [{"profiles": "bad"}, {}, {"items": []}])
def test_list_profiles_rejects_unexpected_shapes(payload):
    def opener(request, timeout):
        return FakeResponse(200, payload)

    client = VoiceboxClient(base_url="http://127.0.0.1:17493", opener=opener)

    with pytest.raises(VoiceboxError, match="unexpected response"):
        client.list_profiles()
