import sys
from types import SimpleNamespace

import chatparser


class FakeVoiceboxClient:
    def __init__(self):
        self.transcribed = []
        self.generated = []

    def transcribe_audio(self, audio_path, model):
        self.transcribed.append((str(audio_path), model))
        return SimpleNamespace(text="voicebox transcript", language="en")

    def generate_speech(self, text, output_path, profile_id=None, language="en"):
        self.generated.append(
            {
                "text": text,
                "output_path": str(output_path),
                "profile_id": profile_id,
                "language": language,
            }
        )
        output_path.write_bytes(b"generated audio")
        return output_path


class FakeAudioSegment:
    @classmethod
    def empty(cls):
        return cls()

    @classmethod
    def from_file(cls, audio_file):
        assert audio_file
        return cls()

    def __iadd__(self, other):
        return self

    def export(self, handle, format):
        assert format == "mp3"
        handle.write(b"mp3-bytes")


def test_transcribe_audio_line_appends_voicebox_transcript(tmp_path, monkeypatch):
    audio = tmp_path / "00000001-AUDIO-2024-01-01-00-00-00.ogg"
    audio.write_bytes(b"audio")
    match = SimpleNamespace(group=lambda _idx: f"<attached: {audio.name}>")
    file_out = []
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_MODEL", "whisper-turbo")
    monkeypatch.setattr(chatparser, "__VERBOSE", False)

    transcription = chatparser.transcribe_audio_line(
        str(tmp_path),
        match,
        "unused-model-dir",
        "unused prompt",
        "01/01/2024, 12:00:00",
        "Alice",
        file_out,
        7,
    )

    assert fake_client.transcribed == [(str(audio), "turbo")]
    assert transcription == file_out[0]
    assert "[01/01/2024, 12:00:00] Alice: [Transcribed]: voicebox transcript" in transcription
    assert "(en)" in transcription
    assert f"[File: {audio.name}]" in transcription


def test_format_parsed_whatsapp_line_preserves_plain_message():
    parsed = chatparser.parse_whatsapp_line("[01/01/2024, 12:00:00] Alice: hello")

    assert parsed is not None
    assert chatparser.format_parsed_whatsapp_line(parsed) == "[01/01/2024, 12:00:00] Alice: hello\n"


def test_normalize_voicebox_transcription_model_accepts_legacy_whisper_prefix():
    assert chatparser.normalize_voicebox_transcription_model("whisper-turbo") == "turbo"
    assert chatparser.normalize_voicebox_transcription_model("small") == "small"


def test_line_to_audio_generates_speech_through_voicebox(tmp_path, monkeypatch):
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE", "voice-123")
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE_MAP", {})
    monkeypatch.setattr(chatparser, "__VOICEBOX_LANGUAGE", "en")
    monkeypatch.setattr(chatparser, "__VERBOSE", False)

    output_path = chatparser.line_to_audio(
        "Meet at the station",
        "Alice",
        "01/02/2024, 18:30:00",
        str(tmp_path),
        4,
        {},
    )

    assert output_path.endswith("00000005-AUDIO-2024-02-01-18-30-00.wav")
    assert (tmp_path / "audio_out" / "00000005-AUDIO-2024-02-01-18-30-00.wav").read_bytes() == b"generated audio"
    assert fake_client.generated == [
        {
            "text": "Meet at the station",
            "output_path": output_path,
            "profile_id": "voice-123",
            "language": "en",
        }
    ]


def test_line_to_audio_uses_speaker_profile_mapping(tmp_path, monkeypatch):
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE", "fallback-profile")
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE_MAP", {"Alice": "alice-profile"})
    monkeypatch.setattr(chatparser, "__VOICEBOX_LANGUAGE", "en")
    monkeypatch.setattr(chatparser, "__VERBOSE", False)

    chatparser.line_to_audio(
        "Voice mapped text",
        "Alice",
        "01/02/2024, 18:30:00",
        str(tmp_path),
        6,
        {},
    )

    assert fake_client.generated[0]["profile_id"] == "alice-profile"


def test_line_to_audio_uses_android_export_naming(tmp_path, monkeypatch):
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE", "voice-123")
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE_MAP", {})
    monkeypatch.setattr(chatparser, "__VOICEBOX_LANGUAGE", "en")
    monkeypatch.setattr(chatparser, "__VERBOSE", False)

    output_path = chatparser.line_to_audio(
        "Meet at the station",
        "Alice",
        "01/02/2024, 18:30:00",
        str(tmp_path),
        4,
        {},
        source_format="android",
    )

    assert output_path.endswith("PTT-20240201-WA0004.wav")
    assert (tmp_path / "audio_out" / "PTT-20240201-WA0004.wav").read_bytes() == b"generated audio"


def test_parse_voicebox_profile_map_merges_json_and_cli_entries(tmp_path):
    mapping_file = tmp_path / "profiles.json"
    mapping_file.write_text('{"Alice": "alice-profile"}')

    result = chatparser.parse_voicebox_profile_map(
        ["Bob=bob-profile"],
        str(mapping_file),
    )

    assert result == {"Alice": "alice-profile", "Bob": "bob-profile"}


def test_audio_export_uses_first_speaker_profile_mapping(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    chat_file = export_dir / "_chat.txt"
    chat_file.write_text(
        "[01/02/2024, 18:30:00] Alice: Meet at the station\n"
        "[01/02/2024, 18:31:00] Alice: Bring the tickets\n",
        encoding="utf-8",
    )
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE", None)
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE_MAP", {"Alice": "alice-profile"})
    monkeypatch.setattr(chatparser, "__VOICEBOX_LANGUAGE", "en")
    monkeypatch.setattr(chatparser, "__FORCE_REDO", True)
    monkeypatch.setattr(chatparser, "__PROGRESS_BAR", False)
    monkeypatch.setattr(chatparser, "__VERBOSE", False)
    monkeypatch.setattr(chatparser, "__ENABLE_TIMINGS", False)
    monkeypatch.setattr(chatparser, "__NUM_WORKERS", 1)
    monkeypatch.setattr(chatparser, "q", chatparser.Queue())
    monkeypatch.setattr(chatparser, "workers", [])

    chatparser.process_chat_file_by_type(
        str(chat_file),
        str(export_dir),
        "",
        [],
        to_type="audio",
    )

    assert fake_client.generated
    assert fake_client.generated[0]["profile_id"] == "alice-profile"


def test_audio_export_preserves_multiline_message_continuation(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    chat_file = export_dir / "_chat.txt"
    chat_file.write_text(
        "[01/02/2024, 18:30:00] Alice: Meet at the station\n"
        "and bring the tickets\n",
        encoding="utf-8",
    )
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE", "alice-profile")
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE_MAP", {})
    monkeypatch.setattr(chatparser, "__VOICEBOX_LANGUAGE", "en")
    monkeypatch.setattr(chatparser, "__FORCE_REDO", True)
    monkeypatch.setattr(chatparser, "__PROGRESS_BAR", False)
    monkeypatch.setattr(chatparser, "__VERBOSE", False)
    monkeypatch.setattr(chatparser, "__ENABLE_TIMINGS", False)
    monkeypatch.setattr(chatparser, "__NUM_WORKERS", 1)
    monkeypatch.setattr(chatparser, "_check_spelling", lambda text: text)
    monkeypatch.setattr(chatparser, "q", chatparser.Queue())
    monkeypatch.setattr(chatparser, "workers", [])

    chatparser.process_chat_file_by_type(
        str(chat_file),
        str(export_dir),
        "",
        [],
        to_type="audio",
    )

    assert fake_client.generated
    assert "Meet at the station. and bring the tickets" in fake_client.generated[0]["text"]


def test_android_audio_export_uses_first_message_sequence_for_grouped_speech(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    chat_file = export_dir / "WhatsApp Chat with Alice.txt"
    chat_file.write_text(
        "30/05/2026, 00:32 - Alice: First Android line\n"
        "continued Android text\n"
        "30/05/2026, 00:33 - Bob: Reply from Bob\n",
        encoding="utf-8",
    )
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE", "default-profile")
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE_MAP", {"Alice": "alice-profile", "Bob": "bob-profile"})
    monkeypatch.setattr(chatparser, "__VOICEBOX_LANGUAGE", "en")
    monkeypatch.setattr(chatparser, "__FORCE_REDO", True)
    monkeypatch.setattr(chatparser, "__PROGRESS_BAR", False)
    monkeypatch.setattr(chatparser, "__VERBOSE", False)
    monkeypatch.setattr(chatparser, "__ENABLE_TIMINGS", False)
    monkeypatch.setattr(chatparser, "__NUM_WORKERS", 1)
    monkeypatch.setattr(chatparser, "_check_spelling", lambda text: text)
    monkeypatch.setattr(chatparser, "q", chatparser.Queue())
    monkeypatch.setattr(chatparser, "workers", [])

    chatparser.process_chat_file_by_type(
        str(chat_file),
        str(export_dir),
        "",
        [],
        to_type="audio",
    )

    assert [entry["output_path"].split("/")[-1] for entry in fake_client.generated] == [
        "PTT-20260530-WA0000.wav",
        "PTT-20260530-WA0001.wav",
    ]
    assert fake_client.generated[0]["text"] == "First Android line. continued Android text"
    assert fake_client.generated[0]["profile_id"] == "alice-profile"


def test_audio_export_preserves_unparseable_preamble(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    chat_file = export_dir / "_chat.txt"
    chat_file.write_text(
        "Messages and calls are end-to-end encrypted.\n"
        "[01/02/2024, 18:30:00] Alice: Meet at the station\n",
        encoding="utf-8",
    )
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE", "alice-profile")
    monkeypatch.setattr(chatparser, "__VOICEBOX_PROFILE_MAP", {})
    monkeypatch.setattr(chatparser, "__VOICEBOX_LANGUAGE", "en")
    monkeypatch.setattr(chatparser, "__FORCE_REDO", True)
    monkeypatch.setattr(chatparser, "__PROGRESS_BAR", False)
    monkeypatch.setattr(chatparser, "__VERBOSE", False)
    monkeypatch.setattr(chatparser, "__ENABLE_TIMINGS", False)
    monkeypatch.setattr(chatparser, "__NUM_WORKERS", 1)
    monkeypatch.setattr(chatparser, "_check_spelling", lambda text: text)
    monkeypatch.setattr(chatparser, "q", chatparser.Queue())
    monkeypatch.setattr(chatparser, "workers", [])
    file_out = []

    chatparser.process_chat_file_by_type(
        str(chat_file),
        str(export_dir),
        "",
        file_out,
        to_type="audio",
    )

    assert file_out == ["Messages and calls are end-to-end encrypted.\n"]
    assert fake_client.generated


def test_text_export_preserves_non_attachment_messages(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    chat_file = export_dir / "_chat.txt"
    chat_file.write_text(
        "[01/02/2024, 18:30:00] Alice: Meet at the station\n"
        "[01/02/2024, 18:31:00] Bob: See you there\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(chatparser, "__FORCE_REDO", True)
    monkeypatch.setattr(chatparser, "__PROGRESS_BAR", False)
    monkeypatch.setattr(chatparser, "__VERBOSE", False)
    monkeypatch.setattr(chatparser, "__ENABLE_TIMINGS", False)
    monkeypatch.setattr(chatparser, "__NUM_WORKERS", 1)
    monkeypatch.setattr(chatparser, "q", chatparser.Queue())
    monkeypatch.setattr(chatparser, "workers", [])
    file_out = []

    chatparser.process_chat_file_by_type(
        str(chat_file),
        str(export_dir),
        "",
        file_out,
        to_type="text",
    )

    assert file_out == [
        "[01/02/2024, 18:30:00] Alice: Meet at the station\n",
        "[01/02/2024, 18:31:00] Bob: See you there\n",
    ]


def test_audio_export_converts_existing_attachment_with_visible_output(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    chat_file = export_dir / "_chat.txt"
    chat_file.write_text(
        "[01/02/2024, 18:30:00] Alice: <attached: clip.ogg>\n",
        encoding="utf-8",
    )
    (export_dir / "clip.ogg").write_bytes(b"ogg-bytes")
    monkeypatch.setitem(sys.modules, "pydub", SimpleNamespace(AudioSegment=FakeAudioSegment))
    monkeypatch.setattr(chatparser, "__FORCE_REDO", True)
    monkeypatch.setattr(chatparser, "__PROGRESS_BAR", False)
    monkeypatch.setattr(chatparser, "__VERBOSE", False)
    monkeypatch.setattr(chatparser, "__ENABLE_TIMINGS", False)
    monkeypatch.setattr(chatparser, "__NUM_WORKERS", 1)
    monkeypatch.setattr(chatparser, "q", chatparser.Queue())
    monkeypatch.setattr(chatparser, "workers", [])

    chatparser.process_chat_file_by_type(
        str(chat_file),
        str(export_dir),
        "",
        [],
        to_type="audio",
    )

    assert (export_dir / "audio_out" / "clip.mp3").read_bytes() == b"mp3-bytes"


def test_audio_export_missing_attachment_fails_visibly(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    chat_file = export_dir / "_chat.txt"
    chat_file.write_text(
        "[01/02/2024, 18:30:00] Alice: <attached: missing.ogg>\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "pydub", SimpleNamespace(AudioSegment=FakeAudioSegment))
    monkeypatch.setattr(chatparser, "__FORCE_REDO", True)
    monkeypatch.setattr(chatparser, "__PROGRESS_BAR", False)
    monkeypatch.setattr(chatparser, "__VERBOSE", False)
    monkeypatch.setattr(chatparser, "__ENABLE_TIMINGS", False)
    monkeypatch.setattr(chatparser, "__NUM_WORKERS", 1)
    monkeypatch.setattr(chatparser, "q", chatparser.Queue())
    monkeypatch.setattr(chatparser, "workers", [])

    try:
        chatparser.process_chat_file_by_type(
            str(chat_file),
            str(export_dir),
            "",
            [],
            to_type="audio",
        )
    except FileNotFoundError as exc:
        assert "missing.ogg" in str(exc)
    else:
        raise AssertionError("missing attachment should fail the parent process")
