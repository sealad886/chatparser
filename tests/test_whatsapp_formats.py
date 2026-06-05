from datetime import datetime

import chatparser
from test_chatparser_voicebox import FakeVoiceboxClient


def test_parse_ios_whatsapp_line():
    parsed = chatparser.parse_whatsapp_line("[31/12/2024, 23:05:07] Alice: Happy new year")

    assert parsed is not None
    assert parsed.timestamp == datetime(2024, 12, 31, 23, 5, 7)
    assert parsed.date_time_str == "31/12/2024, 23:05:07"
    assert parsed.speaker == "Alice"
    assert parsed.message == "Happy new year"


def test_parse_android_whatsapp_line_with_24_hour_time():
    parsed = chatparser.parse_whatsapp_line("31/12/2024, 23:05 - Alice: Happy new year")

    assert parsed is not None
    assert parsed.timestamp == datetime(2024, 12, 31, 23, 5)
    assert parsed.date_time_str == "31/12/2024, 23:05:00"
    assert parsed.speaker == "Alice"
    assert parsed.message == "Happy new year"


def test_parse_android_whatsapp_line_with_us_12_hour_time():
    parsed = chatparser.parse_whatsapp_line("12/31/24, 8:05 PM - Bob: See attached")

    assert parsed is not None
    assert parsed.timestamp == datetime(2024, 12, 31, 20, 5)
    assert parsed.date_time_str == "31/12/2024, 20:05:00"
    assert parsed.speaker == "Bob"
    assert parsed.message == "See attached"


def test_parse_android_system_line_without_speaker():
    parsed = chatparser.parse_whatsapp_line("31/12/2024, 23:05 - Messages and calls are end-to-end encrypted.")

    assert parsed is not None
    assert parsed.speaker is None
    assert parsed.message == "Messages and calls are end-to-end encrypted."


def test_android_audio_attachment_is_detected_from_file_attached_suffix():
    parsed = chatparser.parse_whatsapp_line("31/12/2024, 23:05 - Alice: AUD-20241231-WA0001.opus (file attached)")

    assert parsed is not None
    attachment = chatparser.find_whatsapp_attachment(parsed.message)

    assert attachment is not None
    assert attachment.filename == "AUD-20241231-WA0001.opus"
    assert attachment.is_audio


def test_ios_audio_attachment_still_detected():
    attachment = chatparser.find_whatsapp_attachment("<attached: 00000001-AUDIO-2024-01-01-00-00-00.ogg>")

    assert attachment is not None
    assert attachment.filename == "00000001-AUDIO-2024-01-01-00-00-00.ogg"
    assert attachment.is_audio


def test_process_android_export_transcribes_audio_attachment(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "_chat.txt").write_text(
        "31/12/2024, 23:05 - Alice: AUD-20241231-WA0001.opus (file attached)\n",
        encoding="utf-8",
    )
    (export_dir / "AUD-20241231-WA0001.opus").write_bytes(b"audio")
    file_out = []
    fake_client = FakeVoiceboxClient()
    monkeypatch.setattr(chatparser, "__VOICEBOX_CLIENT", fake_client)
    monkeypatch.setattr(chatparser, "__VOICEBOX_MODEL", "whisper-turbo")
    monkeypatch.setattr(chatparser, "__FORCE_REDO", True)
    monkeypatch.setattr(chatparser, "__PROGRESS_BAR", False)
    monkeypatch.setattr(chatparser, "__VERBOSE", False)
    monkeypatch.setattr(chatparser, "__ENABLE_TIMINGS", False)
    monkeypatch.setattr(chatparser, "__NUM_WORKERS", 1)
    monkeypatch.setattr(chatparser, "q", chatparser.Queue())
    monkeypatch.setattr(chatparser, "workers", [])

    chatparser.process_chat_file_by_type(
        str(export_dir / "_chat.txt"),
        str(export_dir),
        "",
        file_out,
        to_type="text",
    )

    assert fake_client.transcribed == [(str(export_dir / "AUD-20241231-WA0001.opus"), "whisper-turbo")]
    assert "[31/12/2024, 23:05:00] Alice: [Transcribed]: voicebox transcript" in file_out[0]
