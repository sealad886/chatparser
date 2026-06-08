import os
from pathlib import Path

import pytest

from voicebox_client import VoiceboxClient, VoiceboxError


pytestmark = pytest.mark.voicebox_integration


def _client() -> VoiceboxClient:
    base_url = os.environ.get("VOICEBOX_BASE_URL")
    if not base_url:
        pytest.skip("VOICEBOX_BASE_URL is not set")
    return VoiceboxClient(base_url=base_url, timeout=600)


def _require_reachable_voicebox(client: VoiceboxClient) -> None:
    try:
        client.health()
    except VoiceboxError as exc:
        pytest.skip(f"Voicebox is not reachable: {exc}")


def test_voicebox_transcribe_contract_against_running_service():
    sample = os.environ.get("VOICEBOX_SAMPLE_AUDIO")
    if not sample:
        pytest.skip("VOICEBOX_SAMPLE_AUDIO is not set")
    sample_path = Path(sample)
    if not sample_path.is_file():
        pytest.skip(f"VOICEBOX_SAMPLE_AUDIO is not a file: {sample_path}")
    client = _client()
    _require_reachable_voicebox(client)

    result = client.transcribe_audio(
        sample_path,
        model=os.environ.get("VOICEBOX_TRANSCRIPTION_MODEL", "turbo"),
    )

    assert result.text.strip()


def test_voicebox_generation_contract_against_running_service(tmp_path):
    profile_id = os.environ.get("VOICEBOX_PROFILE_ID")
    if not profile_id:
        pytest.skip("VOICEBOX_PROFILE_ID is not set")
    client = _client()
    _require_reachable_voicebox(client)
    output_path = tmp_path / "voicebox-contract.wav"

    written = client.generate_speech(
        "Voicebox contract smoke test.",
        output_path,
        profile_id=profile_id,
        language=os.environ.get("VOICEBOX_LANGUAGE", "en"),
        poll_interval=0.2,
        max_wait_seconds=600,
    )

    assert written == output_path
    assert output_path.read_bytes()
