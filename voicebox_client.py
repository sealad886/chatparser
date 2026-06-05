from __future__ import annotations

import json
import mimetypes
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class VoiceboxError(RuntimeError):
    """Raised when the local Voicebox API rejects or cannot process a request."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    raw: dict[str, Any] | None = None


class VoiceboxClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:17493",
        timeout: int = 120,
        opener: Callable[[urllib.request.Request, int], Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def health(self) -> dict[str, Any]:
        return self._json_request("GET", "/health")

    def list_profiles(self) -> list[dict[str, Any]]:
        response = self._json_request("GET", "/profiles")
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            profiles = response.get("profiles")
            return profiles if isinstance(profiles, list) else []
        return []

    def transcribe_audio(self, audio_path: str | Path, model: str = "whisper-turbo") -> TranscriptionResult:
        path = Path(audio_path)
        fields = {"model": model}
        files = {"audio": path}
        headers, body = self._multipart_body(fields, files)
        response = self._request("POST", "/transcribe", data=body, headers=headers)
        payload = self._decode_json(response)
        text = str(payload.get("text") or payload.get("transcript") or "")
        language = payload.get("language")
        return TranscriptionResult(text=text, language=str(language) if language else None, raw=payload)

    def generate_speech(
        self,
        text: str,
        output_path: str | Path,
        profile_id: str | None = None,
        language: str = "en",
        profile: str | None = None,
        poll_interval: float = 1.0,
        max_wait_seconds: int = 600,
    ) -> Path:
        if not profile_id:
            raise VoiceboxError("/generate requires profile_id for Voicebox voice cloning")

        payload: dict[str, Any] = {"text": text, "language": language}
        payload["profile_id"] = profile_id

        response = self._request(
            "POST",
            "/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        generation = self._decode_json(response)
        generation_id = generation.get("id")
        if not generation_id:
            raise VoiceboxError("Voicebox /generate response did not include an id")

        self._wait_for_generation(str(generation_id), poll_interval, max_wait_seconds)
        export_response = self._request("GET", f"/history/{generation_id}/export-audio")
        body = export_response.read()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        return destination

    def _wait_for_generation(self, generation_id: str, poll_interval: float, max_wait_seconds: int) -> None:
        deadline = time.monotonic() + max_wait_seconds
        while time.monotonic() < deadline:
            status_payload = self._json_request("GET", f"/generate/{generation_id}/status")
            status = str(status_payload.get("status", "")).lower()
            if status == "completed":
                return
            if status in {"failed", "cancelled", "canceled", "error"}:
                detail = status_payload.get("error") or status_payload
                raise VoiceboxError(f"Voicebox generation {generation_id} failed: {detail}")
            time.sleep(poll_interval)
        raise VoiceboxError(f"Voicebox generation {generation_id} timed out after {max_wait_seconds}s")

    def _json_request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        response = self._request(method, endpoint, data=data, headers=headers)
        return self._decode_json(response)

    def _request(
        self,
        method: str,
        endpoint: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = urllib.parse.urljoin(f"{self.base_url}/", endpoint.lstrip("/"))
        request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            return self._opener(request, self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VoiceboxError(f"{endpoint} failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise VoiceboxError(f"{endpoint} failed: {exc.reason}") from exc

    def _decode_json(self, response: Any) -> dict[str, Any] | list[Any]:
        try:
            body = response.read().decode("utf-8")
            decoded = json.loads(body) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VoiceboxError("Voicebox returned a non-JSON response") from exc
        if not isinstance(decoded, (dict, list)):
            raise VoiceboxError("Voicebox returned an unexpected JSON shape")
        return decoded

    def _multipart_body(self, fields: dict[str, str], files: dict[str, Path]) -> tuple[dict[str, str], bytes]:
        boundary = f"----chatparser-{uuid.uuid4().hex}"
        chunks: list[bytes] = []

        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("utf-8"),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )

        for name, path in files.items():
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("utf-8"),
                    (
                        f'Content-Disposition: form-data; name="{name}"; '
                        f'filename="{path.name}"\r\n'
                    ).encode("utf-8"),
                    f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                    path.read_bytes(),
                    b"\r\n",
                ]
            )

        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        return {"Content-Type": f"multipart/form-data; boundary={boundary}"}, b"".join(chunks)
