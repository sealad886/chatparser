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
            if isinstance(profiles, list):
                return profiles
        raise VoiceboxError("Voicebox /profiles returned an unexpected response")

    def create_voice_profile(
        self,
        name: str,
        language: str = "en",
        description: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "description": description,
            "language": language,
            "voice_type": "cloned",
        }
        response = self._json_request("POST", "/profiles", payload=payload)
        if not isinstance(response, dict):
            raise VoiceboxError("Voicebox /profiles returned an unexpected response")
        return response

    def add_profile_sample(
        self,
        profile_id: str,
        audio_path: str | Path,
        reference_text: str,
    ) -> dict[str, Any]:
        path = Path(audio_path)
        headers, body = self._multipart_body({"reference_text": reference_text}, {"file": path})
        response = self._request("POST", f"/profiles/{profile_id}/samples", data=body, headers=headers)
        payload = self._decode_json(response)
        if not isinstance(payload, dict):
            raise VoiceboxError("Voicebox profile sample upload returned an unexpected response")
        return payload

    def transcribe_audio(
        self,
        audio_path: str | Path,
        model: str = "turbo",
        language: str | None = None,
        poll_interval: float = 2.0,
        max_download_wait_seconds: int = 600,
    ) -> TranscriptionResult:
        path = Path(audio_path)
        fields = {"model": model}
        if language:
            fields["language"] = language
        files = {"file": path}
        headers, body = self._multipart_body(fields, files)
        deadline = time.monotonic() + max_download_wait_seconds
        while True:
            response = self._request("POST", "/transcribe", data=body, headers=headers)
            status = self._status_code(response)
            payload = self._decode_json(response)
            if status != 202:
                break
            if time.monotonic() >= deadline:
                raise VoiceboxError(f"Voicebox model download did not complete before retry timeout: {payload}")
            time.sleep(poll_interval)
        if not isinstance(payload, dict):
            raise VoiceboxError("Voicebox /transcribe returned an unexpected response")
        text = str(payload.get("text") or payload.get("transcript") or "")
        if not text:
            raise VoiceboxError(f"Voicebox /transcribe returned no transcript text: {payload}")
        language = payload.get("language")
        return TranscriptionResult(text=text, language=str(language) if language else None, raw=payload)

    def generate_speech(
        self,
        text: str,
        output_path: str | Path,
        profile_id: str | None = None,
        language: str = "en",
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

        status = str(generation.get("status") or "").lower()
        if status != "completed":
            self._wait_for_generation(str(generation_id), poll_interval, max_wait_seconds)

        audio_response = self._request("GET", f"/audio/{generation_id}")
        body = self._read_audio_response(audio_response)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        return destination

    def _wait_for_generation(self, generation_id: str, poll_interval: float, max_wait_seconds: int) -> None:
        deadline = time.monotonic() + max_wait_seconds
        while time.monotonic() < deadline:
            response = self._request("GET", f"/generate/{generation_id}/status")
            status_payload = self._decode_generation_status(response)
            status = str(status_payload.get("status", "")).lower()
            if status == "completed":
                return
            if status in {"failed", "cancelled", "canceled", "error", "not_found"}:
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
        with response:
            try:
                body = response.read().decode("utf-8")
                decoded = json.loads(body) if body else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VoiceboxError("Voicebox returned a non-JSON response") from exc
        if not isinstance(decoded, (dict, list)):
            raise VoiceboxError("Voicebox returned an unexpected JSON shape")
        return decoded

    def _read_audio_response(self, response: Any) -> bytes:
        content_type = self._content_type(response)
        with response:
            body = response.read()
        if not body:
            raise VoiceboxError("Voicebox returned an empty audio response")
        if content_type and not content_type.startswith("audio/"):
            raise VoiceboxError(f"Voicebox returned non-audio content from /audio: {content_type}")
        return body

    def _status_code(self, response: Any) -> int | None:
        status = getattr(response, "status", None)
        if status is not None:
            return int(status)
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            code = getcode()
            return int(code) if code is not None else None
        return None

    def _decode_generation_status(self, response: Any) -> dict[str, Any]:
        if "text/event-stream" in self._content_type(response):
            return self._decode_generation_status_event(response)

        with response:
            try:
                body = response.read().decode("utf-8")
            except UnicodeDecodeError as exc:
                raise VoiceboxError("Voicebox returned a non-text generation status response") from exc
        if not body:
            return {}
        try:
            decoded = json.loads(body)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass

        latest: dict[str, Any] | None = None
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise VoiceboxError("Voicebox returned malformed generation status SSE") from exc
            if isinstance(decoded, dict):
                latest = decoded
        if latest is None:
            raise VoiceboxError("Voicebox returned an unexpected generation status response")
        return latest

    def _decode_generation_status_event(self, response: Any) -> dict[str, Any]:
        latest: dict[str, Any] | None = None
        terminal_statuses = {"completed", "failed", "cancelled", "canceled", "error", "not_found"}
        with response:
            while True:
                line = response.readline()
                if not line:
                    break
                text = line.decode("utf-8").strip()
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if not payload:
                    continue
                try:
                    decoded = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise VoiceboxError("Voicebox returned malformed generation status SSE") from exc
                if isinstance(decoded, dict):
                    latest = decoded
                    if str(decoded.get("status", "")).lower() in terminal_statuses:
                        return decoded
        if latest is not None:
            return latest
        raise VoiceboxError("Voicebox returned an empty generation status SSE")

    def _content_type(self, response: Any) -> str:
        headers = getattr(response, "headers", {}) or {}
        if hasattr(headers, "get"):
            return str(headers.get("Content-Type") or headers.get("content-type") or "").lower()
        return ""

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
