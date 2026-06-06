# Security

## System Overview

ChatParser processes sensitive WhatsApp conversations and media on a user's Mac.
The app imports local exports, stages media into a project workspace, calls
Voicebox over `http://127.0.0.1:17493`, stores transcripts and generated audio
locally, and lets the user export selected results. The MVP has no cloud backend
and no multi-user server.

Voicebox is treated as a separate local process. Upstream Voicebox README states
that its models, voice data, and captures stay local, and that its backend is
FastAPI with SQLite storage. ChatParser relies on the documented REST boundary,
not internal Voicebox storage.

## Assets

| Asset | Sensitivity | Protection |
|---|---|---|
| WhatsApp source export | High | Read-only access; no mutation or deletion. |
| Project SQLite database | High | Local user permissions; optional encrypted volume guidance. |
| Staged media | High | Stored under project root; checksummed; delete on project cleanup. |
| Transcripts | High | Local only; versioned edits; redaction before exports. |
| Generated audio | High | Local only; linked to source text and profile metadata. |
| Voicebox profiles | High | Referenced by id only; managed by Voicebox. |
| Job logs | Medium | No raw message text or transcript by default. |

## Trust Boundaries

| Boundary | Risk | Control |
|---|---|---|
| User-selected source folder to project workspace | Accidental destructive writes | Source paths opened read-only; derived artifacts written only under project root. |
| ChatParser to Voicebox localhost API | Untrusted local service or stale API shape | Loopback-only URL, short timeouts, response validation, explicit service status. |
| Project workspace to export bundle | Oversharing sensitive content | Export preview, manifest, and explicit destination selection. |
| UI to local helper process | Local cross-process request spoofing | Bind to loopback, random session token, no broad CORS. |
| Logs to support bundle | Sensitive text leakage | Structured logs exclude message body and transcript text by default. |

## Entry Points

- File picker import of WhatsApp folders or ZIP files.
- Voicebox REST calls to `/profiles`, `/transcribe`, and `/generate`.
- Local helper API if UI and processor are split.
- Export destination picker.
- Project open/import from existing artifact root.

## STRIDE Analysis

| Threat | Component | Description | Mitigations | Residual Risk |
|---|---|---|---|---|
| Spoofing | Voicebox API | Another process could bind `127.0.0.1:17493` and imitate Voicebox. | Display service identity from `/profiles`/docs where possible, validate response shape, require user to start Voicebox intentionally. | Medium |
| Spoofing | Local helper | Another local process could call helper endpoints. | Random per-session token, loopback bind, reject missing token, no wildcard CORS. | Low |
| Tampering | Source export | Import code might alter or remove original files. | Open source read-only, write derived files only under project root, include source checksum manifest. | Low |
| Tampering | Malicious ZIP import | A crafted archive could use absolute paths or `..` traversal to write outside the intended project root. | Normalize each entry path, reject absolute paths and parent-directory references after normalization, compute the final extraction target and require it to stay inside the project root, extract through a safe API or sandboxed temporary directory, whitelist expected file names and media extensions where feasible, log and surface rejected entries, and record checksums in the import manifest. | High |
| Tampering | Project artifacts | User or malware could edit transcripts or generated audio outside app. | Store checksums, version edits, show modified artifact warnings. | Medium |
| Repudiation | Review edits | User cannot tell machine transcript from human edit. | Append transcript versions, record edited_at, provenance, and job id. | Low |
| Information Disclosure | Logs | Raw message text or transcripts could leak through logs. | Redact by default, log ids/counts/status only, explicit diagnostic export warning. | Low |
| Information Disclosure | Voicebox calls | Sensitive media sent to a service outside user control. | Hard-code default loopback URL, warn before non-loopback override, document local Voicebox privacy model. | Medium |
| Denial of Service | Batch jobs | Large media folders can exhaust disk, CPU, GPU, or Voicebox queue. | Bounded concurrency, preflight disk estimate, cancellable jobs, progress with ETA. | Medium |
| Denial of Service | Video conversion | Long video extraction can block UI. | Background jobs, timeouts, per-file failures, resume support. | Medium |
| Elevation of Privilege | macOS permissions | App could request excessive file or accessibility permissions. | Use file picker scoped access; no Accessibility permission required for MVP import/transcribe flow. | Low |

## Security Controls

### Local-Only Networking

- Default Voicebox base URL is `http://127.0.0.1:17493`.
- Non-loopback Voicebox URLs are advanced settings and require an explicit
  warning that content may leave the Mac.
- No inbound public listener is needed for the MVP.

### Data Protection

- Preserve source exports.
- Store derived artifacts under user-selected artifact root.
- Recommend FileVault or encrypted external volume for sensitive projects.
- Compute SHA-256 for imported attachments and exported artifacts.
- Provide project delete that removes project-local artifacts and database.

### Secrets

- MVP has no cloud credentials.
- Local helper session tokens are ephemeral and never written to logs.
- Voicebox profile ids are not secrets but are treated as local metadata.

### Logging

- Logs include job id, artifact id, filename basename, status, timing, endpoint,
  model, retry count, and error class.
- Logs exclude raw chat text, full transcript text, generated text, and full
  source path by default.
- Diagnostic exports require user confirmation and include a manifest of what is
  included.

### Dependency and API Security

- Treat Voicebox as external: use documented REST endpoints and current upstream
  docs before changing integration behavior.
- Validate response shapes and fail closed on unexpected output.
- Pin and review any future media parser/conversion dependencies.

## Abuse Cases

| Abuse Case | Mitigation |
|---|---|
| User accidentally exports private chat data into a shared folder | Export preview, destination confirmation, manifest, and clear naming. |
| A malicious local service impersonates Voicebox | Service check, response validation, optional advanced fingerprinting, warning on changed base URL. |
| Generated speech is mistaken for original WhatsApp voice note | Generated artifacts carry provenance and are stored separately from original media. |
| Sensitive messages appear in crash reports | Crash/support bundles exclude raw content unless user opts in. |

## Security Acceptance Criteria

- Import smoke test proves source files are unchanged by checksum.
- Voicebox client tests cover connection refused, timeout, non-2xx, and malformed
  response.
- Logs from a sample import/transcribe job contain no message body or transcript.
- Export bundle includes provenance for transcripts and generated audio.
