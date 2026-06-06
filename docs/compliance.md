# Compliance

## Scope

This is a local-first personal productivity and media transformation app. The MVP
does not provide a hosted service, shared tenant storage, payment processing, or
medical workflow. Compliance focus is GDPR-style privacy principles, OWASP ASVS
control themes for local helper/API boundaries, and SOC 2-style operational
discipline for data handling.

## GDPR Mapping

| Principle | ChatParser Control |
|---|---|
| Lawfulness, fairness, transparency | User imports their own WhatsApp export and sees local processing boundaries before transcription/generation. |
| Purpose limitation | Project is scoped to parsing, transforming, transcribing, generating, reviewing, and exporting selected chat artifacts. |
| Data minimization | Store only project metadata, selected media copies, transcript versions, generated audio, and provenance needed for review/export. |
| Accuracy | Machine transcripts are marked as machine-generated until reviewed; edited versions preserve provenance. |
| Storage limitation | User can delete project-local artifacts; exports are user-controlled snapshots. |
| Integrity and confidentiality | Local-only default, checksums, log redaction, loopback Voicebox integration, FileVault guidance. |
| Accountability | Import/export manifests, job events, provenance JSON, and risk register document processing behavior. |

## Data Subject Rights Support

| Right | MVP Support |
|---|---|
| Access | User can view messages, media metadata, transcripts, generated audio, and export manifests locally. |
| Rectification | User can edit transcript versions and participant aliases. |
| Erasure | User can delete project-local artifacts and exports; original WhatsApp source remains outside ChatParser control. |
| Portability | JSON, CSV, HTML, Markdown, and media bundle exports are planned local output formats. |
| Restriction | User can pause or cancel jobs and exclude attachments from processing. |

## OWASP ASVS Theme Mapping

| ASVS Theme | Local Control |
|---|---|
| Architecture | Local app with explicit trust boundaries; Voicebox isolated behind localhost REST. |
| Authentication | No remote accounts; local helper, if present, uses per-session token. |
| Session Management | Helper token is memory-only and expires when app exits. |
| Access Control | File access comes from user-selected paths and project-local artifact root. |
| Validation | WhatsApp parser validates timestamps, attachment paths, MIME detection, and Voicebox response shape. |
| Stored Cryptography | No app-managed secrets in MVP; recommend FileVault/encrypted volumes for at-rest project protection. |
| Error Handling and Logging | Structured logs redact raw message/transcript text by default. |
| Data Protection | Checksums, provenance, local storage, explicit export confirmation. |
| SSRF/Network | Voicebox base URL defaults to loopback; non-loopback override warns user. |
| File Upload | Imported ZIPs/folders are treated as untrusted local input; extraction must prevent path traversal. |

## SOC 2-Style Operational Themes

| Theme | Control |
|---|---|
| Security | Threat model, local-only defaults, response validation, log minimization. |
| Availability | Resumeable jobs, per-file failures, progress with ETA, Voicebox unavailable handling. |
| Processing Integrity | Source checksums, transcript/generation provenance, append-only job events. |
| Confidentiality | Local storage, no cloud service, explicit export boundaries. |
| Privacy | DPIA-lite, retention rules, deletion behavior, purpose limitation. |

## Voicebox Dependency Compliance Notes

Voicebox README states that models, voice data, and captures never leave the
machine, and documents a local REST API, FastAPI backend, and SQLite database.
ChatParser must not broaden that privacy posture by forwarding media to remote
URLs without explicit user action. Any future remote Voicebox host support is a
material privacy change and requires updated security, DPIA, runbook, and user
consent UI.

## Records and Evidence

- `source-manifest.json`: source path, source kind, checksums, import counts.
- `job_event`: progress, errors, ETA, endpoint metadata without raw content.
- `transcript_version.provenance_json`: Voicebox endpoint, model, source hash,
  job id, response metadata.
- `generated_audio.provenance_json`: source text version, profile id, language,
  endpoint, output hash.
- Export `provenance.json`: source project, included records, artifact hashes,
  app version, Voicebox endpoint metadata.

## Out of Scope

- HIPAA: not a healthcare workflow by default.
- PCI DSS: no cardholder data.
- ISO 27001 certification: controls can inform later organizational processes
  but the MVP is local software.
- SOC 2 audit: no hosted service boundary exists in the MVP.
