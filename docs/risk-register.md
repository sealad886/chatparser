# Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Owner | Status |
|---|---|---:|---:|---|---|---|
| R-001 | Voicebox API changes from documented README shape. | Medium | High | Keep Voicebox client isolated, validate `/profiles`, `/transcribe`, `/generate` in integration tests, link endpoint metadata to artifacts. | Engineering | Monitoring |
| R-002 | Large WhatsApp exports exhaust disk during media staging. | Medium | High | Preflight size estimate, copy/link mode, free-space check, resumable import, per-file progress with ETA. | Engineering | Open |
| R-003 | Machine transcript is treated as authoritative. | Medium | Medium | Mark transcript status as machine until reviewed, preserve versions, show source media playback in review UI. | Product | Open |
| R-004 | Sensitive content leaks through logs or support bundles. | Low | High | Default log redaction, no raw transcript/message text in events, explicit diagnostic export confirmation. | Engineering | Open |
| R-005 | Non-loopback Voicebox URL sends private media off-device. | Low | High | Default to `127.0.0.1`, warn and require explicit confirmation for remote URLs, and ensure the DPIA covers remote URL privacy risks. | Product | Monitoring |
| R-006 | WhatsApp export parser mishandles locale-specific timestamps. | Medium | Medium | Store raw lines, parse confidence, import warnings, locale fixtures, user correction path. | Engineering | Open |
| R-007 | Generated speech is confused with original voice notes. | Medium | High | Store generated audio separately, require provenance labels in UI and exports, never overwrite original media. | Product | Open |
| R-008 | Malicious ZIP import writes outside project root. | Low | High | Use safe archive extraction, reject absolute paths and `..` traversal, checksum extracted files. | Engineering | Open |
| R-009 | Voicebox unavailable blocks batch work. | Medium | Medium | Queue jobs, run service preflight with `GET /profiles`, retry with backoff, preserve job state. | Engineering | Open |
| R-010 | Local project deletion removes source data by mistake. | Low | High | Track source paths separately, delete only project root by default, require confirmation showing exact deletion root. | Engineering | Open |
