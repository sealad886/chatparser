# DPIA Lite

## Processing Summary

ChatParser lets a Mac user import WhatsApp exports, parse messages and
attachments, transcribe audio/video attachments through local Voicebox REST, and
generate local audio from selected text. Processing occurs on the user's Mac.
The default Voicebox endpoint is `http://127.0.0.1:17493`.

## Personal Data Processed

| Data Category | Examples | Source | Purpose |
|---|---|---|---|
| Chat content | Message body, timestamps, participant names | WhatsApp export | Search, review, export, text transformations. |
| Media | Voice notes, videos, images, documents | WhatsApp export folder | Transcription, review, export packaging. |
| Transcripts | Machine and user-edited text | Voicebox `/transcribe` and user edits | Accessibility, search, summarization, export. |
| Generated audio | Speech generated from selected text | Voicebox `/generate` | Review clips, narration, accessibility output. |
| Metadata | Checksums, durations, MIME types, job events | Local processing | Provenance, integrity, progress, troubleshooting. |
| Voice profile reference | Voicebox profile id and label | Voicebox `/profiles` | Select voice for generation. |

## Data Subjects

- The local user importing the export.
- WhatsApp conversation participants.
- People whose voices or images appear in attachments.
- People represented by Voicebox profiles when generation is used.

## Lawful Basis Considerations

ChatParser is local software. The user is responsible for having a valid basis
to process WhatsApp exports and media. Product UX should make clear that
conversation participants may have privacy rights and that generated audio
should not be presented as original media.

## Necessity and Proportionality

| Need | Design Choice |
|---|---|
| Transcribe media without cloud upload | Use Voicebox local REST API on `127.0.0.1:17493`. |
| Preserve provenance | Store checksums, job ids, endpoint metadata, model/profile/language. |
| Avoid unnecessary source mutation | Treat source export as read-only. |
| Avoid hidden data sharing | No cloud backend; non-loopback Voicebox override requires warning. |
| Review generated or uncertain output | Keep machine transcript status and human-edited versions separate. |

## Privacy Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---:|---|
| Private chat data exported to an unintended location. | High | Destination confirmation, export preview, manifest, no automatic sharing. |
| Sensitive data exposed in logs. | High | Redacted structured logs by default; no raw message/transcript text. |
| Media leaves device through remote Voicebox URL. | High | Loopback default, warning for non-loopback override, documented privacy change. |
| Generated audio creates misleading impersonation risk. | High | Generated audio provenance, separate storage, export labels. |
| User cannot delete derived data. | Medium | Project delete removes project-local database, transcripts, generated audio, staged media, and logs. |
| Parser errors create inaccurate record. | Medium | Raw-line preservation, parse confidence, review warnings, editable metadata. |

## Retention

- Source WhatsApp exports remain outside ChatParser control.
- Project-local staged media, transcripts, generated audio, database, and logs
  remain until the user deletes a project or specific artifacts.
- Export bundles remain wherever the user saves them.
- Voicebox retains its own captures/profiles/models according to Voicebox
  settings; ChatParser does not manage Voicebox SQLite retention.

## International Transfers

No international transfer occurs in the default MVP because processing is local
to the user's Mac and Voicebox is called on loopback. A non-loopback Voicebox URL
or future cloud integration changes this answer and requires a DPIA update.

## User Controls

- Choose source folder/ZIP.
- Choose project artifact root.
- Exclude attachments from transcription.
- Select Voicebox model for transcription.
- Select Voicebox profile and language for generation.
- Edit or reject machine transcripts.
- Delete project-local artifacts.
- Export selected subsets with manifest.

## Residual Risk

Residual risk is medium. WhatsApp exports can contain highly sensitive personal
data about multiple people, and local-only processing does not remove user
responsibility for consent, confidentiality, or lawful use. Controls reduce
unintended disclosure and provenance ambiguity but cannot prevent deliberate
misuse of exported data or generated speech.
