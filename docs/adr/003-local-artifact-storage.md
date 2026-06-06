# ADR-003: Local Artifact Storage

Date: 2026-06-05
Status: Proposed

## Context

WhatsApp exports and derived multimedia artifacts contain sensitive personal
data. ChatParser needs reproducible imports, transcript versioning,
generated-audio provenance, progress logs, and export manifests. Original
WhatsApp source exports should not be mutated.

Voicebox uses its own local SQLite storage for its captures, profiles, and model
state. ChatParser needs its own project-level storage for WhatsApp-specific
metadata and derived artifacts.

## Decision

Store each ChatParser project under a user-selected local artifact root with a
project SQLite database and structured folders for import manifests, staged
media, transcripts, generated audio, exports, and logs.

Original source exports remain read-only. ChatParser stores source paths,
checksums, and project-local copies or normalized artifacts according to project
settings. Derived transcript and generated-audio records are versioned and
linked to source checksums and Voicebox endpoint metadata.

## Alternatives Considered

- Store everything next to the WhatsApp export - Rejected because it risks
  mixing source and derived data and makes cleanup ambiguous.
- Use Voicebox SQLite for ChatParser project metadata - Rejected because it
  crosses ownership boundaries and couples unrelated schemas.
- Store only a single exported HTML bundle - Rejected because review,
  re-transcription, versioning, and generation need a working project database.
- Cloud object storage - Rejected because the MVP is local-first and sensitive
  data should not leave the user's Mac by default.

## Consequences

- Source exports remain protected from accidental mutation.
- Project delete can target only project-local derived data.
- Reproducibility improves through manifests, checksums, and provenance.
- Local disk usage can be high for media-heavy exports, so preflight estimates
  and cleanup controls are required.
- Users are responsible for backing up the artifact root and enabling FileVault
  or encrypted storage when needed.

## References

- [Data Model](../data-model.md)
- [DPIA Lite](../dpia-lite.md)
- [Runbook](../runbook.md)
