# ADR-001: macOS SwiftPM Shell

Date: 2026-06-05
Status: Proposed

## Context

Chatparser needs a Mac-first local app for importing WhatsApp exports, reviewing
messages and multimedia, running long local jobs, and coordinating a local
Voicebox service. The app should feel native on macOS, use user-selected file
access, and avoid a hosted backend for the MVP.

The repository rules restrict this task to documentation only. This ADR records
the intended architecture decision for later implementation.

## Decision

Use a SwiftPM-built macOS shell as the primary user experience. The shell owns
file pickers, project selection, settings, review UI, job controls, and export
flow. Heavy parsing, media staging, and Voicebox client code may live in a local
processing module or helper, but the product entry point is a native macOS app.

The app remains local-first: no cloud account, no remote API, and no public
listener in the MVP.

## Alternatives Considered

- Web-only app - Rejected because source folder access, long local jobs, and
  Mac-first file workflows are more ergonomic in a native shell.
- Electron app - Rejected because the product target is Mac-first and a lighter
  native shell avoids bundling a browser runtime for the MVP.
- Command-line only - Rejected because transcript review, generated-audio
  provenance, export previews, and user correction flows need an interactive UI.

## Consequences

- Native file picker and sandbox-friendly user consent flows are straightforward.
- UI can show progress, ETA, and review states without exposing a network
  service.
- Implementation requires Swift/macOS expertise and careful bridging to any
  local processing helper.
- Cross-platform UI is deferred.

## References

- [Architecture](../architecture.md)
- [Runbook](../runbook.md)
