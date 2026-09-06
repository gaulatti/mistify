# Durable Recorded Media Operations

Mistify provides a private, Redis-backed orchestration contract for recorded
media. It probes input before queueing expensive work, persists every stage,
and returns timecoded, uncertainty-preserving results with model provenance.

## Availability and authentication

All `/media/operations` routes require
`Authorization: Bearer <MEDIA_OPERATION_BEARER_TOKEN>`. When the token is not
configured, the routes return `404`; invalid credentials return `401`.
The environment variable is a local/test override. This change does not enable
production; doing so requires a separate Secrets Manager-owned configuration
integration, and deployment workflows deliberately do not inject the token.

Submissions also require an available processor adapter. Mistify deliberately
ships with a fail-closed adapter because choosing production transcription,
diarization, and summary models requires a representative quality/cost
benchmark. Until a pinned adapter is configured, `POST /media/operations`
returns `503`. Status and cancellation remain available for already-persisted
operations.

## Source contract and preflight

`POST /media/operations` accepts this shape:

```json
{
  "source": {
    "uri": "file:///media/recordings/example.wav",
    "checksum_sha256": "<64 lowercase hex characters>",
    "media_type": "audio/wav",
    "language_hint": "en"
  },
  "options": {
    "max_attempts": 3,
    "retention_seconds": 604800
  }
}
```

Supported probe formats are `audio/wav`, `audio/mpeg`, and `video/mp4`.
Only `file://` sources resolved beneath `MEDIA_INPUT_ROOT` are accepted; remote
URLs, credentials, query strings, fragments, path escapes, oversized media,
overlong media, checksum mismatches, and structurally corrupt files are
rejected before queueing. The mounted source must remain immutable at its URI,
and processor adapters receive both its expected checksum and probe result.

An accepted response is `202` with an operation ID and status URL. A source is
not accepted merely because its extension matches: the service reads the WAV,
MP3, or MP4 structure and verifies the complete SHA-256 digest first.

## Durable lifecycle

The status resource moves through `queued`, `running`, `succeeded`, `failed`,
or `canceled`. It includes total attempts and records for `probe`,
`transcription`, `diarization`, and `summary`, each with state, attempts,
timestamps, a bounded error code, output, and pinned provider/model/version
provenance tied to the source checksum.

Each stage is saved before and after execution. After a restart or retry, the
worker reloads the Redis record and skips stages already marked `succeeded`.
A cached terminal result is delivered again without rerunning processors. A
retryable failure returns the original at-least-once queue receipt to the
pending queue; a non-retryable failure or exhausted attempt budget becomes
terminal. Cancellation is cooperative: an active processor call finishes, but
its output is discarded if the cancellation was persisted in the meantime.

Redis applies `retention_seconds` as a TTL whenever the record changes. The
allowed range is 5 minutes through 30 days.

## Result guarantees

Successful output contains:

- detected language and optional confidence;
- ordered transcript segments bounded by media duration;
- aligned speaker segments, including explicit `unknown` state with no
  fabricated label or confidence;
- summary points whose citations exactly reference returned transcript
  timecodes; and
- probe and processor provenance with provider, model, version, and immutable
  source checksum.

Mistify rejects overlapping/out-of-range transcript segments, diarization that
does not align with the transcript, ungrounded summary citations, and
incomplete provenance. Source URIs, checksums, transcript content, error
details, and operation IDs are not used as Prometheus labels or log context.

## Processor adapter boundary

Implementations provide versioned `transcribe`, `diarize`, and `summarize`
methods through `MediaBackend`. A representative fixture adapter exercises the
complete HTTP, probe, Redis-store, worker, retry, cancellation, provenance, and
metrics path in the test suite. This establishes the side-by-side integration
contract without claiming production model quality or physical capture-device
behavior.
