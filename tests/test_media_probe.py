"""Covers `src/operations/media_probe.py`.

The probe is the trust boundary for recorded media: it must resolve a `file://`
URI only beneath the configured root, verify size and checksum, and read a
plausible duration out of WAV, MP3 and MP4 containers, failing with a stable code
(and HTTP status) for everything else. Fixtures are tiny hand-built files.
"""

import hashlib
import io
import os
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.operations.media_models import MediaSource, MediaType
from src.operations.media_probe import LocalMediaProbe, MediaProbeError


# --- fixtures ---------------------------------------------------------------

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wav_bytes(frames=8000, rate=8000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as media:
        media.setnchannels(channels)
        media.setsampwidth(2)
        media.setframerate(rate)
        media.writeframes(b"\x00\x00" * channels * frames)
    return buffer.getvalue()


def box(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def mvhd_v0(timescale=1000, duration=2500):
    return box(b"mvhd", bytes([0, 0, 0, 0]) + b"\x00" * 8 + struct.pack(">II", timescale, duration) + b"\x00" * 20)


def mvhd_v1(timescale=1000, duration=4000):
    return box(b"mvhd", bytes([1, 0, 0, 0]) + b"\x00" * 16 + struct.pack(">IQ", timescale, duration) + b"\x00" * 8)


def mp4_bytes(mvhd=None):
    mvhd = mvhd if mvhd is not None else mvhd_v0()
    return box(b"ftyp", b"isom\x00\x00\x02\x00") + box(b"moov", mvhd)


MP3_FRAME_128K = bytes([0xFF, 0xFB, 0x90, 0x64])  # MPEG-1 layer 3, 128 kbit/s, 44.1 kHz


def mp3_bytes(prefix=b"", frames=40):
    return prefix + MP3_FRAME_128K + b"\x00" * (frames * 417 - 4)


def id3_header(body_size):
    encoded = bytes([(body_size >> 21) & 0x7F, (body_size >> 14) & 0x7F, (body_size >> 7) & 0x7F, body_size & 0x7F])
    return b"ID3\x03\x00\x00" + encoded


def run(tmp_path, data, media_type, name="media.bin", checksum=None, **limits):
    path = tmp_path / name
    path.write_bytes(data)
    return probe_path(tmp_path, path, media_type, checksum or sha(data), **limits)


def probe_path(tmp_path, path, media_type, checksum, root=None, uri=None, **limits):
    limits.setdefault("max_bytes", 10_000_000)
    limits.setdefault("max_duration_seconds", 3600)
    probe = LocalMediaProbe(root or tmp_path, **limits)
    source = SimpleNamespace(uri=uri or path.as_uri(), checksum_sha256=checksum, media_type=media_type)
    return probe._probe(source)


def failure(tmp_path, data, media_type, **kwargs):
    with pytest.raises(MediaProbeError) as caught:
        run(tmp_path, data, media_type, **kwargs)
    return caught.value


# --- error type -------------------------------------------------------------

def test_probe_errors_carry_a_code_and_default_to_422():
    error = MediaProbeError("corrupt_media")
    assert (error.code, error.status_code, str(error)) == ("corrupt_media", 422, "corrupt_media")
    assert MediaProbeError("media_too_large", status_code=413).status_code == 413


# --- successful probes ------------------------------------------------------

def test_wav_duration_size_and_checksum_are_reported(tmp_path):
    data = wav_bytes(frames=12000, rate=8000)

    result = run(tmp_path, data, MediaType.WAV)

    assert result.duration_ms == 1500
    assert result.size_bytes == len(data)
    assert result.checksum_sha256 == sha(data)
    assert result.detected_media_type == MediaType.WAV


def test_sub_millisecond_wav_rounds_up_to_one_millisecond(tmp_path):
    assert run(tmp_path, wav_bytes(frames=1, rate=48000), MediaType.WAV).duration_ms == 1


def test_mp3_duration_comes_from_the_first_valid_frame_bitrate(tmp_path):
    data = mp3_bytes()

    result = run(tmp_path, data, MediaType.MP3)

    assert result.duration_ms == round(len(data) * 8 / 128_000 * 1000)


def test_mp3_id3_tag_is_skipped_before_the_frame_search(tmp_path):
    tag = id3_header(20) + b"\x00" * 20
    data = mp3_bytes(prefix=tag)

    result = run(tmp_path, data, MediaType.MP3)

    assert result.duration_ms == round((len(data) - len(tag)) * 8 / 128_000 * 1000)


def test_mp3_with_an_mpeg2_header_uses_the_mpeg2_bitrate_table(tmp_path):
    frame = bytes([0xFF, 0xF3, 0x90, 0x64])  # MPEG-2 layer 3, index 9 = 80 kbit/s
    data = frame + b"\x00" * 9996

    assert run(tmp_path, data, MediaType.MP3).duration_ms == round(len(data) * 8 / 80_000 * 1000)


@pytest.mark.parametrize(
    "bad_header",
    [
        bytes([0xFF, 0xEB, 0x90, 0x64]),  # reserved MPEG version
        bytes([0xFF, 0xFD, 0x90, 0x64]),  # layer 2, not layer 3
        bytes([0xFF, 0xFB, 0x9C, 0x64]),  # reserved sample-rate index
        bytes([0xFF, 0xFB, 0x00, 0x64]),  # free-format bitrate (index 0)
        bytes([0xFF, 0xFB, 0xF0, 0x64]),  # invalid bitrate (index 15)
    ],
)
def test_mp3_skips_implausible_headers_and_uses_a_later_valid_one(tmp_path, bad_header):
    data = bad_header + b"\x00" * 16 + MP3_FRAME_128K + b"\x00" * 2000

    result = run(tmp_path, data, MediaType.MP3)

    audio_start = len(bad_header) + 16
    assert result.duration_ms == round((len(data) - audio_start) * 8 / 128_000 * 1000)


def test_mp4_duration_comes_from_a_version_zero_mvhd(tmp_path):
    assert run(tmp_path, mp4_bytes(mvhd_v0(1000, 2500)), MediaType.MP4).duration_ms == 2500


def test_mp4_duration_comes_from_a_version_one_mvhd(tmp_path):
    assert run(tmp_path, mp4_bytes(mvhd_v1(1000, 4000)), MediaType.MP4).duration_ms == 4000


def test_mp4_scans_past_other_top_level_boxes_and_moov_children(tmp_path):
    data = box(b"ftyp", b"isom0000") + box(b"free", b"\x00" * 12) + box(
        b"moov", box(b"udta", b"abcd") + mvhd_v0(600, 1200)
    )

    assert run(tmp_path, data, MediaType.MP4).duration_ms == 2000


def test_mp4_box_with_extended_size_and_open_ended_last_box_is_understood(tmp_path):
    ftyp = box(b"ftyp", b"isom0000")
    moov_payload = mvhd_v0(1000, 3000)
    extended_moov = struct.pack(">I4sQ", 1, b"moov", 16 + len(moov_payload)) + moov_payload
    open_ended = struct.pack(">I4s", 0, b"mdat") + b"\x00" * 10

    assert run(tmp_path, ftyp + extended_moov + open_ended, MediaType.MP4).duration_ms == 3000


# --- limits -----------------------------------------------------------------

def test_media_longer_than_the_limit_is_rejected(tmp_path):
    error = failure(tmp_path, wav_bytes(frames=24000, rate=8000), MediaType.WAV, max_duration_seconds=2)

    assert (error.code, error.status_code) == ("media_too_long", 422)


def test_media_exactly_at_the_duration_limit_is_accepted(tmp_path):
    assert run(tmp_path, wav_bytes(frames=16000, rate=8000), MediaType.WAV, max_duration_seconds=2).duration_ms == 2000


def test_oversized_media_is_a_413(tmp_path):
    error = failure(tmp_path, wav_bytes(), MediaType.WAV, max_bytes=10)

    assert (error.code, error.status_code) == ("media_too_large", 413)


def test_an_empty_file_is_corrupt(tmp_path):
    assert failure(tmp_path, b"", MediaType.WAV).code == "corrupt_media"


def test_a_wrong_checksum_is_rejected(tmp_path):
    assert failure(tmp_path, wav_bytes(), MediaType.WAV, checksum="0" * 64).code == "checksum_mismatch"


def test_an_unsupported_media_type_is_a_415(tmp_path):
    error = failure(tmp_path, b"OggS", "audio/ogg")

    assert (error.code, error.status_code) == ("unsupported_media_type", 415)


# --- URI resolution ---------------------------------------------------------

def test_localhost_authority_and_percent_encoding_are_accepted(tmp_path):
    data = wav_bytes()
    path = tmp_path / "my clip.wav"
    path.write_bytes(data)

    result = probe_path(tmp_path, path, MediaType.WAV, sha(data), uri=f"file://localhost{str(path).replace(' ', '%20')}")

    assert result.size_bytes == len(data)


@pytest.mark.parametrize("uri", ["https://example.com/a.wav", "s3://bucket/a.wav", "file://evil.example/etc/passwd"])
def test_non_local_uris_are_a_415(tmp_path, uri):
    path = tmp_path / "a.wav"
    path.write_bytes(wav_bytes())

    with pytest.raises(MediaProbeError) as caught:
        probe_path(tmp_path, path, MediaType.WAV, "0" * 64, uri=uri)

    assert (caught.value.code, caught.value.status_code) == ("unsupported_uri_scheme", 415)


def test_a_missing_file_is_unavailable(tmp_path):
    with pytest.raises(MediaProbeError) as caught:
        probe_path(tmp_path, tmp_path / "missing.wav", MediaType.WAV, "0" * 64)

    assert caught.value.code == "source_unavailable"


def test_a_symlink_loop_is_unavailable(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.symlink_to(second)
    second.symlink_to(first)

    with pytest.raises(MediaProbeError) as caught:
        probe_path(tmp_path, first, MediaType.WAV, "0" * 64)

    assert caught.value.code == "source_unavailable"


def test_a_directory_is_outside_the_media_root(tmp_path):
    with pytest.raises(MediaProbeError) as caught:
        probe_path(tmp_path, tmp_path, MediaType.WAV, "0" * 64)

    assert caught.value.code == "source_outside_media_root"


def test_a_symlink_escaping_the_root_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.wav"
    secret.write_bytes(wav_bytes())
    link = root / "link.wav"
    link.symlink_to(secret)

    with pytest.raises(MediaProbeError) as caught:
        probe_path(root, link, MediaType.WAV, sha(secret.read_bytes()), root=root)

    assert caught.value.code == "source_outside_media_root"


# --- I/O failures after resolution ------------------------------------------

def test_a_file_that_vanishes_after_resolution_is_unavailable(tmp_path, monkeypatch):
    probe = LocalMediaProbe(tmp_path, 1_000_000, 60)
    monkeypatch.setattr(probe, "_resolve", lambda uri: tmp_path / "gone.wav")
    source = SimpleNamespace(uri="file:///x", checksum_sha256="0" * 64, media_type=MediaType.WAV)

    with pytest.raises(MediaProbeError) as caught:
        probe._probe(source)

    assert caught.value.code == "source_unavailable"


def _fail_open_after(monkeypatch, allowed_opens):
    """Make `Path.open` raise after `allowed_opens` further calls."""
    real_open = Path.open
    state = {"calls": 0}

    def flaky(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] > allowed_opens:
            raise OSError("disk went away")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky)


def test_a_read_failure_while_hashing_is_unavailable(tmp_path, monkeypatch):
    data = wav_bytes()
    path = tmp_path / "clip.wav"
    path.write_bytes(data)
    _fail_open_after(monkeypatch, 0)

    with pytest.raises(MediaProbeError) as caught:
        probe_path(tmp_path, path, MediaType.WAV, sha(data))

    assert caught.value.code == "source_unavailable"


def test_a_read_failure_while_sampling_an_mp3_is_unavailable(tmp_path, monkeypatch):
    data = mp3_bytes()
    path = tmp_path / "clip.mp3"
    path.write_bytes(data)
    _fail_open_after(monkeypatch, 1)  # hashing succeeds, the frame sample fails

    with pytest.raises(MediaProbeError) as caught:
        probe_path(tmp_path, path, MediaType.MP3, sha(data))

    assert caught.value.code == "source_unavailable"


# --- corrupt containers -----------------------------------------------------

def test_non_wav_bytes_are_corrupt(tmp_path):
    assert failure(tmp_path, b"this is not riff data", MediaType.WAV).code == "corrupt_media"


def test_a_wav_without_frames_is_corrupt(tmp_path):
    assert failure(tmp_path, wav_bytes(frames=0), MediaType.WAV).code == "corrupt_media"


def test_a_wav_with_a_zero_sample_rate_is_corrupt(tmp_path):
    data = bytearray(wav_bytes())
    data[24:28] = struct.pack("<I", 0)  # sample rate field of the fmt chunk

    assert failure(tmp_path, bytes(data), MediaType.WAV).code == "corrupt_media"


def test_an_mp3_without_any_frame_is_corrupt(tmp_path):
    assert failure(tmp_path, b"\x00" * 500, MediaType.MP3).code == "corrupt_media"


def test_an_mp3_with_an_invalid_id3_size_is_corrupt(tmp_path):
    data = b"ID3\x03\x00\x00" + bytes([0x80, 0, 0, 0]) + MP3_FRAME_128K + b"\x00" * 500

    assert failure(tmp_path, data, MediaType.MP3).code == "corrupt_media"


def test_an_mp3_whose_tag_swallows_the_frames_is_corrupt(tmp_path):
    data = id3_header(100_000) + MP3_FRAME_128K + b"\x00" * 500

    assert failure(tmp_path, data, MediaType.MP3).code == "corrupt_media"


@pytest.mark.parametrize(
    "data",
    [
        b"not media",                                          # truncated first box header
        box(b"moov", mvhd_v0()),                               # no ftyp
        box(b"ftyp", b"isom0000"),                             # no moov
        box(b"ftyp", b"isom0000") + box(b"moov", box(b"udta")),  # moov without mvhd
        box(b"ftyp", b"isom0000") + box(b"moov", mvhd_v0(0, 100)),   # zero timescale
        box(b"ftyp", b"isom0000") + box(b"moov", mvhd_v0(100, 0)),   # zero duration
        box(b"ftyp", b"isom0000") + box(b"moov", box(b"mvhd", bytes([2]) + b"\x00" * 31)),  # unknown version
        box(b"ftyp", b"isom0000") + box(b"moov", box(b"mvhd", bytes([0]) + b"\x00" * 6)),   # truncated mvhd
        box(b"ftyp", b"isom0000") + struct.pack(">I4s", 4, b"moov"),   # size below header size
        box(b"ftyp", b"isom0000") + struct.pack(">I4s", 9999, b"moov"),  # size beyond the file
        box(b"ftyp", b"isom0000") + struct.pack(">I4s", 1, b"moov") + b"\x00\x00",  # truncated extended size
    ],
)
def test_malformed_mp4_containers_are_corrupt(tmp_path, data):
    assert failure(tmp_path, data, MediaType.MP4).code == "corrupt_media"


# --- box and mvhd helpers ---------------------------------------------------

def test_boxes_yields_kind_and_payload_bounds():
    data = box(b"ftyp", b"1234") + box(b"free", b"")

    assert list(LocalMediaProbe._boxes(io.BytesIO(data), 0, len(data))) == [
        (b"ftyp", 8, 12), (b"free", 20, 20),
    ]


def test_boxes_stops_when_less_than_a_header_remains():
    data = box(b"ftyp", b"1234") + b"\x00\x00\x00"

    assert len(list(LocalMediaProbe._boxes(io.BytesIO(data), 0, len(data)))) == 1


def test_mvhd_duration_converts_ticks_to_milliseconds():
    payload = bytes([0, 0, 0, 0]) + b"\x00" * 8 + struct.pack(">II", 30000, 45000)

    assert LocalMediaProbe._mvhd_duration(payload) == 1500


def test_mvhd_duration_rejects_short_payloads():
    with pytest.raises(ValueError, match="truncated"):
        LocalMediaProbe._mvhd_duration(b"\x00" * 19)


def test_a_version_one_mvhd_needs_its_full_header():
    with pytest.raises(ValueError, match="unsupported"):
        LocalMediaProbe._mvhd_duration(bytes([1]) + b"\x00" * 24)


# --- async entry point ------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_runs_the_blocking_work_off_the_event_loop(tmp_path):
    data = wav_bytes()
    path = tmp_path / "clip.wav"
    path.write_bytes(data)
    source = MediaSource(uri=path.as_uri(), checksum_sha256=sha(data), media_type=MediaType.WAV)

    result = await LocalMediaProbe(tmp_path, 1_000_000, 60).probe(source)

    assert result.duration_ms == 1000 and result.checksum_sha256 == sha(data)


def test_the_root_is_resolved_once_at_construction(tmp_path):
    link = tmp_path / "link"
    (tmp_path / "real").mkdir()
    link.symlink_to(tmp_path / "real")

    probe = LocalMediaProbe(link, 100, 5)

    assert probe.root == (tmp_path / "real").resolve()
    assert probe.max_duration_ms == 5000 and probe.max_bytes == 100


def test_boxes_rejects_a_stream_shorter_than_the_range_it_was_asked_to_walk():
    with pytest.raises(ValueError, match="truncated box"):
        list(LocalMediaProbe._boxes(io.BytesIO(b"\x00" * 5), 0, 100))
