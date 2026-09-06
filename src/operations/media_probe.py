from __future__ import annotations

import asyncio
import hashlib
import struct
import wave
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.operations.media_models import MediaProbeResult, MediaSource, MediaType


class MediaProbeError(Exception):
    def __init__(self, code: str, status_code: int = 422) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class LocalMediaProbe:
    """Probe immutable local media mounted beneath one configured root."""

    def __init__(self, root: Path, max_bytes: int, max_duration_seconds: int) -> None:
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.max_duration_ms = max_duration_seconds * 1000

    async def probe(self, source: MediaSource) -> MediaProbeResult:
        return await asyncio.to_thread(self._probe, source)

    def _probe(self, source: MediaSource) -> MediaProbeResult:
        path = self._resolve(source.uri)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise MediaProbeError("source_unavailable") from exc
        if size <= 0:
            raise MediaProbeError("corrupt_media")
        if size > self.max_bytes:
            raise MediaProbeError("media_too_large", status_code=413)

        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise MediaProbeError("source_unavailable") from exc
        checksum = digest.hexdigest()
        if checksum != source.checksum_sha256:
            raise MediaProbeError("checksum_mismatch")

        if source.media_type == MediaType.WAV:
            duration_ms = self._probe_wav(path)
        elif source.media_type == MediaType.MP3:
            duration_ms = self._probe_mp3(path, size)
        elif source.media_type == MediaType.MP4:
            duration_ms = self._probe_mp4(path, size)
        else:
            raise MediaProbeError("unsupported_media_type", status_code=415)

        if duration_ms > self.max_duration_ms:
            raise MediaProbeError("media_too_long")
        return MediaProbeResult(
            checksum_sha256=checksum,
            detected_media_type=source.media_type,
            duration_ms=duration_ms,
            size_bytes=size,
        )

    def _resolve(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise MediaProbeError("unsupported_uri_scheme", status_code=415)
        try:
            candidate = Path(unquote(parsed.path)).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise MediaProbeError("source_unavailable") from exc
        if not candidate.is_file() or not candidate.is_relative_to(self.root):
            raise MediaProbeError("source_outside_media_root")
        return candidate

    @staticmethod
    def _probe_wav(path: Path) -> int:
        try:
            with wave.open(str(path), "rb") as media:
                frames = media.getnframes()
                rate = media.getframerate()
                if frames <= 0 or rate <= 0 or media.getnchannels() <= 0:
                    raise MediaProbeError("corrupt_media")
                return max(1, round(frames / rate * 1000))
        except (wave.Error, EOFError, OSError) as exc:
            raise MediaProbeError("corrupt_media") from exc

    @staticmethod
    def _probe_mp3(path: Path, size: int) -> int:
        try:
            with path.open("rb") as handle:
                sample = handle.read(min(size, 128 * 1024))
        except OSError as exc:
            raise MediaProbeError("source_unavailable") from exc

        offset = 0
        if sample.startswith(b"ID3") and len(sample) >= 10:
            encoded = sample[6:10]
            if any(byte & 0x80 for byte in encoded):
                raise MediaProbeError("corrupt_media")
            offset = 10 + sum(byte << shift for byte, shift in zip(encoded, (21, 14, 7, 0)))

        bitrate_v1_l3 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0)
        bitrate_v2_l3 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)
        for index in range(offset, max(offset, len(sample) - 3)):
            header = int.from_bytes(sample[index : index + 4], "big")
            if header >> 21 != 0x7FF:
                continue
            version_bits = (header >> 19) & 0b11
            layer_bits = (header >> 17) & 0b11
            bitrate_index = (header >> 12) & 0b1111
            sample_rate_index = (header >> 10) & 0b11
            if version_bits == 0b01 or layer_bits != 0b01 or sample_rate_index == 0b11:
                continue
            bitrate = (bitrate_v1_l3 if version_bits == 0b11 else bitrate_v2_l3)[bitrate_index]
            if bitrate:
                audio_bytes = max(1, size - index)
                return max(1, round(audio_bytes * 8 / (bitrate * 1000) * 1000))
        raise MediaProbeError("corrupt_media")

    @classmethod
    def _probe_mp4(cls, path: Path, size: int) -> int:
        try:
            with path.open("rb") as handle:
                boxes = list(cls._boxes(handle, 0, size))
                if not any(kind == b"ftyp" for kind, _, _ in boxes):
                    raise MediaProbeError("corrupt_media")
                for kind, payload_start, payload_end in boxes:
                    if kind != b"moov":
                        continue
                    children = cls._boxes(handle, payload_start, payload_end)
                    for child_kind, child_start, child_end in children:
                        if child_kind == b"mvhd":
                            handle.seek(child_start)
                            payload = handle.read(min(child_end - child_start, 40))
                            return cls._mvhd_duration(payload)
        except (OSError, ValueError, struct.error) as exc:
            raise MediaProbeError("corrupt_media") from exc
        raise MediaProbeError("corrupt_media")

    @staticmethod
    def _boxes(handle, start: int, end: int):
        position = start
        while position + 8 <= end:
            handle.seek(position)
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError("truncated box")
            box_size, kind = struct.unpack(">I4s", header)
            header_size = 8
            if box_size == 1:
                extended = handle.read(8)
                if len(extended) != 8:
                    raise ValueError("truncated extended box")
                box_size = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif box_size == 0:
                box_size = end - position
            if box_size < header_size or position + box_size > end:
                raise ValueError("invalid box size")
            yield kind, position + header_size, position + box_size
            position += box_size

    @staticmethod
    def _mvhd_duration(payload: bytes) -> int:
        if len(payload) < 20:
            raise ValueError("truncated mvhd")
        version = payload[0]
        if version == 0:
            timescale = struct.unpack(">I", payload[12:16])[0]
            duration = struct.unpack(">I", payload[16:20])[0]
        elif version == 1 and len(payload) >= 32:
            timescale = struct.unpack(">I", payload[20:24])[0]
            duration = struct.unpack(">Q", payload[24:32])[0]
        else:
            raise ValueError("unsupported mvhd")
        if timescale <= 0 or duration <= 0:
            raise ValueError("invalid duration")
        return max(1, round(duration / timescale * 1000))
