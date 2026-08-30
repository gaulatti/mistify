import json
import logging
from typing import Optional

from redis.asyncio import Redis

from src import metrics
from src.operations.models import OperationEnvelope, QueuedOperation

logger = logging.getLogger("mistify")

QUEUE_NAME = "mistify:operations"
PROCESSING_QUEUE_NAME = "mistify:operations:processing"
DEAD_LETTER_QUEUE_NAME = "mistify:operations:dead-letter"
IDEMPOTENCY_PREFIX = "mistify:operation:idempotency"
IDEMPOTENCY_TTL_SECONDS = 7 * 24 * 60 * 60


class OperationQueue:
    """Redis-backed queue for async Mistify operations."""

    def __init__(self, redis_client: Redis) -> None:
        self.redis = redis_client
        self.queue_name = QUEUE_NAME
        self.processing_queue_name = PROCESSING_QUEUE_NAME
        self.dead_letter_queue_name = DEAD_LETTER_QUEUE_NAME

    async def enqueue(self, envelope: OperationEnvelope) -> bool:
        queued = QueuedOperation(envelope=envelope)
        serialized = queued.model_dump_json()

        if envelope.operation_type == "analyze_posts":
            return await self._enqueue_analyze_posts(envelope, serialized)

        if (
            envelope.operation_type == "cluster_post"
            and str(envelope.payload.get("source", "")).casefold() == "youtube"
        ):
            # YouTube Scout is interactive and its analysis is already
            # prioritized. Keep its immediate clustering pass adjacent rather
            # than placing it behind the general RSS backlog.
            await self.redis.rpush(self.queue_name, serialized)
            return True

        if envelope.idempotency_key:
            return await self._enqueue_once(envelope.idempotency_key, serialized)

        await self.redis.lpush(self.queue_name, serialized)
        return True

    async def _enqueue_analyze_posts(
        self, envelope: OperationEnvelope, serialized: str
    ) -> bool:
        """Enqueue an analyze_posts operation with per-item idempotency.

        Items whose idempotency_key already exists are filtered out. If no
        items remain, the operation is not enqueued.
        """
        if envelope.payload.get("force"):
            # Manual reanalysis (currently Scout) must not sit behind the
            # autonomous discovery backlog.  Consumers use BRPOP, so RPUSH
            # places forced work at the next-consumed end of the queue.
            await self.redis.rpush(self.queue_name, serialized)
            return True

        items = envelope.payload.get("items") or []
        if not isinstance(items, list):
            items = []

        keys = []
        keyed_items = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            key = item.get("idempotency_key")
            if key:
                keys.append(key)
                keyed_items.append((idx, key))

        if not keys:
            await self.redis.lpush(self.queue_name, serialized)
            return True

        existing = await self._existing_keys(keys)
        if existing:
            logger.debug("Skipping %d duplicate analyze_posts item(s)", len(existing))

        remaining_indices = {
            idx for idx, key in keyed_items if key not in existing
        }
        remaining_items = [
            item for idx, item in enumerate(items) if idx in remaining_indices
        ]

        if not remaining_items:
            logger.debug("All analyze_posts items were duplicates; nothing enqueued")
            return False

        if len(remaining_items) != len(items):
            envelope.payload["items"] = remaining_items
            envelope.payload["skipped_count"] = len(items) - len(remaining_items)
            queued = QueuedOperation(envelope=envelope)
            serialized = queued.model_dump_json()

        new_keys = [key for _, key in keyed_items if key not in existing]
        if new_keys:
            await self._mark_keys(new_keys)

        await self.redis.lpush(self.queue_name, serialized)
        return True

    async def _existing_keys(self, keys: list[str]) -> set[str]:
        """Return the subset of keys that already exist in Redis."""
        if not keys:
            return set()

        pipe = self.redis.pipeline()
        for key in keys:
            pipe.exists(f"{IDEMPOTENCY_PREFIX}:{key}")
        results = await pipe.execute()

        return {key for key, exists in zip(keys, results) if exists}

    async def _mark_keys(self, keys: list[str]) -> None:
        """Mark keys as present in Redis with the configured TTL."""
        if not keys:
            return

        pipe = self.redis.pipeline()
        for key in keys:
            pipe.set(
                f"{IDEMPOTENCY_PREFIX}:{key}",
                "1",
                ex=IDEMPOTENCY_TTL_SECONDS,
            )
        await pipe.execute()

    async def _enqueue_once(self, idempotency_key: str, serialized: str) -> bool:
        dedupe_key = f"{IDEMPOTENCY_PREFIX}:{idempotency_key}"
        script = """
          local ttl = tonumber(ARGV[1])
          local lock
          if ttl and ttl > 0 then
            lock = redis.call('SET', KEYS[1], '1', 'NX', 'EX', ttl)
          else
            lock = redis.call('SET', KEYS[1], '1', 'NX')
          end
          if lock then
            redis.call('LPUSH', KEYS[2], ARGV[2])
            return 1
          end
          return 0
        """

        response = await self.redis.eval(
            script,
            2,
            dedupe_key,
            self.queue_name,
            str(IDEMPOTENCY_TTL_SECONDS),
            serialized,
        )
        return int(response) == 1

    async def dequeue(self, timeout_seconds: int = 5) -> Optional[QueuedOperation]:
        raw = await self.redis.brpoplpush(
            self.queue_name,
            self.processing_queue_name,
            timeout=timeout_seconds,
        )
        if raw is None:
            return None
        queued = await self._parse_claimed(raw)
        if queued is not None:
            metrics.record_queue_event(queued.envelope.operation_type, "claimed")
        return queued

    async def dequeue_nowait(self) -> Optional[QueuedOperation]:
        """Pop the next operation without blocking."""
        raw = await self.redis.rpoplpush(
            self.queue_name,
            self.processing_queue_name,
        )
        if raw is None:
            return None
        queued = await self._parse_claimed(raw)
        if queued is not None:
            metrics.record_queue_event(queued.envelope.operation_type, "claimed")
        return queued

    async def requeue_next(self, queued: QueuedOperation) -> None:
        """Put an inspected operation back at the next-consumed end."""
        await self._release(queued, next_consumed=True)

    async def requeue_later(self, queued: QueuedOperation) -> None:
        """Return failed delivery work behind operations that have not run yet."""
        await self._release(queued, next_consumed=False)
        metrics.record_queue_event(queued.envelope.operation_type, "requeued")

    async def acknowledge(self, queued: QueuedOperation) -> None:
        """Remove successfully completed claims from the processing list."""
        receipts = queued.receipts or [queued.model_dump_json()]
        pipe = self.redis.pipeline(transaction=True)
        for receipt in receipts:
            pipe.lrem(self.processing_queue_name, 1, receipt)
        await pipe.execute()
        metrics.record_queue_event(queued.envelope.operation_type, "acknowledged")

    async def recover_inflight(self) -> int:
        """Return claims left by an interrupted worker to the durable queue."""
        recovered = 0
        while True:
            raw = await self.redis.rpoplpush(
                self.processing_queue_name,
                self.queue_name,
            )
            if raw is None:
                break
            recovered += 1
        if recovered:
            metrics.record_queue_event("unknown", "recovered")
        return recovered

    async def _release(
        self,
        queued: QueuedOperation,
        *,
        next_consumed: bool,
    ) -> None:
        receipts = queued.receipts or [queued.model_dump_json()]
        pipe = self.redis.pipeline(transaction=True)
        for receipt in receipts:
            pipe.lrem(self.processing_queue_name, 1, receipt)
            if next_consumed:
                pipe.rpush(self.queue_name, receipt)
            else:
                pipe.lpush(self.queue_name, receipt)
        await pipe.execute()

    async def _parse_claimed(self, raw: str) -> Optional[QueuedOperation]:
        queued = self._parse(raw)
        if queued is not None:
            queued.receipts = [raw]
            return queued

        pipe = self.redis.pipeline(transaction=True)
        pipe.lrem(self.processing_queue_name, 1, raw)
        pipe.lpush(self.dead_letter_queue_name, raw)
        await pipe.execute()
        metrics.record_queue_event("unknown", "dead_letter")
        return None

    @staticmethod
    def _parse(raw: str) -> Optional[QueuedOperation]:
        try:
            payload = json.loads(raw)
            return QueuedOperation.model_validate(payload)
        except Exception as exc:
            logger.error("Failed to parse queued operation: %s", exc)
            return None

    async def size(self) -> int:
        pipe = self.redis.pipeline(transaction=False)
        pipe.llen(self.queue_name)
        pipe.llen(self.processing_queue_name)
        pending, processing = await pipe.execute()
        return int(pending) + int(processing)
