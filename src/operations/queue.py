import json
import logging
from typing import Optional

from redis.asyncio import Redis

from src.operations.models import OperationEnvelope, QueuedOperation

logger = logging.getLogger("mistify")

QUEUE_NAME = "mistify:operations"
IDEMPOTENCY_PREFIX = "mistify:operation:idempotency"
IDEMPOTENCY_TTL_SECONDS = 7 * 24 * 60 * 60


class OperationQueue:
    """Redis-backed queue for new Mistify async operations.

    This is intentionally separate from the legacy Cronkite polling flow.
    """

    def __init__(self, redis_client: Redis) -> None:
        self.redis = redis_client
        self.queue_name = QUEUE_NAME

    async def enqueue(self, envelope: OperationEnvelope) -> bool:
        queued = QueuedOperation(envelope=envelope)
        serialized = queued.model_dump_json()

        if envelope.operation_type == "analyze_posts":
            return await self._enqueue_analyze_posts(envelope, serialized)

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
            logger.info("Skipping %d duplicate analyze_posts item(s)", len(existing))

        remaining_indices = {
            idx for idx, key in keyed_items if key not in existing
        }
        remaining_items = [
            item for idx, item in enumerate(items) if idx in remaining_indices
        ]

        if not remaining_items:
            logger.info("All analyze_posts items were duplicates; nothing enqueued")
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
        result = await self.redis.brpop(self.queue_name, timeout=timeout_seconds)
        if result is None:
            return None

        _, raw = result
        try:
            payload = json.loads(raw)
            return QueuedOperation.model_validate(payload)
        except Exception as exc:
            logger.error("Failed to parse queued operation: %s", exc)
            return None

    async def size(self) -> int:
        return int(await self.redis.llen(self.queue_name))
