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

        if envelope.idempotency_key:
            return await self._enqueue_once(envelope.idempotency_key, serialized)

        await self.redis.lpush(self.queue_name, serialized)
        return True

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
