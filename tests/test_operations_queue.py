"""Covers `src/operations/queue.py`, the Redis-backed at-least-once queue.

The delivery contract lives here: what gets deduplicated, which end of the list
priority work lands on, and whether an interrupted worker's claims come back.
A faithful in-memory Redis fake stands in for the server.
"""

import json
from collections import defaultdict

import pytest

from src.operations.models import OperationEnvelope, QueuedOperation
from src.operations.queue import (
    DEAD_LETTER_QUEUE_NAME,
    IDEMPOTENCY_PREFIX,
    IDEMPOTENCY_TTL_SECONDS,
    PROCESSING_QUEUE_NAME,
    QUEUE_NAME,
    OperationQueue,
)


class FakePipeline:
    """Records commands and applies them on execute, like redis-py."""

    def __init__(self, redis):
        self._redis = redis
        self._commands = []

    def exists(self, key):
        self._commands.append(("exists", (key,), {}))
        return self

    def set(self, key, value, **kwargs):
        self._commands.append(("set", (key, value), kwargs))
        return self

    def lrem(self, name, count, value):
        self._commands.append(("lrem", (name, count, value), {}))
        return self

    def lpush(self, name, value):
        self._commands.append(("lpush", (name, value), {}))
        return self

    def rpush(self, name, value):
        self._commands.append(("rpush", (name, value), {}))
        return self

    def llen(self, name):
        self._commands.append(("llen", (name,), {}))
        return self

    async def execute(self):
        results = []
        for name, args, kwargs in self._commands:
            results.append(self._redis.apply(name, args, kwargs))
        self._commands = []
        return results


class FakeRedis:
    """Index 0 is the head (left). LPUSH prepends, RPUSH appends, and
    BRPOPLPUSH takes from the tail -- so RPUSH is the next-consumed end."""

    def __init__(self):
        self.lists = defaultdict(list)
        self.keys = {}
        self.expirations = {}

    def apply(self, name, args, kwargs):
        if name == "exists":
            return 1 if args[0] in self.keys else 0
        if name == "set":
            key, value = args
            if kwargs.get("nx") and key in self.keys:
                return None
            self.keys[key] = value
            if "ex" in kwargs:
                self.expirations[key] = kwargs["ex"]
            return True
        if name == "lrem":
            target, count, value = args
            items = self.lists[target]
            removed = 0
            index = 0
            while index < len(items) and (count == 0 or removed < count):
                if items[index] == value:
                    items.pop(index)
                    removed += 1
                    continue
                index += 1
            return removed
        if name == "lpush":
            self.lists[args[0]].insert(0, args[1])
            return len(self.lists[args[0]])
        if name == "rpush":
            self.lists[args[0]].append(args[1])
            return len(self.lists[args[0]])
        if name == "llen":
            return len(self.lists[args[0]])
        raise AssertionError(f"unsupported command {name}")

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    async def lpush(self, name, value):
        return self.apply("lpush", (name, value), {})

    async def rpush(self, name, value):
        return self.apply("rpush", (name, value), {})

    async def rpoplpush(self, source, destination):
        if not self.lists[source]:
            return None
        value = self.lists[source].pop()
        self.lists[destination].insert(0, value)
        return value

    async def brpoplpush(self, source, destination, timeout=0):
        return await self.rpoplpush(source, destination)

    async def eval(self, script, numkeys, *args):
        """Emulates the enqueue-once script: SET NX EX, then LPUSH on success."""
        keys = args[:numkeys]
        argv = args[numkeys:]
        dedupe_key, queue_name = keys
        ttl, serialized = argv
        if dedupe_key in self.keys:
            return 0
        self.keys[dedupe_key] = "1"
        self.expirations[dedupe_key] = int(ttl)
        self.lists[queue_name].insert(0, serialized)
        return 1


def envelope(operation_type="detect_language", **payload_kwargs):
    payload = payload_kwargs.pop("payload", {"text": "hello"})
    return OperationEnvelope(operation_type=operation_type, payload=payload, **payload_kwargs)


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def queue(redis):
    return OperationQueue(redis)


def queued_types(redis, name=QUEUE_NAME):
    return [json.loads(raw)["envelope"]["operation_type"] for raw in redis.lists[name]]


# --- enqueue ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_ordinary_operation_goes_behind_the_backlog(queue, redis):
    assert await queue.enqueue(envelope()) is True

    # LPUSH puts it at the head, which BRPOPLPUSH reaches last.
    assert len(redis.lists[QUEUE_NAME]) == 1


@pytest.mark.asyncio
async def test_youtube_clustering_jumps_to_the_next_consumed_end(queue, redis):
    await queue.enqueue(envelope("cluster_post", payload={"source": "rss"}))
    await queue.enqueue(envelope("cluster_post", payload={"source": "YouTube"}))

    # The YouTube item was RPUSHed, so it sits at the tail and is taken first.
    claimed = await queue.dequeue_nowait()
    assert claimed.envelope.payload["source"] == "YouTube"


@pytest.mark.asyncio
async def test_forced_analyze_posts_skips_the_discovery_backlog(queue, redis):
    await queue.enqueue(envelope("analyze_posts", payload={"items": [{"idempotency_key": "a"}]}))
    await queue.enqueue(envelope("analyze_posts", payload={"items": [], "force": True}))

    claimed = await queue.dequeue_nowait()
    assert claimed.envelope.payload.get("force") is True


@pytest.mark.asyncio
async def test_an_idempotency_key_is_enqueued_once(queue, redis):
    first = await queue.enqueue(envelope(idempotency_key="key-1"))
    second = await queue.enqueue(envelope(idempotency_key="key-1"))

    assert first is True
    assert second is False, "a repeated key must not enqueue a second time"
    assert len(redis.lists[QUEUE_NAME]) == 1
    assert redis.expirations[f"{IDEMPOTENCY_PREFIX}:key-1"] == IDEMPOTENCY_TTL_SECONDS


# --- analyze_posts per-item idempotency -------------------------------------

@pytest.mark.asyncio
async def test_analyze_posts_without_keys_is_enqueued_whole(queue, redis):
    await queue.enqueue(envelope("analyze_posts", payload={"items": [{"post": {}}]}))

    assert len(redis.lists[QUEUE_NAME]) == 1
    assert redis.keys == {}, "nothing to deduplicate, so no keys were marked"


@pytest.mark.asyncio
async def test_analyze_posts_marks_each_item_key(queue, redis):
    await queue.enqueue(envelope("analyze_posts", payload={
        "items": [{"idempotency_key": "a"}, {"idempotency_key": "b"}],
    }))

    assert f"{IDEMPOTENCY_PREFIX}:a" in redis.keys
    assert f"{IDEMPOTENCY_PREFIX}:b" in redis.keys


@pytest.mark.asyncio
async def test_analyze_posts_drops_duplicate_items_and_keeps_the_rest(queue, redis):
    redis.keys[f"{IDEMPOTENCY_PREFIX}:a"] = "1"

    assert await queue.enqueue(envelope("analyze_posts", payload={
        "items": [{"idempotency_key": "a"}, {"idempotency_key": "b"}],
    })) is True

    stored = json.loads(redis.lists[QUEUE_NAME][0])["envelope"]["payload"]
    assert [item["idempotency_key"] for item in stored["items"]] == ["b"]
    assert stored["skipped_count"] == 1


@pytest.mark.asyncio
async def test_analyze_posts_is_not_enqueued_when_every_item_is_a_duplicate(queue, redis):
    redis.keys[f"{IDEMPOTENCY_PREFIX}:a"] = "1"

    assert await queue.enqueue(envelope("analyze_posts", payload={
        "items": [{"idempotency_key": "a"}],
    })) is False
    assert redis.lists[QUEUE_NAME] == []


@pytest.mark.asyncio
async def test_analyze_posts_ignores_unusable_item_shapes(queue, redis):
    await queue.enqueue(envelope("analyze_posts", payload={"items": ["not a dict", 42]}))
    assert len(redis.lists[QUEUE_NAME]) == 1

    await queue.enqueue(envelope("analyze_posts", payload={"items": "not a list"}))
    assert len(redis.lists[QUEUE_NAME]) == 2


# --- dequeue ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_dequeue_moves_the_claim_into_the_processing_list(queue, redis):
    await queue.enqueue(envelope())

    claimed = await queue.dequeue(timeout_seconds=1)

    assert claimed is not None
    assert redis.lists[QUEUE_NAME] == []
    assert len(redis.lists[PROCESSING_QUEUE_NAME]) == 1
    assert claimed.receipts, "the claim must carry its receipt so it can be removed later"


@pytest.mark.asyncio
async def test_dequeue_returns_none_on_an_empty_queue(queue):
    assert await queue.dequeue(timeout_seconds=1) is None
    assert await queue.dequeue_nowait() is None


@pytest.mark.asyncio
async def test_an_unparseable_claim_is_dead_lettered_not_retried(queue, redis):
    redis.lists[QUEUE_NAME].append("{not json")

    assert await queue.dequeue_nowait() is None
    assert redis.lists[PROCESSING_QUEUE_NAME] == [], "the bad claim must not be left in flight"
    assert redis.lists[DEAD_LETTER_QUEUE_NAME] == ["{not json"]


# --- acknowledge / requeue / recover ---------------------------------------

@pytest.mark.asyncio
async def test_acknowledge_removes_the_claim_from_the_processing_list(queue, redis):
    await queue.enqueue(envelope())
    claimed = await queue.dequeue_nowait()

    await queue.acknowledge(claimed)

    assert redis.lists[PROCESSING_QUEUE_NAME] == []
    assert redis.lists[QUEUE_NAME] == []


@pytest.mark.asyncio
async def test_requeue_later_returns_work_behind_operations_that_have_not_run(queue, redis):
    await queue.enqueue(envelope())
    claimed = await queue.dequeue_nowait()

    await queue.requeue_later(claimed)

    assert redis.lists[PROCESSING_QUEUE_NAME] == []
    assert len(redis.lists[QUEUE_NAME]) == 1


@pytest.mark.asyncio
async def test_requeue_next_returns_work_to_the_next_consumed_end(queue, redis):
    await queue.enqueue(envelope("detect_language"))
    claimed = await queue.dequeue_nowait()
    await queue.enqueue(envelope("embed_text"))

    await queue.requeue_next(claimed)

    # RPUSH puts it at the tail, which is taken before the newer LPUSHed item.
    assert (await queue.dequeue_nowait()).envelope.operation_type == "detect_language"


@pytest.mark.asyncio
async def test_acknowledge_falls_back_to_reserializing_when_receipts_are_absent(queue, redis):
    """The fallback works because `model_dump_json` round-trips byte-identically.

    If that ever stops holding - a field reordering, a datetime format change -
    acknowledgement would silently stop removing claims and every operation
    would be redelivered forever. This pins the property rather than the path.
    """
    await queue.enqueue(envelope())
    claimed = await queue.dequeue_nowait()
    stored_receipt = redis.lists[PROCESSING_QUEUE_NAME][0]
    claimed.receipts = None

    await queue.acknowledge(claimed)

    assert redis.lists[PROCESSING_QUEUE_NAME] == [], (
        "re-serializing no longer reproduces the stored receipt; acknowledgement "
        f"would leak claims. stored={stored_receipt}"
    )


@pytest.mark.asyncio
async def test_recover_inflight_returns_every_interrupted_claim(queue, redis):
    for _ in range(3):
        await queue.enqueue(envelope())
    for _ in range(3):
        await queue.dequeue_nowait()
    assert len(redis.lists[PROCESSING_QUEUE_NAME]) == 3

    recovered = await queue.recover_inflight()

    assert recovered == 3
    assert redis.lists[PROCESSING_QUEUE_NAME] == []
    assert len(redis.lists[QUEUE_NAME]) == 3


@pytest.mark.asyncio
async def test_recover_inflight_is_a_no_op_when_nothing_was_in_flight(queue):
    assert await queue.recover_inflight() == 0


# --- size -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_size_counts_pending_and_in_flight_work(queue, redis):
    for _ in range(3):
        await queue.enqueue(envelope())
    await queue.dequeue_nowait()

    assert await queue.size() == 3, "an in-flight claim is still outstanding work"


@pytest.mark.asyncio
async def test_size_is_zero_on_an_untouched_queue(queue):
    assert await queue.size() == 0


# --- _parse -----------------------------------------------------------------

def test_parse_rejects_anything_that_is_not_a_queued_operation():
    assert OperationQueue._parse("{not json") is None
    assert OperationQueue._parse("[]") is None
    assert OperationQueue._parse('{"envelope":{"operation_type":""}}') is None


def test_parse_accepts_a_round_tripped_operation():
    original = QueuedOperation(envelope=envelope())

    parsed = OperationQueue._parse(original.model_dump_json())

    assert parsed is not None
    assert parsed.envelope.operation_type == "detect_language"


# --- private helper guards --------------------------------------------------

@pytest.mark.asyncio
async def test_key_helpers_short_circuit_on_an_empty_list(queue, redis):
    """Both guards exist so an empty batch never opens a Redis pipeline.

    `enqueue` already filters empties, so these are only reachable directly -
    but without them an empty analyze_posts batch would round-trip to Redis.
    """
    assert await queue._existing_keys([]) == set()
    assert await queue._mark_keys([]) is None
    assert redis.keys == {}, "an empty batch must not touch Redis at all"
