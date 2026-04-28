import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
from pydantic import ValidationError

from src.endpoints import analysis
from src.models import UnifiedAnalysisRequest
from src.operations.models import OperationEnvelope, QueuedOperation
from src.operations.queue import OperationQueue

logger = logging.getLogger("mistify")

IDLE_TIMEOUT_SECONDS = 5
CALLBACK_TIMEOUT_SECONDS = 30.0


class OperationWorker:
    def __init__(self, queue: OperationQueue, app_state: Any) -> None:
        self.queue = queue
        self.app_state = app_state

    async def run_forever(self) -> None:
        logger.info(
            "Mistify operation worker started (queue=%s)",
            self.queue.queue_name,
        )

        while True:
            try:
                queued = await self.queue.dequeue(timeout_seconds=IDLE_TIMEOUT_SECONDS)
                if queued is None:
                    continue

                await self.process(queued)
            except asyncio.CancelledError:
                logger.info("Mistify operation worker cancelled")
                raise
            except Exception as exc:
                logger.error("Mistify operation worker error: %s", exc)
                await asyncio.sleep(1)

    async def process(self, queued: QueuedOperation) -> None:
        envelope = queued.envelope
        logger.info(
            "Processing operation %s (%s)",
            envelope.operation_id,
            envelope.operation_type,
        )

        try:
            result = await self._run_operation(envelope)
            await self._deliver_callback(envelope, "succeeded", result=result)
        except Exception as exc:
            logger.error(
                "Operation %s failed: %s",
                envelope.operation_id,
                exc,
            )
            await self._deliver_callback(envelope, "failed", error=str(exc))

    async def _run_operation(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        if envelope.operation_type in {"analyze_post", "analyze_posts"}:
            return await self._analyze_posts(envelope)

        raise ValueError(f"Unsupported operation_type: {envelope.operation_type}")

    async def _analyze_posts(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        items = self._extract_items(envelope.payload)
        request = SimpleNamespace(state=SimpleNamespace(app_state=self.app_state))

        try:
            analysis_request = UnifiedAnalysisRequest(
                items=items,
                translate_to_english=self.app_state.config["PROCESSING_TRANSLATE_TO_ENGLISH"],
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid analyze payload: {exc}") from exc

        response = await analysis.unified_analysis(analysis_request, request)
        return {
            "items": self._merge_processed_items(items, response),
        }

    def _extract_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(payload.get("items"), list):
            return payload["items"]
        if isinstance(payload.get("input"), list):
            return payload["input"]
        if isinstance(payload.get("input"), dict):
            return [payload["input"]]
        return [payload]

    def _merge_processed_items(self, raw_items: List[Dict[str, Any]], analysis_resp: Any) -> List[Dict[str, Any]]:
        processed = []
        results = analysis_resp.results if analysis_resp else []

        for idx, raw in enumerate(raw_items):
            base = dict(raw)
            if idx >= len(results):
                processed.append(base)
                continue

            res = results[idx]
            base["content"] = res.content

            if res.language_detection:
                lang_dump = res.language_detection.model_dump()
                base["language_detection"] = lang_dump
                langs = lang_dump.get("languages") or []
                if langs and not base.get("lang"):
                    base["lang"] = langs[0]

            if res.content_classification:
                cls_dump = res.content_classification.model_dump()
                base["content_classification"] = cls_dump
                labels = (cls_dump.get("full_result") or {}).get("labels") or []
                if labels:
                    base["classification_labels"] = labels

            if res.translation:
                base["translation"] = res.translation.model_dump()

            if res.newsworthiness is not None:
                base["newsworthiness"] = res.newsworthiness

            if res.urgency is not None:
                base["urgency"] = res.urgency

            if res.timings:
                base["analysis_timings"] = res.timings.model_dump()

            processed.append(base)

        return processed

    async def _deliver_callback(
        self,
        envelope: OperationEnvelope,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if envelope.callback is None:
            return

        payload = {
            "operationId": envelope.operation_id,
            "operationType": envelope.operation_type,
            "status": status,
            "idempotencyKey": envelope.idempotency_key,
            "metadata": envelope.metadata,
            "result": result,
            "error": error,
        }

        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_SECONDS) as client:
            response = await client.post(
                envelope.callback.url,
                headers=envelope.callback.headers,
                json=payload,
            )
            response.raise_for_status()
