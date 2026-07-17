import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import grpc
import httpx
from pydantic import ValidationError

from src.endpoints import analysis, classification, clustering, embedding, language, translation
from src.grpc.mistify import operations_pb2, operations_pb2_grpc
from src.models import (
    ClassificationRequest,
    EmbeddingItem,
    LanguageDetectionRequest,
    PostData,
    TranslationRequest,
    UnifiedAnalysisRequest,
)
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
        logger.debug(
            "Processing operation %s (%s)",
            envelope.operation_id,
            envelope.operation_type,
        )

        try:
            result = await self._run_operation(envelope)
        except Exception as exc:
            logger.error(
                "Operation %s failed: %s",
                envelope.operation_id,
                exc,
            )
            await self._deliver_callback_safely(envelope, "failed", error=str(exc))
            return

        await self._deliver_callback_safely(envelope, "succeeded", result=result)

    async def _run_operation(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        if envelope.operation_type == "analyze_posts":
            return await self._analyze_posts(envelope)
        if envelope.operation_type == "detect_language":
            return await self._detect_language(envelope)
        if envelope.operation_type == "classify_content":
            return await self._classify_content(envelope)
        if envelope.operation_type == "translate_text":
            return await self._translate_text(envelope)
        if envelope.operation_type == "embed_text":
            return await self._embed_text(envelope)
        if envelope.operation_type == "cluster_post":
            return await self._cluster_post(envelope)

        raise ValueError(f"Unsupported operation_type: {envelope.operation_type}")

    def _request(self):
        return SimpleNamespace(state=SimpleNamespace(app_state=self.app_state))

    async def _analyze_posts(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        wrappers = self._extract_items(envelope.payload)
        posts = [wrapper["post"] for wrapper in wrappers]

        try:
            analysis_request = UnifiedAnalysisRequest(
                items=posts,
                translate_to_english=self.app_state.config["PROCESSING_TRANSLATE_TO_ENGLISH"],
                classification_labels=envelope.payload.get("classification_labels") or None,
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid analyze payload: {exc}") from exc

        response = await analysis.unified_analysis(analysis_request, self._request())
        merged_posts = self._merge_processed_items(posts, response)

        enriched_items = []
        for wrapper, merged_post in zip(wrappers, merged_posts):
            enriched_items.append(
                {
                    "post": merged_post,
                    "idempotency_key": wrapper.get("idempotency_key"),
                }
            )

        return {"items": enriched_items}

    async def _detect_language(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        response = await language.detect_language(
            LanguageDetectionRequest(**envelope.payload),
            self._request(),
        )
        return response.model_dump()

    async def _classify_content(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        response = await classification.classify_content(
            ClassificationRequest(**envelope.payload),
            self._request(),
        )
        return response.model_dump()

    async def _translate_text(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        response = await translation.translate_text(
            TranslationRequest(**envelope.payload),
            self._request(),
        )
        return response.model_dump()

    async def _embed_text(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        return await embedding.embed_items(
            EmbeddingItem(**envelope.payload),
            self._request(),
        )

    async def _cluster_post(self, envelope: OperationEnvelope) -> Dict[str, Any]:
        response = await clustering.cluster_texts(
            PostData.model_validate(envelope.payload),
            self._request(),
        )
        result = response.model_dump()
        return {
            "items": [{
                "post": result,
                "idempotency_key": envelope.idempotency_key or "",
            }]
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

            if res.embedding is not None:
                base["embeddings"] = res.embedding

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
        if envelope.grpc_callback:
            await self._deliver_grpc_callback(envelope, status, result, error)
            return

        if envelope.callback:
            await self._deliver_http_callback(envelope, status, result, error)
            return

        logger.debug("No callback configured for operation %s", envelope.operation_id)

    async def _deliver_callback_safely(
        self,
        envelope: OperationEnvelope,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        try:
            await self._deliver_callback(envelope, status, result=result, error=error)
        except Exception as exc:
            logger.error(
                "Operation %s %s callback delivery failed (%s): %s",
                envelope.operation_id,
                status,
                self._callback_target(envelope),
                self._format_callback_error(exc),
            )
            logger.debug(
                "Callback delivery traceback for operation %s",
                envelope.operation_id,
                exc_info=True,
            )

    def _callback_target(self, envelope: OperationEnvelope) -> str:
        if envelope.grpc_callback:
            return f"grpc://{envelope.grpc_callback.target}"
        if envelope.callback:
            return envelope.callback.url
        return "none"

    def _format_callback_error(self, exc: Exception) -> str:
        if isinstance(exc, grpc.aio.AioRpcError):
            code = exc.code()
            code_name = getattr(code, "name", str(code))
            return f"{code_name}: {exc.details()}"

        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"

        if isinstance(exc, httpx.RequestError):
            return str(exc)

        return str(exc)

    async def _deliver_http_callback(
        self,
        envelope: OperationEnvelope,
        status: str,
        result: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        payload = {
            "operationId": envelope.operation_id,
            "operationType": envelope.operation_type,
            "status": status,
            "idempotencyKey": envelope.idempotency_key,
            "context": envelope.context.model_dump(),
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

    async def _deliver_grpc_callback(
        self,
        envelope: OperationEnvelope,
        status: str,
        result: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        grpc_callback = envelope.grpc_callback
        items = self._build_analysis_result_items(result)

        request = operations_pb2.AnalysisResultRequest(
            operation_id=envelope.operation_id,
            operation_type=envelope.operation_type,
            status=status,
            items=items,
            error=error or "",
        )

        async with grpc.aio.insecure_channel(grpc_callback.target) as channel:
            stub = operations_pb2_grpc.MonitorIngestStub(channel)
            response = await stub.ReceiveAnalysisResult(
                request,
                timeout=CALLBACK_TIMEOUT_SECONDS,
            )
            if not response.ok:
                raise RuntimeError(f"Monitor rejected callback: ok={response.ok}")

    def _build_analysis_result_items(
        self, result: Optional[Dict[str, Any]]
    ) -> List[operations_pb2.AnalysisResultItem]:
        if not result:
            return []

        items = result.get("items") or []
        pb_items = []
        for item in items:
            if not isinstance(item, dict):
                continue

            post = item.get("post") if isinstance(item.get("post"), dict) else item
            hash_value = item.get("idempotency_key") or post.get("hash") or ""

            from google.protobuf.struct_pb2 import Struct
            payload = Struct()
            if isinstance(post, dict):
                payload.update(post)

            pb_items.append(
                operations_pb2.AnalysisResultItem(
                    hash=hash_value,
                    payload=payload,
                )
            )

        return pb_items
