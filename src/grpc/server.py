import logging
import os

import grpc
from google.protobuf.json_format import MessageToDict

from src import metrics
from src.grpc.mistify import operations_pb2, operations_pb2_grpc
from src.helpers.scout_queries import generate_scout_queries, rank_scout_candidates
from src.operations.models import (
    GrpcCallback,
    HttpCallback,
    OperationContext,
    OperationEnvelope,
)

logger = logging.getLogger("mistify")

GRPC_PORT = int(os.getenv("GRPC_PORT", "50000"))


class MistifyOperationsService(operations_pb2_grpc.MistifyOperationsServicer):
    def __init__(self, operation_queue, app_state) -> None:
        self.operation_queue = operation_queue
        self.app_state = app_state

    @metrics.record_rpc("AnalyzePosts")
    async def AnalyzePosts(self, request, context):
        items = [
            {
                "post": MessageToDict(item.post, preserving_proto_field_name=True),
                "idempotency_key": item.idempotency_key or None,
            }
            for item in request.items
        ]
        payload = {"items": items}
        if request.force:
            payload["force"] = True
        if request.classification_labels:
            payload["classification_labels"] = list(request.classification_labels)

        envelope = OperationEnvelope(
            operation_type="analyze_posts",
            payload=payload,
            context=self._context(request.context),
            metadata=MessageToDict(request.metadata, preserving_proto_field_name=True),
            callback=self._callback(request.callback),
            grpc_callback=self._grpc_callback(
                getattr(request, "grpc_callback", None)
            ),
        )
        try:
            queued = await self.operation_queue.enqueue(envelope)
        except Exception:
            metrics.record_queue_event("analyze_posts", "error")
            raise
        metrics.record_queue_event(
            "analyze_posts", "enqueued" if queued else "duplicate"
        )
        logger.info(
            "Operation %s enqueue %s (type=analyze_posts, items=%d)",
            envelope.operation_id,
            "accepted" if queued else "deduplicated",
            len(items),
        )
        return operations_pb2.EnqueueAnalysisResponse(
            operation_id=envelope.operation_id,
            queued=queued,
        )

    @metrics.record_rpc("DetectLanguage")
    async def DetectLanguage(self, request, context):
        return await self._enqueue_request(
            "detect_language",
            request,
            {
                "text": request.text,
                "k": request.k or 1,
            },
        )

    @metrics.record_rpc("ClassifyContent")
    async def ClassifyContent(self, request, context):
        return await self._enqueue_request(
            "classify_content",
            request,
            {
                "text": request.text,
                "labels": list(request.labels),
            },
        )

    @metrics.record_rpc("TranslateText")
    async def TranslateText(self, request, context):
        return await self._enqueue_request(
            "translate_text",
            request,
            {
                "text": request.text,
                "source_language": request.source_language or None,
                "target_language": request.target_language or "eng",
            },
        )

    @metrics.record_rpc("EmbedText")
    async def EmbedText(self, request, context):
        return await self._enqueue_request(
            "embed_text",
            request,
            {
                "content": request.content,
            },
        )

    @metrics.record_rpc("ClusterPost")
    async def ClusterPost(self, request, context):
        return await self._enqueue_request(
            "cluster_post",
            request,
            MessageToDict(request.post, preserving_proto_field_name=True),
        )

    @metrics.record_rpc("GenerateScoutQueries")
    async def GenerateScoutQueries(self, request, context):
        queries, translated_title, source_language = await generate_scout_queries(
            self.app_state,
            request.title,
            request.source_language or None,
            request.max_queries or 2,
        )
        return operations_pb2.ScoutQueryResponse(
            queries=queries,
            translated_title=translated_title,
            source_language=source_language,
        )

    @metrics.record_rpc("RankScoutCandidates")
    async def RankScoutCandidates(self, request, context):
        ranked = await rank_scout_candidates(
            self.app_state,
            request.seed_text,
            [(candidate.id, candidate.title) for candidate in request.candidates],
            request.min_score or 0.55,
            request.max_candidates or 15,
        )
        return operations_pb2.ScoutRankResponse(
            candidates=[
                operations_pb2.RankedScoutCandidate(id=candidate_id, score=score)
                for candidate_id, score in ranked
            ]
        )

    async def _enqueue_request(self, operation_type, request, payload):
        envelope = OperationEnvelope(
            operation_type=operation_type,
            idempotency_key=request.idempotency_key or None,
            payload=payload,
            context=self._context(request.context),
            metadata=MessageToDict(request.metadata, preserving_proto_field_name=True),
            callback=self._callback(request.callback),
            grpc_callback=self._grpc_callback(
                getattr(request, "grpc_callback", None)
            ),
        )
        try:
            queued = await self.operation_queue.enqueue(envelope)
        except Exception:
            metrics.record_queue_event(operation_type, "error")
            raise
        metrics.record_queue_event(
            operation_type, "enqueued" if queued else "duplicate"
        )
        logger.info(
            "Operation %s enqueue %s (type=%s)",
            envelope.operation_id,
            "accepted" if queued else "deduplicated",
            operation_type,
        )
        return operations_pb2.EnqueueAnalysisResponse(
            operation_id=envelope.operation_id,
            queued=queued,
        )

    def _context(self, context):
        return OperationContext(
            service=context.service,
            tenant=context.tenant,
            request_id=context.request_id,
            trace_id=context.trace_id,
        )

    def _callback(self, callback):
        if not callback or not callback.url:
            return None

        return HttpCallback(
            url=callback.url,
            headers=dict(callback.headers),
        )

    def _grpc_callback(self, grpc_callback):
        if not grpc_callback or not grpc_callback.target:
            return None

        return GrpcCallback(
            target=grpc_callback.target,
            service=grpc_callback.service,
            method=grpc_callback.method,
        )


async def start_grpc_server(operation_queue, app_state):
    server = grpc.aio.server()
    operations_pb2_grpc.add_MistifyOperationsServicer_to_server(
        MistifyOperationsService(operation_queue, app_state),
        server,
    )
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    await server.start()
    logger.info("Mistify gRPC server started on port %d", GRPC_PORT)
    return server
