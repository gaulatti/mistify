import logging
import os

import grpc
from google.protobuf.json_format import MessageToDict

from src.grpc.mistify import operations_pb2, operations_pb2_grpc
from src.operations.models import GrpcCallback, HttpCallback, OperationContext, OperationEnvelope

logger = logging.getLogger("mistify")

GRPC_PORT = int(os.getenv("GRPC_PORT", "50000"))


class MistifyOperationsService(operations_pb2_grpc.MistifyOperationsServicer):
    def __init__(self, operation_queue) -> None:
        self.operation_queue = operation_queue

    async def AnalyzePosts(self, request, context):
        items = [
            {
                "post": MessageToDict(item.post, preserving_proto_field_name=True),
                "idempotency_key": item.idempotency_key or None,
            }
            for item in request.items
        ]
        payload = {"items": items}
        if request.classification_labels:
            payload["classification_labels"] = list(request.classification_labels)

        envelope = OperationEnvelope(
            operation_type="analyze_posts",
            payload=payload,
            context=self._context(request.context),
            metadata=MessageToDict(request.metadata, preserving_proto_field_name=True),
            callback=self._callback(request.callback),
            grpc_callback=self._grpc_callback(request.grpc_callback),
        )
        queued = await self.operation_queue.enqueue(envelope)
        return operations_pb2.EnqueueAnalysisResponse(
            operation_id=envelope.operation_id,
            queued=queued,
        )

    async def DetectLanguage(self, request, context):
        return await self._enqueue_request(
            "detect_language",
            request,
            {
                "text": request.text,
                "k": request.k or 1,
            },
        )

    async def ClassifyContent(self, request, context):
        return await self._enqueue_request(
            "classify_content",
            request,
            {
                "text": request.text,
                "labels": list(request.labels),
            },
        )

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

    async def EmbedText(self, request, context):
        return await self._enqueue_request(
            "embed_text",
            request,
            {
                "content": request.content,
            },
        )

    async def ClusterPost(self, request, context):
        return await self._enqueue_request(
            "cluster_post",
            request,
            MessageToDict(request.post, preserving_proto_field_name=True),
        )

    async def _enqueue_request(self, operation_type, request, payload):
        envelope = OperationEnvelope(
            operation_type=operation_type,
            idempotency_key=request.idempotency_key or None,
            payload=payload,
            context=self._context(request.context),
            metadata=MessageToDict(request.metadata, preserving_proto_field_name=True),
            callback=self._callback(request.callback),
            grpc_callback=self._grpc_callback(request.grpc_callback),
        )
        queued = await self.operation_queue.enqueue(envelope)
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


async def start_grpc_server(operation_queue):
    server = grpc.aio.server()
    operations_pb2_grpc.add_MistifyOperationsServicer_to_server(
        MistifyOperationsService(operation_queue),
        server,
    )
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    await server.start()
    logger.info("Mistify gRPC server started on port %d", GRPC_PORT)
    return server
