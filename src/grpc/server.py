import logging
import os

import grpc
from google.protobuf.json_format import MessageToDict

from src.grpc import mistify_operations_pb2, mistify_operations_pb2_grpc
from src.operations.models import HttpCallback, OperationContext, OperationEnvelope

logger = logging.getLogger("mistify")

GRPC_PORT = int(os.getenv("GRPC_PORT", "50000"))


class MistifyOperationsService(mistify_operations_pb2_grpc.MistifyOperationsServicer):
    def __init__(self, operation_queue) -> None:
        self.operation_queue = operation_queue

    async def AnalyzePost(self, request, context):
        envelope = OperationEnvelope(
            operation_type="analyze_post",
            idempotency_key=request.idempotency_key or None,
            payload=MessageToDict(request.post, preserving_proto_field_name=True),
            context=self._context(request.context),
            metadata=MessageToDict(request.metadata, preserving_proto_field_name=True),
            callback=self._callback(request.callback),
        )
        return await self._enqueue(envelope)

    async def AnalyzePosts(self, request, context):
        envelope = OperationEnvelope(
            operation_type="analyze_posts",
            idempotency_key=request.idempotency_key or None,
            payload={
                "items": [
                    MessageToDict(post, preserving_proto_field_name=True)
                    for post in request.posts
                ],
            },
            context=self._context(request.context),
            metadata=MessageToDict(request.metadata, preserving_proto_field_name=True),
            callback=self._callback(request.callback),
        )
        return await self._enqueue(envelope)

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
        )
        return await self._enqueue(envelope)

    def _context(self, context):
        return OperationContext(
            service=context.service,
            tenant=context.tenant,
            request_id=context.request_id,
            trace_id=context.trace_id,
        )

    async def _enqueue(self, envelope):
        queued = await self.operation_queue.enqueue(envelope)
        return mistify_operations_pb2.EnqueueAnalysisResponse(
            operation_id=envelope.operation_id,
            queued=queued,
        )

    def _callback(self, callback):
        if not callback.url:
            return None

        return HttpCallback(
            url=callback.url,
            headers=dict(callback.headers),
        )


async def start_grpc_server(operation_queue):
    server = grpc.aio.server()
    mistify_operations_pb2_grpc.add_MistifyOperationsServicer_to_server(
        MistifyOperationsService(operation_queue),
        server,
    )
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    await server.start()
    logger.info("Mistify gRPC server started on port %d", GRPC_PORT)
    return server
