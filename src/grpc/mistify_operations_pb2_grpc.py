import grpc

from src.grpc import mistify_operations_pb2 as pb


_METHODS = {
    "AnalyzePost": pb.AnalyzePostRequest,
    "AnalyzePosts": pb.AnalyzePostsRequest,
    "DetectLanguage": pb.DetectLanguageRequest,
    "ClassifyContent": pb.ClassifyContentRequest,
    "TranslateText": pb.TranslateTextRequest,
    "EmbedText": pb.EmbedTextRequest,
    "ClusterPost": pb.ClusterPostRequest,
    "GenerateText": pb.GenerateTextRequest,
}


class MistifyOperationsStub:
    def __init__(self, channel):
        for method_name, request_type in _METHODS.items():
            setattr(
                self,
                method_name,
                channel.unary_unary(
                    f"/mistify.operations.MistifyOperations/{method_name}",
                    request_serializer=request_type.SerializeToString,
                    response_deserializer=pb.EnqueueAnalysisResponse.FromString,
                ),
            )


class MistifyOperationsServicer:
    async def AnalyzePost(self, request, context):
        raise NotImplementedError("Method not implemented")

    async def AnalyzePosts(self, request, context):
        raise NotImplementedError("Method not implemented")

    async def DetectLanguage(self, request, context):
        raise NotImplementedError("Method not implemented")

    async def ClassifyContent(self, request, context):
        raise NotImplementedError("Method not implemented")

    async def TranslateText(self, request, context):
        raise NotImplementedError("Method not implemented")

    async def EmbedText(self, request, context):
        raise NotImplementedError("Method not implemented")

    async def ClusterPost(self, request, context):
        raise NotImplementedError("Method not implemented")

    async def GenerateText(self, request, context):
        raise NotImplementedError("Method not implemented")


def add_MistifyOperationsServicer_to_server(servicer, server):
    rpc_method_handlers = {}
    for method_name, request_type in _METHODS.items():
        rpc_method_handlers[method_name] = grpc.unary_unary_rpc_method_handler(
            getattr(servicer, method_name),
            request_deserializer=request_type.FromString,
            response_serializer=pb.EnqueueAnalysisResponse.SerializeToString,
        )

    generic_handler = grpc.method_handlers_generic_handler(
        "mistify.operations.MistifyOperations",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))
