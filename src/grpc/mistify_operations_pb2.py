from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2

_sym_db = _symbol_database.Default()


def _field(message, name, number, label, field_type, type_name=None):
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name:
        field.type_name = type_name


def _string(message, name, number):
    _field(message, name, number, 1, 9)


def _int32(message, name, number):
    _field(message, name, number, 1, 5)


def _double(message, name, number):
    _field(message, name, number, 1, 1)


def _bool(message, name, number):
    _field(message, name, number, 1, 8)


def _struct(message, name, number):
    _field(message, name, number, 1, 11, ".google.protobuf.Struct")


def _callback(message, number):
    _field(message, "callback", number, 1, 11, ".mistify.operations.HttpCallback")


def _common(message, start_number):
    _string(message, "idempotency_key", start_number)
    _field(message, "context", start_number + 1, 1, 11, ".mistify.operations.OperationContext")
    _struct(message, "metadata", start_number + 2)
    _callback(message, start_number + 3)


file_proto = _descriptor_pb2.FileDescriptorProto()
file_proto.name = "proto/mistify/operations.proto"
file_proto.package = "mistify.operations"
file_proto.syntax = "proto3"
file_proto.dependency.append("google/protobuf/struct.proto")

http_callback = file_proto.message_type.add()
http_callback.name = "HttpCallback"
_string(http_callback, "url", 1)

headers_entry = http_callback.nested_type.add()
headers_entry.name = "HeadersEntry"
headers_entry.options.map_entry = True
_string(headers_entry, "key", 1)
_string(headers_entry, "value", 2)
_field(http_callback, "headers", 2, 3, 11, ".mistify.operations.HttpCallback.HeadersEntry")

operation_context = file_proto.message_type.add()
operation_context.name = "OperationContext"
_string(operation_context, "service", 1)
_string(operation_context, "tenant", 2)
_string(operation_context, "request_id", 3)
_string(operation_context, "trace_id", 4)

analyze_post_request = file_proto.message_type.add()
analyze_post_request.name = "AnalyzePostRequest"
_struct(analyze_post_request, "post", 1)
_common(analyze_post_request, 2)

analyze_posts_request = file_proto.message_type.add()
analyze_posts_request.name = "AnalyzePostsRequest"
_field(analyze_posts_request, "posts", 1, 3, 11, ".google.protobuf.Struct")
_common(analyze_posts_request, 2)

detect_language_request = file_proto.message_type.add()
detect_language_request.name = "DetectLanguageRequest"
_string(detect_language_request, "text", 1)
_int32(detect_language_request, "k", 2)
_common(detect_language_request, 3)

classify_content_request = file_proto.message_type.add()
classify_content_request.name = "ClassifyContentRequest"
_string(classify_content_request, "text", 1)
_field(classify_content_request, "labels", 2, 3, 9)
_common(classify_content_request, 3)

translate_text_request = file_proto.message_type.add()
translate_text_request.name = "TranslateTextRequest"
_string(translate_text_request, "text", 1)
_string(translate_text_request, "source_language", 2)
_string(translate_text_request, "target_language", 3)
_common(translate_text_request, 4)

embed_text_request = file_proto.message_type.add()
embed_text_request.name = "EmbedTextRequest"
_string(embed_text_request, "content", 1)
_common(embed_text_request, 2)

cluster_post_request = file_proto.message_type.add()
cluster_post_request.name = "ClusterPostRequest"
_struct(cluster_post_request, "post", 1)
_common(cluster_post_request, 2)

enqueue_analysis_response = file_proto.message_type.add()
enqueue_analysis_response.name = "EnqueueAnalysisResponse"
_string(enqueue_analysis_response, "operation_id", 1)
_bool(enqueue_analysis_response, "queued", 2)

service = file_proto.service.add()
service.name = "MistifyOperations"

for name, request_type in (
    ("AnalyzePost", "AnalyzePostRequest"),
    ("AnalyzePosts", "AnalyzePostsRequest"),
    ("DetectLanguage", "DetectLanguageRequest"),
    ("ClassifyContent", "ClassifyContentRequest"),
    ("TranslateText", "TranslateTextRequest"),
    ("EmbedText", "EmbedTextRequest"),
    ("ClusterPost", "ClusterPostRequest"),
):
    method = service.method.add()
    method.name = name
    method.input_type = f".mistify.operations.{request_type}"
    method.output_type = ".mistify.operations.EnqueueAnalysisResponse"

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(file_proto.SerializeToString())

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "src.grpc.mistify_operations_pb2", globals())

if _descriptor._USE_C_DESCRIPTORS is False:
    DESCRIPTOR._options = None
    _HTTPCALLBACK_HEADERSENTRY._options = None
    _HTTPCALLBACK_HEADERSENTRY._serialized_options = b"8\001"
