# service/src/models.py

import json

from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional, List, Dict, Union


class LanguageDetectionRequest(BaseModel):
    text: str
    k: int = 1


class LanguageDetectionResponse(BaseModel):
    languages: List[str]
    probabilities: List[float]


class ClassificationRequest(BaseModel):
    text: str
    labels: Optional[List[str]] = None


class ClassificationResponse(BaseModel):
    label: str
    score: float
    full_result: dict


class TranslationRequest(BaseModel):
    text: str
    source_language: Optional[str] = None
    target_language: str = "eng"


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: Optional[str] = None
    target_language: str
    confidence_score: Optional[float] = None


class UnifiedAnalysisItemTimings(BaseModel):
    total_ms: float
    language_detection_ms: Optional[float] = None
    translation_ms: Optional[float] = None
    content_classification_ms: Optional[float] = None
    newsworthiness_ms: Optional[float] = None
    urgency_ms: Optional[float] = None


class UnifiedAnalysisTimings(BaseModel):
    total_ms: float
    item_count: int
    avg_item_ms: float
    language_detection_ms: float = 0.0
    translation_ms: float = 0.0
    content_classification_ms: float = 0.0
    newsworthiness_ms: float = 0.0
    urgency_ms: float = 0.0


class MediaItem(BaseModel):
    url: str
    type: Optional[str] = None  # e.g., image, video


def _normalize_media_to_urls(value: Any) -> Any:
    """Normalize media payload entries to URL strings.

    Accepts:
      - ["https://..."]
      - [{"url": "https://...", "type": "image/png"}]
      - [MediaItem(url="https://...", type="image/png")]
    """
    if value is None:
        return value

    if not isinstance(value, list):
        return value

    normalized: List[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                normalized.append(item.strip())
            continue

        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url.strip():
                normalized.append(url.strip())
            continue

        url_attr = getattr(item, "url", None)
        if isinstance(url_attr, str) and url_attr.strip():
            normalized.append(url_attr.strip())

    return normalized


def _normalize_string_list(value: Any) -> Any:
    """Keep only useful string labels from noisy upstream arrays."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    if not isinstance(value, list):
        return []

    normalized: List[str] = []
    for item in value:
        if isinstance(item, str):
            label = item.strip()
            if label:
                normalized.append(label)
        elif isinstance(item, dict):
            label = item.get("name") or item.get("label") or item.get("slug") or item.get("_")
            if isinstance(label, str) and label.strip():
                normalized.append(label.strip())

    return normalized


def _normalize_id(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, dict):
        for key in ("id", "guid", "uri", "url", "link", "_"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    return str(value)


class UnifiedAnalysisItemRequest(BaseModel):
    id: str
    source: str
    uri: str
    content: str
    createdAt: str
    relevance: Optional[int] = None
    lang: Optional[Union[str, Dict]] = None
    author: Optional[Dict] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    # Accept mixed payloads, normalize to URL strings.
    media: Optional[List[str]] = Field(default_factory=list)
    linkPreview: Optional[str] = None
    score: Optional[float] = None
    scores: Optional[List] = Field(default_factory=list)
    categories: Optional[List[str]] = Field(default_factory=list)
    labels: Optional[List[str]] = Field(default_factory=list)
    # External pipelines may send precomputed embeddings/classification data.
    embeddings: Optional[List[float]] = None
    classification_labels: Optional[List[str]] = Field(default_factory=list)
    hash: str
    
    class Config:
        extra = "allow"

    @field_validator("media", mode="before")
    @classmethod
    def normalize_media(cls, value: Any) -> Any:
        return _normalize_media_to_urls(value)

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: Any) -> Any:
        return _normalize_id(value)

    @field_validator("categories", "labels", "classification_labels", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: Any) -> Any:
        return _normalize_string_list(value)


class UnifiedAnalysisRequest(BaseModel):
    items: List[UnifiedAnalysisItemRequest]
    detect_language: bool = True
    classify_content: bool = True
    translate_to_english: bool = True
    include_timings: bool = False
    language_count: int = 1
    classification_labels: Optional[List[str]] = None


class UnifiedAnalysisItemResponse(BaseModel):
    id: str
    content: str
    hash: str
    language_detection: Optional[LanguageDetectionResponse] = None
    content_classification: Optional[ClassificationResponse] = None
    translation: Optional[TranslationResponse] = None
    newsworthiness: Optional[float] = None
    urgency: Optional[float] = None
    embedding: Optional[List[float]] = None
    timings: Optional[UnifiedAnalysisItemTimings] = None


class UnifiedAnalysisResponse(BaseModel):
    results: List[UnifiedAnalysisItemResponse]
    timings: Optional[UnifiedAnalysisTimings] = None


class EmbeddingItem(BaseModel):
    content: str

    class Config:
        extra = "allow"


class CategoryRelation(BaseModel):
    id: int
    slug: str
    name: str
    Tagging: Dict


class PostData(BaseModel):
    # Required core fields for clustering & identification
    id: int
    content: str
    source: str
    createdAt: str
    hash: str
    uuid: str

    # Fields that may be absent in nested similarPosts payloads
    score: Optional[float] = None
    embeddings: Optional[List[float]] = None  # Raw embedding may be omitted; service re-embeds content
    source_id: Optional[str] = None
    uri: Optional[str] = None
    relevance: Optional[int] = None
    posted_at: Optional[str] = None
    received_at: Optional[str] = None

    # Metadata (all optional)
    lang: Optional[str] = None
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    author_handle: Optional[str] = None
    author_avatar: Optional[str] = None
    # Accept mixed payloads, normalize to URL strings.
    media: Optional[List[str]] = None
    linkPreview: Optional[str] = None
    original: Optional[str] = None
    author: Optional[str] = None
    embedding: Optional[List[float]] = None
    categories_relation: Optional[List[CategoryRelation]] = None

    # Nested similar posts (avoid mutable default list)
    similarPosts: Optional[List['PostData']] = Field(default_factory=list)

    @field_validator("media", mode="before")
    @classmethod
    def normalize_media(cls, value: Any) -> Any:
        return _normalize_media_to_urls(value)


class PostClusteringRequest(BaseModel):
    """Request model for clustering a single post"""
    post: PostData
    similarity_entity: Optional[float] = 0.40
    similarity_global: Optional[float] = 0.60
    big_community_size: Optional[int] = 30
    avg_similarity_min: Optional[float] = 0.50
    topic_labels: Optional[List[str]] = ["economy", "politics", "sports", "conflict", "misc"]
    debug: Optional[bool] = False


class ClusteringRequest(BaseModel):
    texts: List[str]
    similarity_entity: Optional[float] = 0.40
    similarity_global: Optional[float] = 0.60
    big_community_size: Optional[int] = 30
    avg_similarity_min: Optional[float] = 0.50
    topic_labels: Optional[List[str]] = ["economy", "politics", "sports", "conflict", "misc"]
    debug: Optional[bool] = False


class ClusterGroup(BaseModel):
    group_id: int
    texts: List[str]
    indices: List[int]
    size: int
    primary_topic: Optional[str] = None
    primary_entities: Optional[List[str]] = None
    avg_similarity: Optional[float] = None


class ClusteredPost(BaseModel):
    id: int
    hash: str
    content: str


class PostClusterGroup(BaseModel):
    group_id: int
    posts: List[ClusteredPost]  # Array of posts in the cluster (main + similar posts)
    size: int
    primary_topic: Optional[str] = None
    primary_entities: Optional[List[str]] = None
    avg_similarity: Optional[float] = None


class ClusteringResponse(BaseModel):
    total_texts: int
    total_groups: int
    groups: List[ClusterGroup]
    processing_time: Optional[float] = None
    debug_info: Optional[Dict] = None


class PostClusteringResponse(BaseModel):
    total_posts: int = 1  # Always 1 for single post
    total_groups: int = 1  # Always 1 for single post
    group: PostClusterGroup  # Single group instead of list
    processing_time: Optional[float] = None
    debug_info: Optional[Dict] = None
