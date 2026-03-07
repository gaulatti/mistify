# service/src/models.py

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


class TextGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 600          # Upper bound (service will clamp to <=800)
    min_new_tokens: int = 120          # Encourage substantive output
    num_beams: int = 4                 # Beam search for better coverage
    temperature: float = 0.7           # Only used if do_sample=True
    do_sample: bool = False            # Default deterministic
    no_repeat_ngram_size: int = 4      # Reduce repetition
    length_penalty: float = 1.0        # Can adjust brevity/verbosity


class TextGenerationResponse(BaseModel):
    prompt: str
    generated_text: str
