# service/src/models.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union


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


class UnifiedAnalysisRequest(BaseModel):
    text: str
    detect_language: bool = True
    classify_content: bool = True
    translate_to_english: bool = False
    language_count: int = 1
    classification_labels: Optional[List[str]] = None


class UnifiedAnalysisResponse(BaseModel):
    text: str
    language_detection: Optional[LanguageDetectionResponse] = None
    content_classification: Optional[ClassificationResponse] = None
    translation: Optional[TranslationResponse] = None


class EmbeddingItem(BaseModel):
    content: str

    class Config:
        extra = "allow"


class CategoryRelation(BaseModel):
    id: int
    slug: str
    name: str
    Tagging: Dict


class MediaItem(BaseModel):
    url: str
    type: Optional[str] = None  # e.g., image, video


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
    # Accept either a plain URL string or an object with url/type
    media: Optional[List[Union[str, MediaItem]]] = None
    linkPreview: Optional[str] = None
    original: Optional[str] = None
    author: Optional[str] = None
    embedding: Optional[List[float]] = None
    categories_relation: Optional[List[CategoryRelation]] = None

    # Nested similar posts (avoid mutable default list)
    similarPosts: Optional[List['PostData']] = Field(default_factory=list)


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
