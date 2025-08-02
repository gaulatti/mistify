# service/src/models.py

from pydantic import BaseModel
from typing import Optional, List, Dict


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


class ClusteringResponse(BaseModel):
    total_texts: int
    total_groups: int
    groups: List[ClusterGroup]
    processing_time: Optional[float] = None
    debug_info: Optional[Dict] = None
