# server.py — Unified Text Analysis API
# =====================================
# Language detection, content classification, translation, text clustering,
# and batch sentence embeddings generation in one service.

import os

# ---- HARD CAP hidden thread pools for PyTorch compatibility -------------------
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "RAYON_NUM_THREADS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    "CUDA_LAUNCH_BLOCKING": "0",
})

# ---- Imports -------------------------------------------------------------------
import asyncio
import logging
import psutil
import torch
import warnings
import re
import itertools
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Set, Dict, Any

import fasttext
import networkx as nx
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from networkx.algorithms.community import louvain_communities

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", message="transformers.deepspeed module is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.deepspeed")
warnings.filterwarnings("ignore", message="You must either specify a `tgt_lang`")
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*and `max_length`.*seem to have been set")

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ---- Logging Setup -------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified-text-analysis")

# ---- PyTorch Configuration -----------------------------------------------------
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---- FastAPI App ---------------------------------------------------------------
app = FastAPI(
    title="Unified Text Analysis API",
    description="Language detection, content classification, translation, sentence embeddings, and entity-aware clustering",
    version="1.3.0"
)

# ---- Configuration -------------------------------------------------------------
# Language Detection
FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "lid.176.bin")
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Content Classification
DEFAULT_CLASSIFICATION_LABELS = [
    "breaking news",
    "newsworthy factual",
    "humor / satire",
    "non-newsworthy rant",
]
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))
MIN_MARGIN = float(os.getenv("MIN_MARGIN", "0.10"))
POOL_WORKERS = int(os.getenv("POOL_WORKERS", "4"))
TIMEOUT = int(os.getenv("TIMEOUT", "10"))  # seconds

# Clustering Configuration
CLUSTERING_SIM_ENTITY = 0.40  # Similarity threshold for entities
CLUSTERING_SIM_GLOBAL = 0.60  # Global similarity threshold
CLUSTERING_BIG_COMM = 30      # Max size before splitting communities
CLUSTERING_AVG_SIM_MIN = 0.50 # Minimum average similarity for large communities
CLUSTERING_TOPIC_LABELS = ["economy", "politics", "sports", "conflict", "misc"]
CLUSTERING_ALIAS_THR = 0.20   # Alias discovery threshold (cosine distance)
CLUSTERING_TOK_REMOVE = {"the", "a", "an", "of"}
CLUSTERING_PUN_RE = re.compile(r"[^\w\s]")

# Persistent cache for HuggingFace models - use HF_HOME environment variable if set
HF_CACHE = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".hf_models"))
HF_CACHE.mkdir(parents=True, exist_ok=True)

# ---- Global Variables ----------------------------------------------------------
fasttext_model = None
classifier = None
translator = None
embedder = None  # For clustering
nlp = None      # SpaCy model for clustering
classification_lock = asyncio.Lock()
translation_lock = asyncio.Lock()
clustering_lock = asyncio.Lock()
thread_pool = ThreadPoolExecutor(max_workers=POOL_WORKERS)

# ---- Model Initialization ------------------------------------------------------
def download_fasttext_model(model_path: str, model_url: str) -> bool:
    """Download FastText language detection model if not present"""
    try:
        import urllib.request
        logger.info("🌍 Downloading FastText language detection model...")
        urllib.request.urlretrieve(model_url, model_path)
        logger.info("✓ FastText model downloaded successfully")
        return True
    except Exception as e:
        logger.error("❌ Failed to download FastText model: %s", e)
        return False

def initialize_models():
    global fasttext_model, classifier, translator, embedder, nlp

    logger.info("🔧 Initializing models...")

    # Initialize FastText for language detection
    try:
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            logger.info("FastText model not found at %s, downloading...", FASTTEXT_MODEL_PATH)
            if not download_fasttext_model(FASTTEXT_MODEL_PATH, FASTTEXT_MODEL_URL):
                logger.error("❌ Failed to download FastText model")
                fasttext_model = None
                return
        
        logger.info("🔧 Loading FastText model from: %s", FASTTEXT_MODEL_PATH)
        fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        
        # Test the model with a simple prediction
        try:
            test_labels, test_probs = fasttext_model.predict("Hello world", k=1)
            logger.info("✓ FastText language detection model loaded and tested successfully")
        except Exception as test_e:
            logger.error("❌ FastText model loaded but failed test prediction: %s", test_e)
            fasttext_model = None
    except Exception as e:
        logger.error("❌ Failed to load FastText model: %s", e)
        fasttext_model = None
    
    # Initialize BART classifier for content classification
    try:
        # Universal device detection: CUDA (Linux/Windows) > MPS (Mac) > CPU
        if torch.cuda.is_available():
            device = "cuda"
            device_id = 0
            logger.info("🔧 Using device: CUDA GPU")
            # Clear CUDA cache before loading models
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            device = "mps"
            device_id = 0
            logger.info("🔧 Using device: MPS (Apple Silicon GPU)")
        else:
            device = "cpu"
            device_id = -1
            logger.info("🔧 Using device: CPU")

        try:
            classifier = pipeline(
                "zero-shot-classification",
                model="valhalla/distilbart-mnli-12-3",
                device=device_id,
                hypothesis_template="This post is {}.",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                model_kwargs={"low_cpu_mem_usage": True} if device == "cuda" else {},
                cache_dir=str(HF_CACHE)
            )
            logger.info("✓ BART classification model loaded successfully")
        except Exception as e:
            logger.error("❌ Failed to load BART classification model: %s", e)
            classifier = None
    except Exception as e:
        logger.error("❌ Failed to load BART model: %s", e)

    # Initialize Seamless M4T v2 for translation
    try:
        logger.info("🔧 Loading Seamless M4T v2 translation model...")
        # Use accelerate for automatic device placement (don't specify device manually)
        translator = pipeline(
            "translation",
            model="facebook/seamless-m4t-v2-large",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            model_kwargs={
                "low_cpu_mem_usage": True,
                "device_map": "auto"  # Let accelerate handle device placement
            },
            cache_dir=str(HF_CACHE)
        )
        logger.info("✓ Seamless M4T v2 translation model loaded successfully")
    except Exception as e:
        logger.error("❌ Failed to load Seamless M4T v2 model: %s", e)
        logger.warning("Translation functionality will use fallback model")
        # Fallback: try using a simpler multilingual model
        try:
            logger.info("🔧 Trying fallback translation model...")
            translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-mul-en",
                device=device_id,
                cache_dir=str(HF_CACHE)
            )
            logger.info("✓ Fallback translation model (Helsinki-NLP) loaded successfully")
        except Exception as fallback_e:
            logger.error("❌ Fallback translation model also failed: %s", fallback_e)
            translator = None

    # Initialize clustering models
    try:
        logger.info("🔧 Loading clustering models...")
        
        # Initialize sentence transformer for embeddings
        try:
            torch_device = torch.device(device)
            embedder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device=torch_device,
                cache_folder=str(HF_CACHE)
            )
            logger.info("✓ Sentence transformer embedder loaded successfully")
        except Exception as e:
            logger.error("❌ Failed to load sentence transformer: %s", e)
            embedder = None
        
        # Initialize SpaCy model for entity extraction
        try:
            nlp = spacy.load("en_core_web_sm", disable=("tagger", "parser", "lemmatizer"))
            logger.info("✓ SpaCy model loaded for entity extraction")
        except Exception as e:
            logger.error("❌ Failed to load SpaCy model: %s", e)
            logger.info("Please install SpaCy English model: python -m spacy download en_core_web_sm")
            nlp = None
            
    except Exception as e:
        logger.error("❌ Failed to initialize clustering models: %s", e)

# Initialize models on startup
initialize_models()

# ---- Pydantic Models -----------------------------------------------------------
class LanguageDetectionRequest(BaseModel):
    text: str
    k: int = 1  # number of languages to return

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
    source_language: Optional[str] = None  # Auto-detect if not provided
    target_language: str = "eng"  # Default to English

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

# Clustering I/O
class EmbeddingItem(BaseModel):
    content: str
    # Allow any additional fields
    class Config:
        extra = "allow"

class ClusteringRequest(BaseModel):
    texts: List[str]
    similarity_entity: Optional[float] = CLUSTERING_SIM_ENTITY
    similarity_global: Optional[float] = CLUSTERING_SIM_GLOBAL
    big_community_size: Optional[int] = CLUSTERING_BIG_COMM
    avg_similarity_min: Optional[float] = CLUSTERING_AVG_SIM_MIN
    topic_labels: Optional[List[str]] = CLUSTERING_TOPIC_LABELS
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

# ---- Helper Functions ----------------------------------------------------------
def scrub_text(s: str) -> str:
    """Clean text for alias discovery"""
    s = CLUSTERING_PUN_RE.sub("", s.lower()).strip()
    return " ".join(t for t in s.split() if t not in CLUSTERING_TOK_REMOVE)

def grouper(seq, n):
    """Group sequence into chunks of size n"""
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def discover_aliases(ents_per_doc: List[Set[str]]) -> Dict[str, str]:
    """Discover entity aliases using cosine similarity clustering"""
    if not embedder:
        return {}
        
    raw = sorted({e for ents in ents_per_doc for e in ents if e != "__NOENT__"})
    if not raw:
        return {}
    
    clean = [scrub_text(r) for r in raw]
    embs = embedder.encode(
        clean, convert_to_tensor=True, normalize_embeddings=True
    ).cpu().numpy()
    
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=CLUSTERING_ALIAS_THR
    )
    labels = model.fit_predict(embs)

    canons, by_label = {}, {}
    for r, c, l in zip(raw, clean, labels):
        by_label.setdefault(l, []).append((len(c), r))
    for items in by_label.values():
        items.sort()
        canon = items[0][1]
        for _, raw_txt in items:
            canons[raw_txt] = canon
    return canons

def get_entities(docs) -> List[Set[str]]:
    """Extract and normalize entities from spaCy documents"""
    sets = []
    for d in docs:
        ents = {e.text.strip() for e in d.ents
                if e.label_ in {"GPE", "LOC", "ORG", "FAC", "PERSON"}}
        sets.append(ents if ents else {"__NOENT__"})
    
    alias = discover_aliases(sets)
    return [{alias.get(e, e) for e in s} for s in sets]

def get_primary_topics(texts: List[str], show_bar: bool = False) -> List[str]:
    """Get primary topic classification for texts"""
    if not classifier:
        return ["misc"] * len(texts)
        
    prim = []
    for chunk in tqdm(
        list(grouper(texts, 16)),
        desc="classify", disable=not show_bar
    ):
        try:
            res = classifier(chunk, CLUSTERING_TOPIC_LABELS, multi_label=True)
            prim.extend(r["labels"][0] for r in res)
        except Exception as e:
            logger.warning("Topic classification failed for chunk: %s", e)
            prim.extend(["misc"] * len(chunk))
    return prim

def build_clustering_graph(texts: List[str], show_bar: bool = False, config: Dict = None):
    """Build similarity graph for clustering"""
    if not nlp or not embedder:
        raise HTTPException(status_code=503, detail="Clustering models not available")
    
    # Use config values or defaults
    sim_entity = config.get("similarity_entity", CLUSTERING_SIM_ENTITY) if config else CLUSTERING_SIM_ENTITY
    sim_global = config.get("similarity_global", CLUSTERING_SIM_GLOBAL) if config else CLUSTERING_SIM_GLOBAL
    
    docs = list(tqdm(nlp.pipe(texts, batch_size=64),
                     total=len(texts), desc="spaCy",
                     disable=not show_bar))
    entities = get_entities(docs)
    topics = get_primary_topics(texts, show_bar)

    emb = embedder.encode(texts, batch_size=64, convert_to_tensor=True,
                         normalize_embeddings=True)
    sims = util.cos_sim(emb, emb).cpu().numpy()

    G, why = nx.Graph(), {}
    for i, j in itertools.combinations(range(len(texts)), 2):
        if topics[i] != topics[j] or topics[i] == "misc":
            continue
        shared = bool(entities[i] & entities[j])
        sim = sims[i, j]
        ok = False
        if shared and sim >= sim_entity:
            ok = True
            why[(i, j)] = f"ENT & {sim:.2f}"
        elif sim >= sim_global:
            ok = True
            why[(i, j)] = f"{sim:.2f}"
        if ok:
            G.add_edge(i, j, weight=float(sim))
    return G, sims, why, topics, entities

def split_large_communities(comm, sims, config: Dict = None, depth: int = 0):
    """Recursively split large communities"""
    big_comm = config.get("big_community_size", CLUSTERING_BIG_COMM) if config else CLUSTERING_BIG_COMM
    avg_sim_min = config.get("avg_similarity_min", CLUSTERING_AVG_SIM_MIN) if config else CLUSTERING_AVG_SIM_MIN
    sim_global = config.get("similarity_global", CLUSTERING_SIM_GLOBAL) if config else CLUSTERING_SIM_GLOBAL
    
    if len(comm) <= big_comm:
        return [comm]
    subM = sims[np.ix_(comm, comm)]
    avg = (subM.sum() - len(comm)) / (len(comm) * (len(comm) - 1))
    if avg >= avg_sim_min or depth > 2:
        return [comm]
    tight = nx.Graph()
    tight.add_nodes_from(range(len(comm)))
    for a, b in itertools.combinations(range(len(comm)), 2):
        if subM[a, b] >= sim_global + 0.05:
            tight.add_edge(a, b)
    sub = louvain_communities(tight, weight=None, resolution=1.1)
    out = []
    for c in sub:
        mapped = [comm[i] for i in c]
        out.extend(split_large_communities(mapped, sims, config, depth + 1))
    return out

def _cluster_sync(texts: List[str], config: Dict = None, debug: bool = False):
    """Synchronous clustering function for thread execution"""
    try:
        # Clear GPU cache before clustering
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        G, sims, why, topics, entities = build_clustering_graph(texts, show_bar=debug, config=config)
        comms = louvain_communities(G, weight=None, resolution=1.0)
        groups = []
        for c in comms:
            groups.extend(split_large_communities(list(c), sims, config))
        groups.sort(key=lambda g: (-len(g), min(g)))

        # Clear cache after clustering
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create detailed cluster information
        cluster_groups = []
        for gid, idx in enumerate(groups):
            group_texts = [texts[i] for i in sorted(idx)]
            
            # Get primary topic for this group
            group_topics = [topics[i] for i in idx]
            primary_topic = max(set(group_topics), key=group_topics.count) if group_topics else None
            
            # Get primary entities for this group
            group_entities = set()
            for i in idx:
                group_entities.update(entities[i])
            group_entities.discard("__NOENT__")
            primary_entities = list(group_entities)[:5]  # Top 5 entities
            
            # Calculate average similarity within group
            if len(idx) > 1:
                group_sims = []
                for i, j in itertools.combinations(idx, 2):
                    group_sims.append(sims[i, j])
                avg_similarity = float(np.mean(group_sims)) if group_sims else 0.0
            else:
                avg_similarity = 1.0
            
            cluster_groups.append(ClusterGroup(
                group_id=gid,
                texts=group_texts,
                indices=sorted(idx),
                size=len(idx),
                primary_topic=primary_topic,
                primary_entities=primary_entities,
                avg_similarity=avg_similarity
            ))

        result = {
            "groups": cluster_groups,
            "debug_info": {
                "edge_reasons": {f"{i}-{j}": reason for (i, j), reason in why.items()},
                "total_edges": len(why),
                "topics": topics,
                "entities_count": sum(len(e) for e in entities)
            } if debug else None
        }

        return result
    except Exception as e:
        logger.error("❌ Clustering error: %s", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

def _classify_sync(text: str, labels: List[str]):
    """Synchronous classification function for thread execution"""
    try:
        # Clear GPU cache before classification to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Limit text length to reduce memory usage
        max_text_length = 512
        if len(text) > max_text_length:
            text = text[:max_text_length]

        result = classifier(text, candidate_labels=labels)

        # Clear cache after classification
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
            logger.warning("GPU out of memory for classification, clearing cache and retrying")
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Try with shorter text
            short_text = text[:256] if len(text) > 256 else text
            try:
                return classifier(short_text, candidate_labels=labels)
            except RuntimeError:
                logger.error("Still out of memory even with shorter text")
                raise e
        else:
            raise e

def _translate_sync(text: str, source_lang: Optional[str] = None, target_lang: str = "eng"):
    """Synchronous translation function for thread execution"""
    try:
        # Clear GPU cache before translation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log the input for debugging
        logger.info("Translation input - Text length: %d, Source: %s, Target: %s", 
                   len(text), source_lang, target_lang)
        logger.info("Translation input text: %s", text[:200] + "..." if len(text) > 200 else text)

        # Limit text length to reduce memory usage
        max_text_length = 512
        if len(text) > max_text_length:
            logger.warning("Truncating text from %d to %d characters", len(text), max_text_length)
            text = text[:max_text_length]

        # For Helsinki-NLP model, we need to format the text properly
        if hasattr(translator, 'model') and 'Helsinki' in str(translator.model.config._name_or_path):
            # Helsinki-NLP models expect specific format
            logger.info("Using Helsinki-NLP model for translation")
            result = translator(text, max_new_tokens=128)  # Use max_new_tokens instead of max_length
        else:
            # For Seamless M4T or other models
            logger.info("Using Seamless M4T model for translation")
            # Map common language codes
            lang_mapping = {
                "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita",
                "pt": "por", "ru": "rus", "zh": "cmn", "ja": "jpn", "ko": "kor",
                "ar": "arb", "hi": "hin", "tr": "tur", "pl": "pol", "nl": "nld",
                "he": "heb"  # Added Hebrew mapping
            }

            # Always specify target language for Seamless M4T
            mapped_target = lang_mapping.get(target_lang[:3], "eng")
            logger.info("Mapped target language: %s", mapped_target)

            # Calculate appropriate max_length based on input length
            input_length = len(text.split())  # Word count
            max_length = max(512, input_length * 2)  # At least 512, or 2x input length
            logger.info("Setting max_length to %d for input length %d", max_length, input_length)

            # Try with language specification if available
            if source_lang:
                mapped_source = lang_mapping.get(source_lang[:2], source_lang)
                logger.info("Mapped source language: %s", mapped_source)
                # For Seamless M4T, use the correct parameter format
                result = translator(
                    text,
                    src_lang=mapped_source,
                    tgt_lang=mapped_target
                )
            else:
                # Auto-detect source language but always specify target
                logger.info("Auto-detecting source language")
                # For Seamless M4T, just specify target language
                result = translator(
                    text, 
                    tgt_lang=mapped_target
                )
        
        # Log the result for debugging
        logger.info("Translation result type: %s", type(result))
        if isinstance(result, list) and len(result) > 0:
            logger.info("Translation result: %s", result[0])
        elif isinstance(result, dict):
            logger.info("Translation result: %s", result)
        else:
            logger.info("Translation result: %s", str(result)[:200])

        # Clear cache after translation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
    except Exception as e:
        logger.error("❌ Translation error: %s", e)
        # Clear cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Fallback: try basic translation without language specification
        try:
            result = translator(text, max_new_tokens=128)
            return result
        except Exception as fallback_e:
            logger.error("❌ Fallback translation also failed: %s", fallback_e)
            return None

# ---- Embeddings helper ---------------------------------------------------------
def _embed_sync(texts: List[str], batch_size: int, normalize: bool) -> np.ndarray:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    ).astype(np.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vecs

# ---- API Endpoints -------------------------------------------------------------

@app.post("/detect", response_model=LanguageDetectionResponse)
async def detect_language(request: LanguageDetectionRequest):
    """Detect the language(s) of the input text using FastText"""
    if fasttext_model is None:
        raise HTTPException(status_code=503, detail="FastText model not available")

    try:
        # Clean the text: remove newlines and normalize whitespace
        cleaned_text = request.text.replace('\n', ' ').replace('\r', ' ').strip()
        # Replace multiple spaces with single space
        cleaned_text = ' '.join(cleaned_text.split())

        # Ensure we have some text to process
        if not cleaned_text:
            logger.error("Empty text provided for language detection")
            raise HTTPException(status_code=400, detail="Empty text provided")

        # Log the cleaned text for debugging (first 100 chars)
        logger.debug("Processing text for language detection: %s", cleaned_text[:100])

        labels, probs = fasttext_model.predict(cleaned_text, k=request.k)
        languages = [label.replace("__label__", "") for label in labels]
        probabilities = [round(float(p), 4) for p in probs]

        logger.info("✓ Language detection completed: %s", languages)

        return LanguageDetectionResponse(
            languages=languages,
            probabilities=probabilities
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("❌ Language detection error - Type: %s, Message: %s", type(e).__name__, str(e))
        logger.error("❌ Input text length: %d, cleaned length: %d", len(request.text), len(cleaned_text) if 'cleaned_text' in locals() else 0)
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_content(request: ClassificationRequest):
    """Classify the content of the input text using BART"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classification model not available")

    # Clean the text: remove newlines and normalize whitespace
    cleaned_text = request.text.replace('\n', ' ').replace('\r', ' ').strip()
    cleaned_text = ' '.join(cleaned_text.split())

    # Ensure we have some text to process
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Empty text provided for classification")

    labels = request.labels or DEFAULT_CLASSIFICATION_LABELS

    # Ensure at least two labels for classification
    if len(labels) < 2:
        labels = labels + ["other"]

    async with classification_lock:
        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    thread_pool, _classify_sync, cleaned_text, labels
                ),
                timeout=TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Classification timed out")
            return ClassificationResponse(
                label="timeout", 
                score=0.0, 
                full_result={}
            )
        except Exception as e:
            logger.error("❌ Classification error: %s", e)
            return ClassificationResponse(
                label="error",
                score=0.0,
                full_result={"error": str(e)}
            )

    # Validate result structure
    if not result or "scores" not in result or "labels" not in result:
        logger.error("Invalid classification result structure: %s", result)
        return ClassificationResponse(
            label="error",
            score=0.0,
            full_result={}
        )

    # Extract results
    scores = result["scores"]
    labels_result = result["labels"]

    if not scores or not labels_result:
        logger.error("Empty scores or labels in classification result")
        return ClassificationResponse(
            label="error",
            score=0.0,
            full_result=result
        )

    best_score = scores[0]
    second_score = scores[1] if len(scores) > 1 else 0.0
    best_label = labels_result[0]

    # Apply confidence thresholds
    if best_score < MIN_SCORE or (best_score - second_score) < MIN_MARGIN:
        best_label = "uncertain"

    logger.info("✓ Classification: label=%s, score=%.2f", best_label, best_score)

    return ClassificationResponse(
        label=best_label,
        score=float(best_score),
        full_result=result
    )

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text to English using Seamless M4T v2"""
    if translator is None:
        raise HTTPException(status_code=503, detail="Translation model not available")

    async with translation_lock:
        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    thread_pool, 
                    _translate_sync, 
                    request.text, 
                    request.source_language,
                    request.target_language
                ),
                timeout=TIMEOUT * 2,  # Translation may take longer
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Translation request timed out")
        except Exception as e:
            logger.error("❌ Translation error: %s", e)
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    if result is None:
        raise HTTPException(status_code=500, detail="Translation failed")

    # Extract translation result
    try:
        if isinstance(result, list) and len(result) > 0:
            translated_text = result[0].get('translation_text', '')
        elif isinstance(result, dict):
            translated_text = result.get('translation_text', '')
        else:
            translated_text = str(result)

        logger.info("✓ Translation completed: %d -> %d chars", 
                   len(request.text), len(translated_text))

        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            confidence_score=None  # Seamless M4T doesn't provide confidence scores
        )
    except Exception as e:
        logger.error("❌ Failed to parse translation result: %s", e)
        raise HTTPException(status_code=500, detail="Failed to parse translation result")

@app.post("/embed")
async def embed_items(request: List[EmbeddingItem]) -> List[dict]:
    """
    Compute sentence embeddings for an array of items.
    Uses item['content'] as the text field.
    Returns the same items with a new key "embeddings" (list[float], len=384).
    """
    # Log incoming request details
    logger.info("🔍 /embed endpoint called")
    logger.info(f"📦 Request type: {type(request)}")
    logger.info(f"📊 Request length: {len(request) if hasattr(request, '__len__') else 'unknown'}")
    
    if request:
        logger.info(f"🔬 First item type: {type(request[0]) if len(request) > 0 else 'empty'}")
        if len(request) > 0 and isinstance(request[0], dict):
            logger.info(f"🗝️ First item keys: {list(request[0].keys())}")
            if 'content' in request[0]:
                content_preview = str(request[0]['content'])[:100]
                logger.info(f"📝 First item content preview: {content_preview}...")
    
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embeddings model not available")

    items = [item.dict() for item in request] or []
    if not isinstance(items, list):
        logger.error(f"❌ Input validation failed: Expected list, got {type(items)}")
        raise HTTPException(status_code=400, detail="Input must be a list")

    logger.info(f"✅ Processing {len(items)} items")

    # Hardcoded configuration
    text_field = "content"
    batch_size = 64
    normalize = True

    texts = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            logger.error(f"❌ Item validation failed at index {idx}: Expected dict, got {type(it)}")
            raise HTTPException(status_code=400, detail=f"Item at index {idx} is not an object")
        
        text = str(it.get(text_field, "") or "")
        logger.debug(f"📝 Item {idx}: content length = {len(text)}")
        texts.append(text)

    if not texts:
        logger.warning("⚠️ No texts to process, returning original items")
        return items

    logger.info(f"🔢 Total texts to embed: {len(texts)}")

    try:
        logger.info("🚀 Starting embedding computation...")
        vecs = await asyncio.get_running_loop().run_in_executor(
            thread_pool, _embed_sync, texts, batch_size, normalize
        )
        logger.info(f"✅ Embedding computation completed: {len(vecs)} vectors generated")
    except Exception as e:
        logger.error("❌ Embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    logger.info("📋 Adding embeddings to items...")
    for idx, (it, v) in enumerate(zip(items, vecs)):
        it["embeddings"] = v.tolist()
        logger.debug(f"✅ Added embeddings to item {idx}: vector length = {len(v)}")

    logger.info(f"🎉 Successfully processed {len(items)} items with embeddings")
    return items

@app.post("/cluster", response_model=ClusteringResponse)
async def cluster_texts(request: ClusteringRequest):
    """Cluster a list of texts using entity-aware, topic-gated community detection"""
    if not embedder or not nlp:
        raise HTTPException(status_code=503, detail="Clustering models not available")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided for clustering")
    
    if len(request.texts) < 2:
        # For single text, return single cluster
        return ClusteringResponse(
            total_texts=1,
            total_groups=1,
            groups=[ClusterGroup(
                group_id=0,
                texts=request.texts,
                indices=[0],
                size=1,
                avg_similarity=1.0
            )],
            processing_time=0.0
        )

    # Prepare configuration
    config = {
        "similarity_entity": request.similarity_entity,
        "similarity_global": request.similarity_global,
        "big_community_size": request.big_community_size,
        "avg_similarity_min": request.avg_similarity_min
    }

    import time
    start_time = time.time()

    async with clustering_lock:
        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    thread_pool, 
                    _cluster_sync, 
                    request.texts, 
                    config,
                    request.debug
                ),
                timeout=TIMEOUT * 4,  # Clustering may take longer
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Clustering request timed out")
        except Exception as e:
            logger.error("❌ Clustering error: %s", e)
            raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    processing_time = time.time() - start_time
    
    logger.info("✓ Clustering completed: %d texts -> %d groups in %.2fs", 
               len(request.texts), len(result["groups"]), processing_time)

    return ClusteringResponse(
        total_texts=len(request.texts),
        total_groups=len(result["groups"]),
        groups=result["groups"],
        processing_time=processing_time,
        debug_info=result.get("debug_info")
    )

@app.post("/analyze", response_model=UnifiedAnalysisResponse)
async def unified_analysis(request: UnifiedAnalysisRequest):
    """Perform language detection, content classification, and translation on the input text"""
    response = UnifiedAnalysisResponse(text=request.text)

    # Language detection
    detected_language = None
    if request.detect_language and fasttext_model is not None:
        try:
            logger.debug("Starting language detection for unified analysis")
            lang_request = LanguageDetectionRequest(
                text=request.text,
                k=request.language_count
            )
            response.language_detection = await detect_language(lang_request)
            # Get the most likely detected language for translation
            if response.language_detection and response.language_detection.languages:
                detected_language = response.language_detection.languages[0]
                logger.info("✓ Language detected in unified analysis: %s", detected_language)
        except Exception as e:
            logger.error("❌ Language detection failed in unified analysis - Type: %s, Message: %s", type(e).__name__, str(e))
            logger.error("❌ FastText model status: %s", "loaded" if fasttext_model is not None else "not loaded")

    # Translation to English
    if request.translate_to_english and translator is not None:
        try:
            # Only translate if the detected language is not English
            should_translate = True
            if detected_language:
                # Check if detected language is already English
                if detected_language.lower() in ['en', 'eng', 'english']:
                    should_translate = False
                    response.translation = TranslationResponse(
                        original_text=request.text,
                        translated_text=request.text,
                        source_language=detected_language,
                        target_language="eng",
                        confidence_score=1.0
                    )
            
            if should_translate:
                trans_request = TranslationRequest(
                    text=request.text,
                    source_language=detected_language,
                    target_language="eng"
                )
                response.translation = await translate_text(trans_request)
        except Exception as e:
            logger.error("❌ Translation failed in unified analysis: %s", e)

    # Content classification (use translated text if available)
    if request.classify_content and classifier is not None:
        try:
            # Use translated text for classification if available, otherwise original
            text_to_classify = request.text
            if response.translation and response.translation.translated_text:
                text_to_classify = response.translation.translated_text

            # Clean the text for classification (similar to language detection)
            text_to_classify = text_to_classify.replace('\n', ' ').replace('\r', ' ').strip()
            text_to_classify = ' '.join(text_to_classify.split())

            # Ensure we have some text to classify
            if not text_to_classify:
                logger.warning("Empty text for classification after cleaning")
            else:
                class_request = ClassificationRequest(
                    text=text_to_classify,
                    labels=request.classification_labels
                )
                response.content_classification = await classify_content(class_request)
                logger.info("✓ Classification completed successfully")
        except Exception as e:
            logger.error("❌ Content classification failed in unified analysis: %s", e)
            # Don't fail the entire request, just log the error

    return response

@app.get("/health")
def health():
    """Health check endpoint with system information"""
    process = psutil.Process()

    return {
        "status": "healthy",
        "models": {
            "fasttext_loaded": fasttext_model is not None,
            "classifier_loaded": classifier is not None,
            "translator_loaded": translator is not None,
            "embedder_loaded": embedder is not None,
            "nlp_loaded": nlp is not None,
        },
        "system": {
            "threads": process.num_threads(),
            "memory_mb": round(process.memory_info().rss / 2**20, 1),
            "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        }
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "Unified Text Analysis API",
        "version": "1.2.0",
        "capabilities": ["language_detection", "content_classification", "translation", "text_clustering"],
        "endpoints": {
            "language_detection": "/detect",
            "content_classification": "/classify",
            "translation": "/translate",
            "embeddings": "/embed",
            "text_clustering": "/cluster",
            "unified_analysis": "/analyze",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
def root():
    """Root endpoint with API information"""
    return {
        "service": "Unified Text Analysis API",
        "version": "1.2.0",
        "capabilities": ["language_detection", "content_classification", "translation", "text_clustering"],
        "endpoints": {
            "language_detection": "/detect",
            "content_classification": "/classify",
            "translation": "/translate",
            "embeddings": "/embed",
            "text_clustering": "/cluster",
            "unified_analysis": "/analyze",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
