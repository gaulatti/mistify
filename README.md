# Unified Text Analysis API

A comprehensive FastAPI service that provides language detection, content classification, translation, and text clustering capabilities in a single, unified endpoint.

## Features

### Language Detection
- Uses Facebook's FastText model for fast and accurate language identification
- Supports detection of 176 languages
- Returns multiple language predictions with confidence scores

### Content Classification  
- Uses DistilBART for zero-shot text classification
- Classifies content into categories like "breaking news", "humor/satire", etc.
- Supports custom classification labels
- GPU-accelerated when available

### Translation
- Uses Facebook's Seamless M4T v2 Large model for high-quality translation
- Fallback to Helsinki-NLP models for broader compatibility
- Supports translation from multiple languages to English
- Auto-detects source language when not specified
- Handles 100+ language pairs with robust error handling

### Text Clustering 🆕
- Entity-aware, topic-gated community detection for grouping similar texts
- Automatic alias discovery for entity normalization
- Uses Sentence Transformers for semantic embeddings
- SpaCy for named entity recognition
- Graph-based clustering with Louvain community detection
- Configurable similarity thresholds and community splitting

### Unified Analysis
- Single endpoint that performs language detection, content classification, and translation
- Configurable to run any combination of analyses
- Optimized for performance with async processing
- Intelligent workflow: detects language → translates if needed → classifies content

## API Endpoints

### `/detect`
Detect the language(s) of input text.

**Request:**
```json
{
  "text": "Hello, how are you today?",
  "k": 2
}
```

**Response:**
```json
{
  "languages": ["en", "es"],
  "probabilities": [0.9923, 0.0045]
}
```

### `/classify`
Classify the content type of input text.

**Request:**
```json
{
  "text": "Breaking: Major earthquake hits the region",
  "labels": ["breaking news", "opinion", "humor"]
}
```

**Response:**
```json
{
  "label": "breaking news",
  "score": 0.8542,
  "full_result": {...}
}
```

### `/translate`
Translate text to English using Seamless M4T v2.

**Request:**
```json
{
  "text": "Bonjour, comment allez-vous?",
  "source_language": "fr",
  "target_language": "eng"
}
```

**Response:**
```json
{
  "original_text": "Bonjour, comment allez-vous?",
  "translated_text": "Hello, how are you?",
  "source_language": "fr",
  "target_language": "eng",
  "confidence_score": null
}
```

### `/analyze`
Perform unified analysis (language detection, content classification, and translation).

**Request:**
```json
{
  "text": "C'est une nouvelle très importante sur la technologie",
  "detect_language": true,
  "classify_content": true,
  "translate_to_english": true,
  "language_count": 1,
  "classification_labels": ["breaking news", "technology", "opinion"]
}
```

**Response:**
```json
{
  "text": "C'est une nouvelle très importante sur la technologie",
  "language_detection": {
    "languages": ["fr"],
    "probabilities": [0.9984]
  },
  "translation": {
    "original_text": "C'est une nouvelle très importante sur la technologie",
    "translated_text": "This is very important news about technology",
    "source_language": "fr",
    "target_language": "eng"
  },
  "content_classification": {
    "label": "breaking news",
    "score": 0.8421,
    "full_result": {...}
  }
}
```

### `/cluster` 🆕
Cluster a list of texts using entity-aware, topic-gated community detection.

**Request:**
```json
{
  "texts": [
    "Tesla stock surges after earnings beat expectations",
    "Apple announces new iPhone with revolutionary features", 
    "Electric vehicle sales continue to grow worldwide",
    "Breaking: Major earthquake hits California coast",
    "Emergency services respond to 7.2 magnitude quake"
  ],
  "similarity_entity": 0.40,
  "similarity_global": 0.60,
  "debug": true
}
```

**Response:**
```json
{
  "total_texts": 5,
  "total_groups": 3,
  "processing_time": 2.45,
  "groups": [
    {
      "group_id": 0,
      "texts": [
        "Tesla stock surges after earnings beat expectations",
        "Electric vehicle sales continue to grow worldwide"
      ],
      "indices": [0, 2],
      "size": 2,
      "primary_topic": "economy",
      "primary_entities": ["Tesla"],
      "avg_similarity": 0.72
    },
    {
      "group_id": 1,
      "texts": [
        "Breaking: Major earthquake hits California coast",
        "Emergency services respond to 7.2 magnitude quake"
      ],
      "indices": [3, 4],
      "size": 2,
      "primary_topic": "conflict",
      "primary_entities": ["California"],
      "avg_similarity": 0.89
    },
    {
      "group_id": 2,
      "texts": [
        "Apple announces new iPhone with revolutionary features"
      ],
      "indices": [1],
      "size": 1,
      "primary_topic": "economy",
      "primary_entities": ["Apple", "iPhone"],
      "avg_similarity": 1.0
    }
  ],
  "debug_info": {
    "edge_reasons": {"0-2": "ENT & 0.45", "3-4": "0.89"},
    "total_edges": 2,
    "topics": ["economy", "economy", "economy", "conflict", "conflict"],
    "entities_count": 8
  }
}
```

### `/health`
Health check with system information.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "fasttext_loaded": true,
    "classifier_loaded": true,
    "translator_loaded": true,
    "embedder_loaded": true,
    "nlp_loaded": true
  },
  "system": {
    "threads": 8,
    "memory_mb": 1024.5,
    "device": "gpu"
  }
}
```

## Environment Variables

- `FASTTEXT_MODEL_PATH`: Path to the FastText language model (default: `lid.176.bin`)
- `MIN_SCORE`: Minimum confidence score for classification (default: `0.30`)
- `MIN_MARGIN`: Minimum margin between top two scores (default: `0.10`)

## Installation & Setup

### Quick Setup (Recommended)
```bash
# Run the automated setup script
./setup.sh
```

### Manual Installation

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Download SpaCy Model (for clustering)
```bash
python -m spacy download en_core_web_sm
```

#### Download FastText Language Model
```bash
curl -LO https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

#### Pre-download All AI Models (recommended)
```bash
python download_models.py
```

**Note:** All models (~9GB total) are pre-downloaded during Docker build. For local development, you can either let models download on first use or run the download script manually for offline operation.

## Running with Docker

```bash
# Build the image (all models are pre-downloaded and verified during build)
docker build -t unified-text-analysis .

# Run the container (requires significant memory for translation model)
docker run --memory=12g -p 8000:8000 unified-text-analysis
```

**Note:** The Docker build process pre-downloads and verifies all models (~9GB total) during the build phase, ensuring 100% offline operation. The container starts immediately without any network requests. The build process may take 15-20 minutes depending on your internet connection, but subsequent runs are instant.

## Running Locally

```bash
# Run the server
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Testing

Run the provided test script to verify all functionality:

```bash
python test_translation.py
```

## Migration from Separate Services

This unified service now includes:
- **FastText service** (language detection) → `/detect`
- **Opinion service** (content classification) → `/classify`  
- **NEW: Translation service** → `/translate`

### Breaking Changes
- Port consolidation: All services now run on port 8000
- **No endpoint path changes** for existing functionality
- New translation capabilities added

### Migration Guide
1. Update client code to use port 8000 for all services
2. **No endpoint path changes needed** - same `/detect` and `/classify` paths
3. Optionally use the new `/translate` endpoint for translation
4. Use the enhanced `/analyze` endpoint for combined analysis with translation

## Performance Notes

- GPU acceleration is recommended for translation (requires ~8GB VRAM)
- Translation model is large (~9GB) and requires significant memory
- CPU-only mode is supported but will be slower for translation
- All models are loaded once at startup for consistent response times
- Thread pooling and async processing ensure optimal performance

## Supported Languages for Translation

The Seamless M4T v2 model supports translation from 100+ languages to English, including:
- European languages: Spanish, French, German, Italian, Portuguese, Russian, etc.
- Asian languages: Chinese, Japanese, Korean, Hindi, Arabic, etc.
- And many more...
