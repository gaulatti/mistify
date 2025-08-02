# Clustering Integration Summary

## What's Been Added

### 1. New Dependencies (requirements.txt)
- `networkx>=3.0` - Graph algorithms for community detection
- `sentence-transformers>=2.2.2` - Semantic text embeddings  
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `tqdm>=4.64.0` - Progress bars

### 2. Server Enhancements (server.py)

#### New Models
- **Sentence Transformer**: `all-MiniLM-L6-v2` for text embeddings
- **SpaCy NLP**: `en_core_web_sm` for named entity recognition
- **Clustering Pipeline**: Integrated with existing BART classifier

#### New Endpoint: `/cluster`
- **Input**: List of texts with configurable parameters
- **Output**: Grouped clusters with metadata (topics, entities, similarity scores)
- **Features**:
  - Entity-aware clustering (texts sharing entities cluster more easily)
  - Topic-gated (only texts in same topic can cluster)
  - Automatic alias discovery for entity normalization
  - Community detection using Louvain algorithm
  - Large community splitting for better granularity

#### Configuration Parameters
- `similarity_entity`: Threshold for entity-based clustering (default: 0.40)
- `similarity_global`: Threshold for content-based clustering (default: 0.60)  
- `big_community_size`: Max community size before splitting (default: 30)
- `avg_similarity_min`: Min average similarity for large communities (default: 0.50)
- `topic_labels`: Custom topic classification labels

### 3. Updated API Response Models
- `ClusteringRequest`: Input validation and configuration
- `ClusteringResponse`: Structured cluster results with metadata
- `ClusterGroup`: Individual cluster information

### 4. Enhanced Health Check
- Added `embedder_loaded` and `nlp_loaded` status
- Updated version to 1.2.0
- Added clustering capability to endpoints list

### 5. Documentation

#### New Files
- `CLUSTERING.md`: Comprehensive clustering documentation

#### Updated Files
- `README.md`: Added clustering features and setup instructions
- `Dockerfile`: Added SpaCy model download step

## Algorithm Overview

The clustering algorithm works in stages:

1. **Text Preprocessing**: Clean and normalize input texts
2. **Entity Extraction**: Use SpaCy to extract named entities (people, places, orgs)
3. **Alias Discovery**: Cluster similar entity names using cosine similarity
4. **Topic Classification**: Classify texts into predefined topics
5. **Embedding Generation**: Create semantic embeddings using Sentence Transformers
6. **Graph Construction**: Build similarity graph with weighted edges:
   - Higher weight for texts sharing entities + moderate similarity
   - Lower weight for texts with high content similarity alone
   - No edges between different topics or "misc" topic
7. **Community Detection**: Use Louvain algorithm to find communities
8. **Community Splitting**: Recursively split large communities using tighter thresholds

## Key Features

### Entity-Aware Clustering
- Texts mentioning the same entities (even with different names) cluster together
- Automatic alias discovery (e.g., "USA" and "United States")
- Entity types: People, Places, Organizations, Facilities

### Topic-Gated Clustering  
- Only texts classified into the same topic can cluster
- Prevents mixing unrelated content (e.g., sports with politics)
- Configurable topic labels

### Configurable Thresholds
- `similarity_entity`: Lower bar for texts sharing entities
- `similarity_global`: Higher bar for content similarity alone
- Allows fine-tuning for different domains

### Scalable Community Detection
- Louvain algorithm for efficient community detection
- Automatic splitting of large communities
- Handles 100s-1000s of texts efficiently

## Performance Characteristics

- **Memory**: ~2-4GB for embedder + classifier models
- **Processing**: O(n²) similarity computation, O(n log n) clustering
- **Throughput**: ~100-500 texts per request (depends on text length)
- **Device Support**: CUDA > MPS > CPU (automatic detection)

## Usage Examples

### Basic Clustering
```python
response = requests.post("http://localhost:8000/cluster", json={
    "texts": ["text1", "text2", "text3"]
})
```

### Custom Configuration
```python
response = requests.post("http://localhost:8000/cluster", json={
    "texts": ["text1", "text2", "text3"],
    "similarity_entity": 0.45,
    "similarity_global": 0.65,
    "topic_labels": ["news", "opinion", "analysis"],
    "debug": True
})
```

## Integration Benefits

1. **Unified API**: Clustering joins language detection, classification, and translation
2. **Shared Infrastructure**: Reuses device management, error handling, and async processing
3. **Model Efficiency**: Shares BART classifier for topic classification
4. **Consistent Interface**: Same FastAPI patterns and response formats

## Next Steps

1. Install dependencies with `pip install -r requirements.txt`
2. Download the SpaCy model with `python -m spacy download en_core_web_sm`
3. Start the server with `python server.py` or `uvicorn server:app`
4. Read `CLUSTERING.md` for detailed documentation
5. Customize parameters for your use case
