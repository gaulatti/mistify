# Text Clustering API Documentation

## Overview

The Unified Text Analysis API now includes advanced text clustering capabilities using entity-aware, topic-gated community detection. This feature can automatically group similar texts based on content similarity, shared entities, and topic coherence.

## How It Works

The clustering algorithm uses a multi-step approach:

1. **Entity Extraction**: Uses SpaCy to extract named entities (people, places, organizations, etc.)
2. **Topic Classification**: Classifies texts into topics (economy, politics, sports, conflict, misc)
3. **Alias Discovery**: Automatically discovers entity aliases using cosine similarity
4. **Graph Construction**: Builds a similarity graph with weighted edges based on:
   - Shared entities + content similarity (lower threshold)
   - High content similarity alone (higher threshold)
5. **Community Detection**: Uses Louvain algorithm to detect communities
6. **Large Community Splitting**: Recursively splits large communities using tighter similarity constraints

## API Endpoint

### POST /cluster

Clusters a list of texts using entity-aware, topic-gated community detection.

#### Request Body

```json
{
  "texts": ["text1", "text2", "text3"],
  "similarity_entity": 0.40,     // Similarity threshold for entity matches (optional)
  "similarity_global": 0.60,     // Global similarity threshold (optional)
  "big_community_size": 30,      // Max size before splitting (optional)
  "avg_similarity_min": 0.50,    // Min avg similarity for large communities (optional)
  "topic_labels": ["economy", "politics", "sports", "conflict", "misc"], // Custom topics (optional)
  "debug": false                 // Include debug information (optional)
}
```

#### Response

```json
{
  "total_texts": 10,
  "total_groups": 3,
  "processing_time": 2.45,
  "groups": [
    {
      "group_id": 0,
      "texts": ["text1", "text2"],
      "indices": [0, 1],
      "size": 2,
      "primary_topic": "economy",
      "primary_entities": ["Tesla", "Apple"],
      "avg_similarity": 0.85
    }
  ],
  "debug_info": {  // Only included if debug=true
    "edge_reasons": {"0-1": "ENT & 0.75"},
    "total_edges": 5,
    "topics": ["economy", "politics"],
    "entities_count": 12
  }
}
```

## Configuration Parameters

- **similarity_entity** (default: 0.40): Cosine similarity threshold for texts sharing entities
- **similarity_global** (default: 0.60): Cosine similarity threshold for all texts
- **big_community_size** (default: 30): Maximum community size before splitting
- **avg_similarity_min** (default: 0.50): Minimum average similarity for large communities
- **topic_labels**: Custom list of topics for classification

## Example Usage

### Python

```python
import requests

texts = [
    "Tesla stock surges after earnings beat",
    "Apple announces new iPhone features", 
    "Electric vehicle sales grow worldwide",
    "Major earthquake hits California",
    "Emergency services respond to quake"
]

response = requests.post("http://localhost:8000/cluster", json={
    "texts": texts,
    "debug": True
})

result = response.json()
print(f"Found {result['total_groups']} clusters from {result['total_texts']} texts")

for group in result['groups']:
    print(f"\\nGroup {group['group_id']} ({group['size']} items):")
    print(f"Topic: {group['primary_topic']}")
    print(f"Entities: {group['primary_entities']}")
    for text in group['texts']:
        print(f"  - {text}")
```

### curl

```bash
curl -X POST "http://localhost:8000/cluster" \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": [
      "Tesla stock surges after earnings beat",
      "Apple announces new iPhone features"
    ],
    "debug": true
  }'
```

## Model Dependencies

The clustering functionality requires additional models:

1. **Sentence Transformer**: `all-MiniLM-L6-v2` for text embeddings
2. **SpaCy Model**: `en_core_web_sm` for entity extraction
3. **BART Classifier**: Uses existing classifier for topic classification

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Performance Considerations

- **Memory Usage**: Large text collections may require significant GPU/CPU memory
- **Processing Time**: Scales roughly O(n²) with number of texts due to similarity computation
- **Batch Processing**: Texts are processed in batches for efficiency
- **Device Support**: Automatically uses CUDA > MPS > CPU

## Error Handling

The endpoint handles various error conditions:

- **503 Service Unavailable**: Clustering models not loaded
- **400 Bad Request**: Empty or invalid text list
- **408 Request Timeout**: Processing took too long (configurable timeout)
- **500 Internal Server Error**: Unexpected clustering failure

## Integration with Other Endpoints

The clustering functionality integrates seamlessly with existing endpoints:

- Can be combined with `/translate` to cluster multilingual content
- Uses the same classification pipeline as `/classify`
- Shares device management and error handling patterns

## Best Practices

1. **Text Preprocessing**: Clean texts of excessive whitespace and normalize encoding
2. **Batch Size**: Process 100-1000 texts per request for optimal performance  
3. **Parameter Tuning**: Adjust similarity thresholds based on your domain
4. **Topic Labels**: Customize topic labels for your specific use case
5. **Debug Mode**: Use debug=true during development to understand clustering decisions
