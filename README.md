# Mistify: Unified AI Helpers

Mistify is a set of modular AI helpers for text analysis, clustering, classification, translation, and more. It provides a unified FastAPI service and reusable components for building advanced language and content workflows.

For detailed documentation, please visit the [Mistify Wiki](https://github.com/gaulatti/mistify/wiki).

## Features

- **Language Detection**: FastText-based, supports 176 languages, confidence scores
- **Content Classification**: Zero-shot (DistilBART), custom labels, GPU-accelerated
- **Translation**: Seamless M4T v2, Helsinki-NLP fallback, 100+ language pairs
- **Text Clustering**: Entity-aware, topic-gated, alias discovery, graph-based
- **Unified Analysis**: Combine detection, translation, and classification in one call
- **Modular API**: Use endpoints independently or compose for advanced workflows

## API Endpoints

Mistify exposes a unified FastAPI service with endpoints for each helper:

- `/detect` — Language detection
- `/classify` — Content classification
- `/translate` — Translation
- `/cluster` — Text clustering
- `/analyze` — Unified multi-step analysis
- `/health` — Health check

For detailed API documentation, including request/response examples, please see the [API Documentation on the Wiki](https://github.com/gaulatti/mistify/wiki/Text-Clustering-API-Documentation).

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (for containerized deployment)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/gaulatti/mistify.git
    cd mistify
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    The required models will be downloaded automatically on first use. For a full offline installation, please refer to the [documentation on the wiki](https://github.com/gaulatti/mistify/wiki).

### Running with Docker

The easiest way to run Mistify is with Docker. The Docker build pre-downloads all models, ensuring offline operation.

1.  **Build the Docker image:**

    ```bash
    docker build -t mistify-service .
    ```

2.  **Run the container:**
    ```bash
    docker run --memory=12g -p 8000:8000 mistify-service
    ```

### Running Locally

You can also run the FastAPI server directly with Uvicorn.

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Configuration

Configure the service using the following environment variables:

- `FASTTEXT_MODEL_PATH`: Path to the FastText language model (default: `lid.176.bin`)
- `MIN_SCORE`: Minimum confidence score for classification (default: `0.30`)
- `MIN_MARGIN`: Minimum margin between top two scores (default: `0.10`)
- `TRANSFORMERS_OFFLINE`: Set to `1` to enforce offline mode for Hugging Face models (default in Docker).
