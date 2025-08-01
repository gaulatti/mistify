# Docker Build Process Documentation

This project now uses a two-stage Docker build process to separate model downloading from application serving.

## Overview

The build process has been refactored into two separate Docker images:

1. **Models Image** (`Dockerfile.models`) - Downloads and caches all AI models
2. **Application Image** (`Dockerfile`) - Contains application code and copies pre-built models

## Build Instructions

### Step 1: Build the Models Image

```bash
docker build -f Dockerfile.models -t myproject-models:latest .
```

This image:
- Downloads all HuggingFace models to `/models/.hf_models/`
- Downloads FastText language detection model to `/models/lid.176.bin`
- Downloads SpaCy English model (installed as Python package)
- Verifies all models are working correctly
- **Note:** This build can take 30+ minutes and downloads ~9GB of models

### Step 2: Build the Application Image

```bash
docker build -t myproject-app:latest .
```

Or, to specify a different models image:

```bash
docker build --build-arg MODELS_IMAGE=my-custom-models:v1.0 -t myproject-app:latest .
```

This image:
- Installs Python dependencies
- Downloads SpaCy model (required in both images)
- Copies all pre-built models from `myproject-models:latest` using `COPY --from`
- Copies application code
- Sets environment variables to use the copied models

### Step 3: Run the Application

```bash
docker run -p 8000:8000 myproject-app:latest
```

## Benefits

- **Faster iterations**: Application changes don't require re-downloading models
- **Separate concerns**: Model management is isolated from application deployment
- **CI/CD friendly**: Models image can be built once and reused across deployments
- **Storage efficient**: Models are cached in a separate layer

## Environment Variables

The application uses these environment variables to locate models:

- `FASTTEXT_MODEL_PATH=/models/lid.176.bin`
- `HF_HOME=/models/.hf_models`
- `TRANSFORMERS_CACHE=/models/.hf_models`
- `HF_HUB_CACHE=/models/.hf_models`
- `HF_HUB_OFFLINE=1` (forces offline model usage)

## Model Locations

After the build process, models are located at:

```
/models/
├── lid.176.bin                 # FastText language detection model
└── .hf_models/                 # HuggingFace models cache
    ├── models--valhalla--distilbart-mnli-12-3/
    ├── models--facebook--seamless-m4t-v2-large/
    ├── models--Helsinki-NLP--opus-mt-mul-en/
    ├── models--sentence-transformers--all-MiniLM-L6-v2/
    └── ...
```

SpaCy models are installed as Python packages and available system-wide in both images.

## Troubleshooting

### Models Image Build Fails
- Ensure sufficient disk space (>10GB free)
- Check internet connectivity for model downloads
- Increase Docker build timeout if needed

### Application Image Build Fails
- Ensure the models image `myproject-models:latest` exists
- Check that `COPY --from=myproject-models:latest` can access the models image

### Runtime Model Loading Issues
- Verify environment variables are set correctly
- Check that `/models/` directory exists and contains expected files
- Ensure offline mode environment variables are set (`HF_HUB_OFFLINE=1`)

## Development Workflow

For development, you can:

1. Build models image once: `docker build -f Dockerfile.models -t myproject-models:latest .`
2. Iterate on application code by rebuilding only the app image: `docker build -t myproject-app:latest .`
3. The app image build will be much faster since it doesn't download models

## CI/CD Integration

In CI/CD pipelines:

1. Build and push models image to registry (infrequently, when models change)
2. For each deployment, build app image using the cached models image
3. Deploy the lightweight app image

Example:
```bash
# In CI/CD - build models (run rarely)
docker build -f Dockerfile.models -t registry.com/myproject-models:v1.0 .
docker push registry.com/myproject-models:v1.0

# In CI/CD - build app (run frequently)  
docker build --build-arg MODELS_IMAGE=registry.com/myproject-models:v1.0 -t registry.com/myproject-app:latest .
docker push registry.com/myproject-app:latest
```