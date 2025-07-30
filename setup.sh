#!/bin/bash
# Setup script for the Unified Text Analysis API with clustering support

echo "🔧 Setting up Unified Text Analysis API..."

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Download SpaCy English model
echo "🧠 Downloading SpaCy English model..."
python -m spacy download en_core_web_sm

# Download FastText language detection model if not present
if [ ! -f "lid.176.bin" ]; then
    echo "🌍 Downloading FastText language detection model..."
    wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
fi

# Pre-download all AI models (transformers, sentence-transformers, etc.)
echo "🤖 Pre-downloading AI models..."
echo "This may take several minutes and download ~9GB of models..."
python download_models.py

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the server, run:"
echo "   python server.py"
echo ""
echo "🔍 To test clustering, run:"
echo "   python test_clustering.py"
echo ""
echo "🌟 To see comprehensive examples, run:"
echo "   python clustering_examples.py"
echo ""
echo "📚 See CLUSTERING.md for detailed documentation"
echo ""
echo "💡 All models are now cached locally for offline operation!"
