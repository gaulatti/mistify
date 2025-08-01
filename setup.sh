#!/bin/bash
# Setup script for the Unified Text Analysis API with clustering support

echo "🔧 Setting up Unified Text Analysis API..."

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Download SpaCy English model
echo "🧠 Downloading SpaCy English model..."
python -m spacy download en_core_web_sm

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
echo "💡 Models will be downloaded automatically at runtime when needed!"
