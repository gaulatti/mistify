#!/usr/bin/env python3
"""
Test script for the Seamless M4T v2 translation integration
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_translation():
    """Test the translation endpoint"""
    print("🧪 Testing Translation Endpoint")
    print("=" * 50)
    
    # Test cases with different languages
    test_cases = [
        {
            "text": "Hola, ¿cómo estás? Me gusta mucho este servicio de traducción.",
            "source_language": "es",
            "description": "Spanish to English"
        },
        {
            "text": "Bonjour, comment allez-vous? J'espère que ce service fonctionne bien.",
            "source_language": "fr", 
            "description": "French to English"
        },
        {
            "text": "Guten Tag, wie geht es Ihnen? Das ist ein interessanter Übersetzungsservice.",
            "source_language": "de",
            "description": "German to English"
        },
        {
            "text": "こんにちは、元気ですか？この翻訳サービスはとても便利です。",
            "source_language": "ja",
            "description": "Japanese to English"
        },
        {
            "text": "צימר במטוס? פלסטיני מציע חוויית נופש ייחודית בתוך מטוס שהפך למתחם אירוח",
            "source_language": "he",
            "description": "Hebrew to English"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Original: {test_case['text']}")
        
        payload = {
            "text": test_case["text"],
            "source_language": test_case["source_language"],
            "target_language": "eng"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/translate", json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Translated: {result['translated_text']}")
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")

def test_unified_analysis_with_translation():
    """Test the unified analysis endpoint with translation enabled"""
    print("\n\n🧪 Testing Unified Analysis with Translation")
    print("=" * 50)
    
    test_text = "Ceci est un article de presse très important sur la technologie moderne."
    
    payload = {
        "text": test_text,
        "detect_language": True,
        "classify_content": True,
        "translate_to_english": True,
        "language_count": 2
    }
    
    print(f"Original text: {test_text}")
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Analysis Results:")
            print(f"   Language Detection: {result.get('language_detection', {}).get('languages', 'N/A')}")
            print(f"   Translation: {result.get('translation', {}).get('translated_text', 'N/A')}")
            print(f"   Classification: {result.get('content_classification', {}).get('label', 'N/A')} "
                  f"({result.get('content_classification', {}).get('score', 0):.2f})")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_health_endpoint():
    """Test the health endpoint to check model status"""
    print("\n\n🧪 Testing Health Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ Health Check Results:")
            print(f"   Status: {result['status']}")
            print(f"   FastText Model: {'✓' if result['models']['fasttext_loaded'] else '❌'}")
            print(f"   Classifier Model: {'✓' if result['models']['classifier_loaded'] else '❌'}")
            print(f"   Translator Model: {'✓' if result['models']['translator_loaded'] else '❌'}")
            print(f"   Device: {result['system']['device']}")
            print(f"   Memory: {result['system']['memory_mb']} MB")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    print("🚀 Seamless M4T v2 Translation Integration Test")
    print("Make sure the server is running on http://localhost:8000")
    print("\nStarting tests...\n")
    
    # Test health first
    test_health_endpoint()
    
    # Test translation endpoint
    test_translation()
    
    # Test unified analysis with translation
    test_unified_analysis_with_translation()
    
    print("\n\n🎉 Tests completed!")
