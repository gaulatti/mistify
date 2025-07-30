#!/usr/bin/env python3
"""
Quick test script to verify the running translation service
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ Service Status:", result['status'])
            print(f"   FastText: {'✓' if result['models']['fasttext_loaded'] else '❌'}")
            print(f"   Classifier: {'✓' if result['models']['classifier_loaded'] else '❌'}")
            print(f"   Translator: {'✓' if result['models']['translator_loaded'] else '❌'}")
            print(f"   Device: {result['system']['device']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check exception: {e}")
        return False

def test_translation():
    """Test translation functionality"""
    print("\n🌍 Testing Translation")
    
    test_cases = [
        {
            "text": "Bonjour, comment allez-vous?",
            "source_language": "fr",
            "description": "French to English"
        },
        {
            "text": "Hola, ¿cómo estás?",
            "source_language": "es", 
            "description": "Spanish to English"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📝 {test_case['description']}")
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
                print(f"   ❌ Translation failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ Translation exception: {e}")

def test_unified_analysis():
    """Test unified analysis with translation"""
    print("\n🔬 Testing Unified Analysis with Translation")
    
    test_text = "Esto es una noticia muy importante sobre tecnología."
    
    payload = {
        "text": test_text,
        "detect_language": True,
        "classify_content": True,
        "translate_to_english": True
    }
    
    print(f"   Text: {test_text}")
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Results:")
            
            if result.get('language_detection'):
                lang = result['language_detection']['languages'][0]
                prob = result['language_detection']['probabilities'][0]
                print(f"      Language: {lang} ({prob:.3f})")
            
            if result.get('translation'):
                print(f"      Translation: {result['translation']['translated_text']}")
            
            if result.get('content_classification'):
                label = result['content_classification']['label']
                score = result['content_classification']['score']
                print(f"      Classification: {label} ({score:.3f})")
        else:
            print(f"   ❌ Analysis failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Analysis exception: {e}")

if __name__ == "__main__":
    print("🚀 Testing Translation Service")
    print("=" * 50)
    
    if test_health():
        test_translation()
        test_unified_analysis()
    
    print("\n🎉 Testing completed!")
