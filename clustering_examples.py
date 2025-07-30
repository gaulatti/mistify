#!/usr/bin/env python3
"""
Comprehensive example demonstrating the clustering endpoint
with different types of text collections
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_news_headlines():
    """Test clustering with news headlines"""
    print("🔍 Testing news headlines clustering...")
    
    headlines = [
        "Tesla stock surges 15% after better-than-expected earnings report",
        "Apple unveils revolutionary new iPhone with AI capabilities",
        "Microsoft acquires major AI startup for $2.5 billion",
        "Electric vehicle sales jump 40% in Q3 as demand grows",
        "Elon Musk announces new Tesla Gigafactory in Texas",
        "Breaking: 7.2 magnitude earthquake strikes California coast",
        "Emergency crews rush to earthquake disaster zone",
        "Tsunami warning issued after major Pacific earthquake",
        "Local basketball team wins state championship",
        "Football season kicks off with record attendance",
        "Soccer world cup preparations intensify",
        "Federal Reserve hints at interest rate changes",
        "Apple stock drops 3% on supply chain concerns",
        "Amazon reports strong quarterly earnings growth"
    ]
    
    response = requests.post(f"{BASE_URL}/cluster", json={
        "texts": headlines,
        "debug": True
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {result['total_groups']} clusters from {result['total_texts']} headlines")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        for group in result['groups']:
            print(f"\n🧠 Group {group['group_id']} - {group['primary_topic']} ({group['size']} items)")
            print(f"   Entities: {group['primary_entities'][:3]}")  # Show first 3 entities
            print(f"   Similarity: {group['avg_similarity']:.3f}")
            for text in group['texts']:
                print(f"   • {text}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_product_reviews():
    """Test clustering with product reviews"""
    print("\n🔍 Testing product reviews clustering...")
    
    reviews = [
        "This phone has amazing battery life and great camera quality",
        "Love the new camera features, photos look professional",
        "Battery lasts all day even with heavy usage, very impressed",
        "Delivery was super fast, arrived next day as promised",
        "Package arrived quickly and was well protected",
        "Shipping was delayed but customer service was helpful",
        "Screen is too small and hard to read text",
        "Display quality is poor, colors look washed out",
        "Touch screen is unresponsive and frustrating to use",
        "Great value for money, features exceed expectations",
        "Excellent build quality and premium materials used",
        "Price is reasonable for all the features included"
    ]
    
    response = requests.post(f"{BASE_URL}/cluster", json={
        "texts": reviews,
        "topic_labels": ["product_quality", "shipping", "price", "user_experience", "misc"],
        "similarity_global": 0.55,  # Slightly lower threshold for reviews
        "debug": False
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Found {result['total_groups']} review clusters")
        
        for group in result['groups']:
            print(f"\n📱 Review Group {group['group_id']} - {group['primary_topic']} ({group['size']} reviews)")
            for text in group['texts']:
                print(f"   • {text}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_custom_config():
    """Test clustering with custom configuration"""
    print("\n🔍 Testing custom clustering configuration...")
    
    texts = [
        "Apple CEO Tim Cook announces new sustainability initiative",
        "Microsoft CEO Satya Nadella discusses AI future at conference", 
        "Google parent company Alphabet reports strong earnings",
        "Tesla CEO Elon Musk tweets about Mars colonization plans",
        "Amazon founder Jeff Bezos invests in space technology",
        "OpenAI releases new language model with improved capabilities",
        "Meta unveils latest VR headset with enhanced features",
        "Netflix announces price increase for premium subscriptions"
    ]
    
    # Custom configuration for tighter clustering
    response = requests.post(f"{BASE_URL}/cluster", json={
        "texts": texts,
        "similarity_entity": 0.50,    # Higher threshold for entity matches
        "similarity_global": 0.70,    # Higher threshold for content similarity
        "big_community_size": 20,     # Smaller max community size
        "topic_labels": ["technology", "business", "leadership", "innovation", "finance"],
        "debug": True
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Custom clustering: {result['total_groups']} groups")
        print(f"  Debug: {result['debug_info']['total_edges']} edges created")
        
        for group in result['groups']:
            print(f"\n💼 Group {group['group_id']} ({group['size']} items)")
            for i, text in enumerate(group['texts']):
                print(f"   {i+1}. {text}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_health_check():
    """Check if all models are loaded properly"""
    print("🔍 Checking server health...")
    
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✓ Server status: {health['status']}")
        print(f"  Device: {health['system']['device']}")
        print(f"  Memory: {health['system']['memory_mb']} MB")
        
        models = health['models']
        print(f"  Models loaded:")
        for model, loaded in models.items():
            status = "✓" if loaded else "❌"
            print(f"    {status} {model}")
            
        if not models.get('embedder_loaded') or not models.get('nlp_loaded'):
            print("\n⚠️  Clustering models not loaded. Run 'python -m spacy download en_core_web_sm'")
            return False
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False
    
    return True

def main():
    print("🚀 Unified Text Analysis API - Clustering Examples")
    print("=" * 60)
    
    # Check health first
    if not test_health_check():
        print("\n❌ Server not ready. Please check setup and try again.")
        return
    
    print("\n" + "=" * 60)
    
    try:
        # Run clustering tests
        test_news_headlines()
        time.sleep(1)  # Brief pause between tests
        
        test_product_reviews()
        time.sleep(1)
        
        test_custom_config()
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on localhost:8000")
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print("\n🎉 Clustering examples completed!")
    print("\n📚 For more details, see CLUSTERING.md")

if __name__ == "__main__":
    main()
