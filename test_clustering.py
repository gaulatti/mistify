#!/usr/bin/env python3
"""
Test script for the clustering endpoint
"""
import requests
import json

# Sample headlines for testing
test_headlines = [
    "Tesla stock surges after earnings beat expectations",
    "Apple announces new iPhone with revolutionary features",
    "Microsoft acquires AI startup for $2 billion",
    "Electric vehicle sales continue to grow worldwide",
    "Elon Musk announces new Tesla factory in Texas",
    "Breaking: Major earthquake hits California coast",
    "Emergency services respond to 7.2 magnitude quake",
    "Scientists warn of potential tsunami after earthquake",
    "Local football team wins championship game",
    "Basketball season kicks off with exciting matchups",
    "Soccer world cup preparations underway",
    "Tech giants face new regulatory challenges",
    "Apple stock drops after supply chain concerns",
    "Amazon reports strong quarterly earnings"
]

def test_clustering_endpoint():
    """Test the clustering endpoint with sample data"""
    url = "http://localhost:8000/cluster"
    
    payload = {
        "texts": test_headlines,
        "debug": True
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Clustering successful!")
            print(f"Total texts: {result['total_texts']}")
            print(f"Total groups: {result['total_groups']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            
            print("\n--- Clusters ---")
            for group in result['groups']:
                print(f"\n🧠 Group {group['group_id']} (size: {group['size']}):")
                print(f"   Topic: {group['primary_topic']}")
                print(f"   Entities: {group['primary_entities']}")
                print(f"   Avg similarity: {group['avg_similarity']:.3f}")
                for text in group['texts']:
                    print(f"   - {text}")
            
            if result.get('debug_info'):
                print(f"\nDebug info: {result['debug_info']['total_edges']} edges created")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health_endpoint():
    """Test if the server is running and models are loaded"""
    url = "http://localhost:8000/health"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            health = response.json()
            print("✓ Server is healthy!")
            print(f"Models loaded: {health['models']}")
            print(f"System info: {health['system']}")
            
            if not health['models']['embedder_loaded']:
                print("⚠️  Warning: Embedder not loaded - clustering may not work")
            if not health['models']['nlp_loaded']:
                print("⚠️  Warning: SpaCy model not loaded - clustering may not work")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("Testing Unified Text Analysis API - Clustering")
    print("=" * 50)
    
    # First check if server is healthy
    test_health_endpoint()
    print()
    
    # Test clustering
    test_clustering_endpoint()
