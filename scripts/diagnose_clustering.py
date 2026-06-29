"""
Diagnostic script for the /cluster endpoint.

Simulates Monitor-style inputs (main post + similarPosts from Qdrant) and prints
the resulting cluster. This makes it easy to see whether related posts are being
merged and unrelated posts are being kept out.

Run:
    uv run python scripts/diagnose_clustering.py
"""
import asyncio
import logging
import os
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reduce noise from transformers/sentence-transformers.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
os.environ.setdefault("TRANSLATION_ENABLED", "false")

from src.helpers.models import initialize_models
from src.helpers.clustering import build_clustering_graph, split_large_communities
from src.endpoints.clustering import cluster_texts
from src.models import PostData


CASES = [
    (
        "same_event",
        "BREAKING: Magnitude 6.1 earthquake strikes northeastern Japan, tsunami warning issued",
        [
            "Strong earthquake hits Japan's northeast coast, residents urged to evacuate",
            "Tsunami warning after M6.1 quake off Fukushima prefecture",
            "Japan earthquake: no immediate reports of major damage",
            # distractor
            "Apple announces new iPhone model at developer conference",
        ],
    ),
    (
        "different_angles_same_event",
        "Senator Smith proposes new climate bill in Congress",
        [
            "Climate legislation introduced by Senator Smith faces Republican opposition",
            "Smith's climate bill would allocate $50B to renewable energy",
            "Analysts say Senator Smith's proposal has little chance of passing",
            # distractor
            "Senator Johnson calls for infrastructure spending in rural areas",
        ],
    ),
    (
        "shared_person_different_events",
        "Taylor Swift announces new album release date",
        [
            "Taylor Swift fans queue for tickets to upcoming concert",
            "Taylor Swift donates $1M to food bank in hometown",
            # distractor
            "Beyonce releases surprise single on streaming platforms",
        ],
    ),
    (
        "near_duplicates",
        "Volkswagen CEO announces major restructuring and job cuts",
        [
            "Volkswagen CEO announces major restructuring and job cuts",
            "Volkswagen to cut jobs as part of CEO's restructuring plan",
        ],
    ),
]


def make_request(case_id, main_content, similar_contents):
    def post(pid: int, content: str):
        return PostData(
            id=pid,
            content=content,
            source="test",
            createdAt="2026-06-29T00:00:00Z",
            hash=f"hash-{case_id}-{pid}",
            uuid=f"uuid-{case_id}-{pid}",
        )

    similar_posts = [
        post(i + 1, content)
        for i, content in enumerate(similar_contents)
    ]
    return post(0, main_content).model_copy(update={"similarPosts": similar_posts})


async def main():
    config = {
        "FASTTEXT_MODEL_PATH": "lid.176.bin",
        "FASTTEXT_MODEL_URL": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        "DEFAULT_CLASSIFICATION_LABELS": [],
        "MIN_SCORE": 0.30,
        "MIN_MARGIN": 0.10,
        "POOL_WORKERS": 4,
        "TIMEOUT": 10,
        "HF_CACHE": os.path.expanduser("~/.hf_models"),
    }
    print("Loading models...")
    (
        fasttext_model,
        classifier,
        translator,
        embedder,
        nlp,
        translator_model_name,
    ) = initialize_models(config)

    if embedder is None or nlp is None or classifier is None:
        logger.error("Required models not loaded")
        return

    app_state = SimpleNamespace(
        fasttext_model=fasttext_model,
        classifier=classifier,
        translator=translator,
        embedder=embedder,
        nlp=nlp,
        translator_model_name=translator_model_name,
        clustering_lock=asyncio.Lock(),
        thread_pool=None,
        config=config,
    )
    fake_request = SimpleNamespace(state=SimpleNamespace(app_state=app_state))

    print("\n" + "=" * 80)
    print("Clustering diagnostic")
    print("=" * 80)

    for case_id, main, similars in CASES:
        req = make_request(case_id, main, similars)
        print(f"\n--- Case: {case_id} ---")
        print(f"Main: {main}")
        print(f"Similar posts: {len(similars)}")
        for i, s in enumerate(similars, 1):
            print(f"  {i}. {s}")

        start = time.time()
        try:
            resp = await cluster_texts(req, fake_request)
            elapsed = time.time() - start
            print(f"\nResult: group_size={resp.group.size}, total_groups={resp.total_groups}, time={elapsed:.2f}s")
            print(f"Primary topic: {resp.group.primary_topic}")
            print(f"Primary entities: {resp.group.primary_entities}")
            print(f"Avg similarity: {resp.group.avg_similarity:.3f}")
            print("Clustered posts:")
            for p in resp.group.posts:
                mark = "MAIN" if p.id == f"{case_id}-main" else "    "
                print(f"  [{mark}] {p.id}: {p.content[:100]}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
