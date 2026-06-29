"""
Diagnostic script for DistilBART zero-shot editorial classification.

Run:
    uv run python scripts/diagnose_distilbart.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.endpoints.analysis import EDITORIAL_PRIORITY_LABELS, EDITORIAL_SCORE_MAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_classifier():
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    logger.info("Loading DistilBART zero-shot classifier (device=%s)", device)
    return pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-3",
        device=device,
        hypothesis_template="This post is {}.",
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )


SAMPLES = [
    ("breaking", "BREAKING NEWS: M6.1 quake jolts northeastern Japan, no tsunami warning issued"),
    ("important", "Top Boy actor Micheal Ward raped woman in car, court told"),
    ("routine", "Volkswagen CEO targets power shift alongside deep cuts - Reuters"),
    ("niche", "Amanda Batula Won't Be a Part of 'Summer House' Season 11"),
    ("not newsworthy", "Local bake sale announced for next Saturday"),
]


def main():
    classifier = load_classifier()

    print("\n" + "=" * 80)
    print("DistilBART zero-shot editorial classification diagnostic")
    print("=" * 80)

    for sample_label, text in SAMPLES:
        result = classifier(text, candidate_labels=EDITORIAL_PRIORITY_LABELS)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        n, u = EDITORIAL_SCORE_MAP[top_label]
        print(f"\n[{sample_label}] {text[:80]!r}")
        print(f"  Top label: {top_label} (score={top_score:.3f})")
        print(f"  Urgency: {u}  Newsworthiness: {n}")
        print(f"  All scores: {dict(zip(result['labels'], [round(s, 3) for s in result['scores']]))}")


if __name__ == "__main__":
    main()
