"""
Diagnostic script for DistilBART editorial scoring.

Loads the DistilBART zero-shot classifier and prints the full probability
distribution and resulting continuous urgency/newsworthiness scores for a
set of sample posts. This demonstrates that scores are no longer hard 0/10
buckets.

Run:
    uv run python scripts/diagnose_distilbart_editorial.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


EDITORIAL_PRIORITY_LABELS = [
    "major breaking story",
    "important story but not breaking",
    "routine publishable update",
    "niche or low-priority item",
    "not newsworthy",
]

EDITORIAL_SCORE_MAP = {
    "major breaking story": (10.0, 10.0),
    "important story but not breaking": (8.5, 5.0),
    "routine publishable update": (6.0, 1.5),
    "niche or low-priority item": (2.5, 0.5),
    "not newsworthy": (0.0, 0.0),
}

SAMPLES = [
    ("breaking", "BREAKING NEWS: M6.1 quake jolts northeastern Japan, no tsunami warning issued"),
    ("important", "Top Boy actor Micheal Ward raped woman in car, court told"),
    ("routine", "Volkswagen CEO targets power shift alongside deep cuts - Reuters"),
    ("niche", "Amanda Batula Won't Be a Part of 'Summer House' Season 11"),
    ("not newsworthy", "Local bake sale announced for next Saturday"),
]


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


def scores_from_distribution(labels, scores):
    """Mirror the weighted scoring logic in src/endpoints/analysis.py."""
    weighted_newsworthiness = 0.0
    weighted_urgency = 0.0
    total_weight = 0.0

    for label, score in zip(labels, scores):
        newsworthiness_weight, urgency_weight = EDITORIAL_SCORE_MAP.get(label, (0.0, 0.0))
        weight = float(score) ** 1.5
        weighted_newsworthiness += newsworthiness_weight * weight
        weighted_urgency += urgency_weight * weight
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0, 0.0

    return (
        round(weighted_newsworthiness / total_weight, 2),
        round(weighted_urgency / total_weight, 2),
    )


def main():
    classifier = load_classifier()

    print("\n" + "=" * 80)
    print("DistilBART editorial scoring diagnostic")
    print("=" * 80)

    for sample_label, text in SAMPLES:
        result = classifier(text, candidate_labels=EDITORIAL_PRIORITY_LABELS)
        labels = result["labels"]
        scores = result["scores"]
        newsworthiness, urgency = scores_from_distribution(labels, scores)

        print(f"\n[{sample_label}] {text[:80]!r}")
        print(f"  newsworthiness={newsworthiness}, urgency={urgency}, wake={urgency >= 8.0}")
        for label, score in zip(labels, scores):
            print(f"    - {label:40}: {score:.4f}")


if __name__ == "__main__":
    main()
