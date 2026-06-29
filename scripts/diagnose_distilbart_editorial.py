"""
Diagnostic script for DistilBART editorial scoring.

Loads the DistilBART zero-shot classifier and prints the full probability
distribution and resulting continuous urgency/newsworthiness scores for a
set of sample posts. This demonstrates that scores are no longer hard 0/10
buckets and that rule-based urgency signals can override low classifier
urgency for public-safety incidents.

Run:
    uv run python scripts/diagnose_distilbart_editorial.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from src.endpoints.analysis import (
    EDITORIAL_PRIORITY_LABELS,
    EDITORIAL_SCORE_MAP,
    _compute_urgency_signals,
    _final_urgency,
    _scores_from_editorial_result,
)
from src.models import ClassificationResponse

SAMPLES = [
    ("breaking", "BREAKING NEWS: M6.1 quake jolts northeastern Japan, no tsunami warning issued"),
    ("important", "Top Boy actor Micheal Ward raped woman in car, court told"),
    ("routine", "Volkswagen CEO targets power shift alongside deep cuts - Reuters"),
    ("niche", "Amanda Batula Won't Be a Part of 'Summer House' Season 11"),
    ("not newsworthy", "Local bake sale announced for next Saturday"),
    ("public_safety", "Manhattan hotel evacuated after mace sprayed inside, FDNY says"),
    ("sports", "Brazil break Japan hearts with last-gasp winner. How did Carlo Ancelotti do it?"),
    ("policy", "Adolescent criminal responsibility: Arrau values the proposal of the Alejandro Law after a fatal lockdown in San Bernardo"),
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


def main():
    classifier = load_classifier()

    print("\n" + "=" * 80)
    print("DistilBART editorial scoring diagnostic")
    print("=" * 80)

    for sample_label, text in SAMPLES:
        result = classifier(text, candidate_labels=EDITORIAL_PRIORITY_LABELS)
        class_resp = ClassificationResponse(
            label=result["labels"][0],
            score=result["scores"][0],
            full_result=result,
        )
        newsworthiness, base_urgency, _ = _scores_from_editorial_result(class_resp)
        urgency = _final_urgency(text, base_urgency)
        signal_boost = _compute_urgency_signals(text)

        print(f"\n[{sample_label}] {text[:80]!r}")
        print(f"  newsworthiness={newsworthiness}, base_urgency={base_urgency}, signal_boost={signal_boost}, final_urgency={urgency}, wake={urgency >= 8.0}")
        for label, score in zip(result["labels"], result["scores"]):
            print(f"    - {label:40}: {score:.4f}")


if __name__ == "__main__":
    main()
