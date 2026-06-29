"""
Diagnostic script for binary DistilBART zero-shot wake classification.

Run:
    uv run python scripts/diagnose_distilbart_binary.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )


PROMPTS = [
    ("This post is {}.", ["major breaking story", "not urgent"]),
    ("This post is {}.", ["urgent breaking news", "routine news"]),
    ("This post requires {}.", ["waking an editor immediately", "no immediate action"]),
    ("This post is {}.", ["a major breaking story", "not newsworthy"]),
    ("An editor should {} this post.", ["wake up for", "ignore until morning"]),
]

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
    print("DistilBART binary wake classification diagnostic")
    print("=" * 80)

    for template, labels in PROMPTS:
        print(f"\n--- Template: {template} | Labels: {labels} ---")
        classifier.hypothesis_template = template
        for sample_label, text in SAMPLES:
            result = classifier(text, candidate_labels=labels)
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            print(f"  [{sample_label:15}] {top_label} ({top_score:.3f})  |  {text[:60]!r}")


if __name__ == "__main__":
    main()
