"""Language-detector selection and label-space alignment.

The detector (fastText) and the translator (Seamless M4T v2) speak different
label spaces. `lid.176` emits ~176 two-letter labels. GlotLID-v3 emits 2,000+
`<iso639-3>_<Script>` labels (for example `uzn_Cyrl`). Whatever the detector
emits must end in one of three explicit outcomes, so a confident detection of
a language the translator cannot handle can never silently pick a wrong
fallback:

* ``supported``: the normalised code is a language the translator accepts;
* ``relative``: the language has no translator support but maps to a mutually
  intelligible supported relative (``UNSUPPORTED_TRANSLATION_FALLBACKS``);
* ``declined``: neither; the caller keeps the original text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LID176 = "lid176"
GLOTLID = "glotlid"
DETECTORS = (LID176, GLOTLID)

GLOTLID_MODEL_URL = "https://huggingface.co/cis-lmu/glotlid/resolve/main/model_v3.bin"
GLOTLID_DEFAULT_PATH = "glotlid_v3.bin"

# Candidates requested from a script-suffixed model before merging, so that
# folding `srp_Cyrl` and `srp_Latn` into one language cannot starve top-k.
_GLOTLID_OVERFETCH = 3


def detector_settings(name: str, env: dict) -> tuple[str, str]:
    """Return ``(model_path, model_url)`` for a detector name; unknown names are an error."""
    if name == LID176:
        return env.get("FASTTEXT_MODEL_PATH", "lid.176.bin"), "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    if name == GLOTLID:
        return env.get("GLOTLID_MODEL_PATH", GLOTLID_DEFAULT_PATH), GLOTLID_MODEL_URL
    raise ValueError(f"Unknown LANGUAGE_DETECTOR {name!r}; expected one of {', '.join(DETECTORS)}")


def base_language(label: str) -> str:
    """Strip the fastText prefix and any script suffix: ``__label__uzn_Cyrl`` -> ``uzn``."""
    return label.replace("__label__", "").split("_", 1)[0]


@dataclass(frozen=True)
class Resolution:
    kind: Literal["supported", "relative", "declined"]
    language: str


def resolve_label(label: str) -> Resolution:
    """Classify any detector label. Total: every string yields exactly one outcome."""
    from src.endpoints.translation import (
        SUPPORTED_TRANSLATION_LANGUAGES,
        UNSUPPORTED_TRANSLATION_FALLBACKS,
        _normalize_translation_language,
    )

    cleaned = label.strip().lower().replace("__label__", "")
    base = cleaned.split("_", 1)[0].split("-", 1)[0]
    normalised = _normalize_translation_language(label.replace("__label__", "").split("_", 1)[0]) or ""
    if base in UNSUPPORTED_TRANSLATION_FALLBACKS:
        return Resolution("relative", normalised)
    if normalised in SUPPORTED_TRANSLATION_LANGUAGES:
        return Resolution("supported", normalised)
    return Resolution("declined", normalised or base)


class ScriptMergingDetector:
    """Presents a script-suffixed fastText model through the plain `predict` contract.

    Probabilities of labels that differ only by script are summed, so one
    language is not split across several candidates. Output labels keep the
    ``__label__`` prefix, so existing callers need no change.
    """

    def __init__(self, model):
        self._model = model

    def predict(self, text: str, k: int = 1):
        labels, probs = self._model.predict(text, k=max(k, 1) * _GLOTLID_OVERFETCH)
        merged: dict[str, float] = {}
        for label, prob in zip(labels, probs):
            language = base_language(label)
            merged[language] = merged.get(language, 0.0) + float(prob)
        ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)[: max(k, 1)]
        return tuple(f"__label__{language}" for language, _ in ranked), tuple(prob for _, prob in ranked)

    def get_labels(self):
        return self._model.get_labels()
