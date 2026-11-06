from src.helpers.async_wrappers import (
    approximate_token_count,
    seamless_generation_kwargs,
)

JAPANESE_HEADLINE = "政府は新しい経済対策を発表しました"
THAI_HEADLINE = "รัฐบาลประกาศมาตรการเศรษฐกิจใหม่"
SPANISH_HEADLINE = "El gobierno anuncia nuevas medidas economicas"


def test_spaced_text_is_counted_by_words():
    assert approximate_token_count("Government announces new budget") == 4
    assert approximate_token_count(SPANISH_HEADLINE) == 6


def test_unspaced_scripts_are_counted_by_character():
    """`str.split()` returns 1 for these regardless of length."""
    assert len(JAPANESE_HEADLINE.split()) == 1
    assert approximate_token_count(JAPANESE_HEADLINE) == len(JAPANESE_HEADLINE)

    assert len(THAI_HEADLINE.split()) == 1
    assert approximate_token_count(THAI_HEADLINE) == len(THAI_HEADLINE)


def test_unspaced_estimate_grows_with_length():
    """The old whitespace estimate was flat, so long CJK text looked tiny."""
    short = approximate_token_count(JAPANESE_HEADLINE)
    long_text = JAPANESE_HEADLINE * 8
    assert len(long_text.split()) == len(JAPANESE_HEADLINE.split()) == 1
    assert approximate_token_count(long_text) > short * 7


def test_mixed_script_counts_both_halves():
    # Two latin words plus four Japanese characters.
    assert approximate_token_count("Toyota EV 新型発表") == 6


def test_empty_and_whitespace_never_report_zero():
    assert approximate_token_count("") == 1
    assert approximate_token_count("   \n\t ") == 1


def test_seamless_generation_blocks_repeated_trigrams():
    """Regression guard for degenerate loops ("I'm sorry, I'm sorry, ...").

    Beam search with `do_sample=False` and no repetition guard loops whenever the
    decoder's EOS probability stays low, which is common for unspaced scripts.
    """
    kwargs = seamless_generation_kwargs(SPANISH_HEADLINE)
    assert kwargs["no_repeat_ngram_size"] == 3
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 4


def test_seamless_decode_budget_scales_with_the_input():
    """The old flat 96-token floor gave a five-word headline an eight-fold budget."""
    short = seamless_generation_kwargs("Government announces new budget")
    long_text = seamless_generation_kwargs(JAPANESE_HEADLINE * 20)

    assert short["max_new_tokens"] == 48, "floor, not the old 96"
    assert long_text["max_new_tokens"] > short["max_new_tokens"]
    assert long_text["max_new_tokens"] <= 384, "capped"


def test_seamless_decode_budget_tracks_unspaced_script_length():
    """`str.split()` is flat for CJK, so the budget must not be derived from it."""
    one_line = seamless_generation_kwargs(JAPANESE_HEADLINE)
    eight_lines = seamless_generation_kwargs(JAPANESE_HEADLINE * 8)

    assert len(JAPANESE_HEADLINE.split()) == len((JAPANESE_HEADLINE * 8).split()) == 1
    assert eight_lines["max_new_tokens"] > one_line["max_new_tokens"]
