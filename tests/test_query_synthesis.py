"""Tests for CelebA query synthesis."""

from __future__ import annotations

import random
from pathlib import Path

from poi.data import CelebASample, synthesize_query


def _make_sample(attrs: dict[str, bool]) -> CelebASample:
    return CelebASample(
        filename="000001.jpg",
        path=Path("/fake/000001.jpg"),
        attributes=attrs,
        split="train",
    )


class TestSynthesizeQuery:
    def test_uses_only_positive_attributes(self) -> None:
        sample = _make_sample({"Smiling": True, "Eyeglasses": False, "Wearing_Hat": True})
        rng = random.Random(0)
        # Several runs to check no negative attribute leaks in
        for _ in range(20):
            q = synthesize_query(sample, n_attributes=3, rng=rng)
            assert "wearing glasses" not in q  # negative attribute

    def test_handles_no_attributes(self) -> None:
        sample = _make_sample({})
        q = synthesize_query(sample, n_attributes=3, rng=random.Random(0))
        assert q == "A person."

    def test_handles_few_attributes(self) -> None:
        sample = _make_sample({"Smiling": True})
        q = synthesize_query(sample, n_attributes=3, rng=random.Random(0))
        assert "smiling" in q

    def test_deterministic_with_seed(self) -> None:
        sample = _make_sample(
            {
                "Smiling": True,
                "Eyeglasses": True,
                "Black_Hair": True,
                "Wearing_Hat": True,
                "Bangs": True,
            }
        )
        q1 = synthesize_query(sample, n_attributes=2, rng=random.Random(123))
        q2 = synthesize_query(sample, n_attributes=2, rng=random.Random(123))
        assert q1 == q2

    def test_excludes_demographic_attrs(self) -> None:
        # Even if upstream sets these, our curated list filters them out.
        sample = _make_sample({"Young": True, "Male": True, "Attractive": True, "Smiling": True})
        rng = random.Random(0)
        for _ in range(10):
            q = synthesize_query(sample, n_attributes=3, rng=rng)
            # Should never appear in a query — they're not in _VISUAL_ATTRIBUTES
            assert "young" not in q.lower()
            assert "attractive" not in q.lower()

    def test_grammar_for_one_two_three_phrases(self) -> None:
        attrs1 = {"Smiling": True}
        attrs2 = {"Smiling": True, "Eyeglasses": True}
        attrs3 = {"Smiling": True, "Eyeglasses": True, "Wearing_Hat": True}

        rng = random.Random(0)
        q1 = synthesize_query(_make_sample(attrs1), n_attributes=1, rng=rng)
        q2 = synthesize_query(_make_sample(attrs2), n_attributes=2, rng=rng)
        q3 = synthesize_query(_make_sample(attrs3), n_attributes=3, rng=rng)

        # 1 phrase has no joiner
        assert " and " not in q1
        # 2 phrases joined by 'and'
        assert " and " in q2
        assert ", and " not in q2
        # 3 phrases use Oxford comma
        assert ", and " in q3
