"""Text normalization and labeling rules.

These two functions define what "correct" means for every number in the project,
so they are pinned by tests: a silent change here would invalidate old runs.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers  # noqa: F401  (puts pipeline/ on sys.path)

from normalize import normalize
from label import label_row, filtered_labels, question_phrase_hit


class NormalizeTest(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalize("Hello, World!"), "hello world")

    def test_drops_articles(self):
        self.assertEqual(normalize("The quick brown fox"), "quick brown fox")
        self.assertEqual(normalize("a cat and an owl"), "cat and owl")

    def test_none_and_empty(self):
        self.assertEqual(normalize(None), "")
        self.assertEqual(normalize("   "), "")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize("one   two\n\tthree"), "one two three")


class LabelRowTest(unittest.TestCase):
    def row(self, **kw):
        base = {
            "status": "ok",
            "generated": True,
            "truncated": False,
            "answer": "Paris",
            "gold": ["Paris", "paris"],
            "pos_a": 10,
            "pos_c": 12,
        }
        base.update(kw)
        return base

    def test_strict_match(self):
        strict, relaxed, n_tok, matched, _ = label_row(self.row())
        self.assertEqual(strict, 1)
        self.assertEqual(relaxed, 1)
        self.assertEqual(n_tok, 2)
        self.assertEqual(matched, "paris")

    def test_no_match(self):
        strict, relaxed, _, _, _ = label_row(self.row(answer="London"))
        self.assertEqual((strict, relaxed), (0, 0))

    def test_truncated_rows_are_unlabeled(self):
        strict, relaxed, _, _, _ = label_row(self.row(truncated=True))
        self.assertIsNone(strict)
        self.assertIsNone(relaxed)

    def test_skipped_rows_are_unlabeled(self):
        strict, _, _, _, _ = label_row(self.row(status="skipped"))
        self.assertIsNone(strict)

    def test_empty_answer_is_unlabeled(self):
        strict, _, _, _, _ = label_row(self.row(answer="   "))
        self.assertIsNone(strict)

    def test_article_only_answer_does_not_match_a_different_gold(self):
        # normalize() empties "the", so label.py falls back to a version that keeps
        # articles (label.py:47-52). A bare article therefore stays a real answer
        # that simply does not match "Paris" - it is 0, not null.
        strict, relaxed, _, _, _ = label_row(self.row(answer="the"))
        self.assertEqual((strict, relaxed), (0, 0))


class FilteredLabelTest(unittest.TestCase):
    def test_answer_in_question_is_filtered_for_a_and_b(self):
        # the matched phrase is in the question and is not the main gold answer:
        # both filtered labels drop it.
        fa, fb = filtered_labels(1, "paris", "seine", normalize("In Paris, which river?"))
        self.assertEqual((fa, fb), (0, 0))

    def test_b_keeps_the_main_gold_even_if_it_is_in_the_question(self):
        # matched phrase is in the question, but it is the main gold answer:
        # filtered_b keeps it, filtered_a does not.
        fa, fb = filtered_labels(1, "paris", "Paris", normalize("is paris in france"))
        self.assertEqual((fa, fb), (0, 1))

    def test_unfiltered_match(self):
        fa, fb = filtered_labels(1, "paris", "paris", normalize("Which city?"))
        self.assertEqual((fa, fb), (1, 1))

    def test_wrong_answer_stays_zero(self):
        fa, fb = filtered_labels(0, "", "paris", normalize("Which city?"))
        self.assertEqual((fa, fb), (0, 0))

    def test_null_label_propagates(self):
        self.assertEqual(filtered_labels(None, "", "paris", "q"), (None, None))

    def test_phrase_must_be_contiguous(self):
        self.assertTrue(question_phrase_hit("in paris which river", "paris"))
        self.assertFalse(question_phrase_hit("paris is in france", "in paris"))


if __name__ == "__main__":
    unittest.main()
