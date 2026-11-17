import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

if helpers.PIPELINE not in sys.path:
    sys.path.insert(0, helpers.PIPELINE)
import answer_token_baselines as atb


class AnswerTokenBaselineTests(unittest.TestCase):
    def test_aggregate_scores(self):
        got = atb.aggregate_scores(
            [-0.2, -2.1, -0.4],
            [1.0, 3.0, 2.0],
            3,
        )
        self.assertAlmostEqual(got["first_logprob"], -0.2)
        self.assertAlmostEqual(got["mean_logprob"], -0.9)
        self.assertAlmostEqual(got["min_logprob"], -2.1)
        self.assertAlmostEqual(got["last_logprob"], -0.4)
        self.assertAlmostEqual(got["first_neg_entropy"], -1.0)
        self.assertAlmostEqual(got["mean_neg_entropy"], -2.0)
        self.assertAlmostEqual(got["last_neg_entropy"], -2.0)

    def test_fixed_buffer_is_sliced_at_point_c(self):
        got = atb.aggregate_scores(
            [-0.2, -0.4, -9.0, -9.0],
            [1.0, 2.0, 99.0, 99.0],
            2,
        )
        self.assertAlmostEqual(got["mean_logprob"], -0.3)
        self.assertAlmostEqual(got["min_logprob"], -0.4)
        self.assertAlmostEqual(got["last_neg_entropy"], -2.0)

    def test_rejects_short_or_nonfinite_score_arrays(self):
        with self.assertRaises(ValueError):
            atb.aggregate_scores([-0.2], [1.0], 2)
        with self.assertRaises(ValueError):
            atb.aggregate_scores([-0.2, np.nan], [1.0, 2.0], 2)


if __name__ == "__main__":
    unittest.main()
