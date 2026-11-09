"""Saved probes keep source train scaling and resolve sparse target layer IDs."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

import numpy as np
from apply_probe_model import load_probe_model, score_hidden


class ApplyProbeModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "probe_model.npz")

    def save(self, **changes):
        data = {"meta": np.array(json.dumps({"point": "C", "layer": 13, "layer_id": 13, "lambda": 1000})),
                "w": np.asarray([0.5, 2.0, -1.0]), "mu": np.asarray([10.0, 20.0]), "sd": np.asarray([2.0, 4.0])}
        data.update(changes)
        np.savez(self.path, **data)
        return load_probe_model(self.path)

    def test_sparse_target_layer_uses_saved_train_mu_sd_without_refitting(self):
        probe = self.save()
        hidden = np.asarray([[[12.0, 24.0]], [[8.0, 16.0]]], dtype=np.float16)
        score, index = score_hidden(probe, hidden, [0, 1], [13])
        np.testing.assert_array_equal(score, [1.5, -0.5])
        self.assertEqual(index, 0)
        np.testing.assert_array_equal(probe["mu"], [10.0, 20.0])
        np.testing.assert_array_equal(probe["sd"], [2.0, 4.0])

    def test_layer_id_without_source_array_index_is_supported(self):
        probe = self.save(meta=np.array('{"point":"C","layer_id":13}'))
        score, index = score_hidden(probe, np.ones((1, 2, 2)), [0], [3, 13])
        self.assertEqual(index, 1)
        self.assertEqual(score.shape, (1,))

    def test_missing_or_duplicate_layer_ids_and_wrong_shape_fail(self):
        probe = self.save()
        for hidden, layers in ((np.ones((2, 1, 2)), [12]), (np.ones((2, 2, 2)), [13, 13]),
                               (np.ones((2, 1, 3)), [13]), (np.ones((2, 2)), [13])):
            with self.subTest(shape=hidden.shape, layers=layers):
                with self.assertRaises(ValueError):
                    score_hidden(probe, hidden, [0], layers)

    def test_row_bounds_and_nonfinite_hidden_fail(self):
        probe = self.save()
        hidden = np.ones((2, 1, 2))
        for rows in ([-1], [2], [[0]]):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, "row index"):
                    score_hidden(probe, hidden, rows, [13])
        hidden[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_hidden(probe, hidden, [0], [13])

    def test_invalid_scaling_or_weights_fail_without_pickle(self):
        for changes in ({"sd": np.asarray([0.0, 4.0])}, {"w": np.asarray([0.0, np.inf, 1.0])},
                        {"mu": np.asarray([1.0])}, {"meta": np.array({"point": "C"}, dtype=object)}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.save(**changes)


if __name__ == "__main__":
    unittest.main()
