"""Analytical ROC cases, tied thresholds, and classification summaries."""
import unittest

import numpy as np

from src.mlp.metrics import binary_roc_curve, classification_report


class ROCTests(unittest.TestCase):
    def test_known_rank_auc(self) -> None:
        fpr, tpr, auc = binary_roc_curve(np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8]))
        self.assertEqual(auc, .75)
        self.assertEqual((fpr[0], tpr[0], fpr[-1], tpr[-1]), (0, 0, 1, 1))
        self.assertTrue(np.all(np.diff(fpr) >= 0))
        self.assertTrue(np.all(np.diff(tpr) >= 0))

    def test_ties_are_order_independent(self) -> None:
        for y in ([0, 0, 1, 1], [1, 0, 1, 0]):
            self.assertEqual(binary_roc_curve(np.array(y), np.ones(4))[2], .5)

    def test_perfect_and_reversed_ranking(self) -> None:
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(binary_roc_curve(labels, np.array([0., 0., 1., 1.]))[2], 1)
        self.assertEqual(binary_roc_curve(labels, np.array([1., 1., 0., 0.]))[2], 0)

    def test_invalid_roc_inputs(self) -> None:
        for labels, scores in (([1, 1], [0., 1.]), ([0, 2], [0., 1.]), ([0, 1], [0., np.nan])):
            with self.assertRaises(ValueError):
                binary_roc_curve(np.array(labels), np.array(scores))

    def test_hand_calculated_report(self) -> None:
        probabilities = np.array([[.9, .1], [.4, .6], [.2, .8], [.7, .3]])
        report = classification_report(probabilities, np.array([0, 1, 1, 1]))
        self.assertEqual(report["correct"], 3)
        self.assertEqual(report["errors"], 1)
        self.assertEqual(report["confusion_matrix"], [[1, 0], [1, 2]])
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + .8) / 2)
        self.assertEqual(report["roc_auc_macro_ovr"], 1)
        low, high = report["accuracy_wilson_95"]
        self.assertLess(low, .75)
        self.assertGreater(high, .75)

    def test_missing_class_auc_is_explicitly_undefined(self) -> None:
        report = classification_report(np.array([[.7, .3], [.8, .2]]), np.array([0, 0]))
        self.assertIsNone(report["roc_auc_macro_ovr"])
        self.assertIsNone(report["per_class"][1]["roc_auc_ovr"])


if __name__ == "__main__":
    unittest.main()
