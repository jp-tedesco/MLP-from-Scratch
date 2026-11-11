"""Integration checks for metric aggregation and reproducible training."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.mlp.losses import CrossEntropyLoss
from src.mlp.metrics import accuracy
from src.mlp.network import MLP
from src.mlp.optimizers import SGD
from src.mlp.trainer import Trainer, evaluate


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.random.default_rng(7).normal(size=(13, 3)).astype(np.float32)
        self.y = (self.x[:, 0] > 0).astype(np.int64)

    def test_evaluation_weights_partial_batches_by_sample_count(self) -> None:
        model = MLP((3, 5, 2), seed=8)
        logits = model.forward(self.x)
        expected_loss = CrossEntropyLoss().forward(logits, self.y)
        measured = evaluate(model, self.x, self.y, batch_size=5)
        self.assertAlmostEqual(measured["loss"], expected_loss, places=6)
        self.assertAlmostEqual(measured["accuracy"], accuracy(logits, self.y))

    def test_training_is_reproducible_and_saves_selected_epoch(self) -> None:
        histories = []
        models = []
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.npz"
            for _ in range(2):
                model = MLP((3, 5, 2), seed=2)
                trainer = Trainer(model, SGD(0.05), batch_size=5, seed=9)
                history = trainer.fit(self.x, self.y, self.x, self.y,
                                      epochs=4, checkpoint=checkpoint)
                histories.append([{k: v for k, v in row.items() if k != "seconds"}
                                  for row in history])
                models.append(model)
                losses = [row["val_loss"] for row in history]
                self.assertEqual(trainer.best_epoch, int(np.argmin(losses)) + 1)
                saved_loss = evaluate(MLP.load(checkpoint), self.x, self.y)["loss"]
                self.assertAlmostEqual(saved_loss, min(losses), places=6)
            self.assertEqual(histories[0], histories[1])
            for first, second in zip(models[0].parameters(), models[1].parameters()):
                np.testing.assert_array_equal(first.value, second.value)

    def test_training_includes_final_partial_batch(self) -> None:
        model = MLP((3, 2), seed=2)
        expected = MLP((3, 2), seed=2)
        optimizer = SGD(0.03)
        order = np.random.default_rng(11).permutation(len(self.x))
        criterion = CrossEntropyLoss()
        for start in range(0, len(order), 5):
            indices = order[start:start + 5]
            criterion.forward(expected.forward(self.x[indices], training=True), self.y[indices])
            expected.backward(criterion.backward())
            optimizer.step(expected.parameters())
        Trainer(model, SGD(0.03), batch_size=5, seed=11).fit(
            self.x, self.y, self.x, self.y, epochs=1)
        for actual, reference in zip(model.parameters(), expected.parameters()):
            np.testing.assert_array_equal(actual.value, reference.value)


if __name__ == "__main__":
    unittest.main()
