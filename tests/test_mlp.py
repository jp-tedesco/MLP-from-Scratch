"""Numerical, training, and persistence checks using only unittest and NumPy."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.mlp.activations import softmax
from src.mlp.losses import CrossEntropyLoss
from src.mlp.metrics import accuracy
from src.mlp.network import MLP
from src.mlp.optimizers import SGD


class NumericalTests(unittest.TestCase):
    """Check stable losses and the complete chain of parameter derivatives."""

    def test_extreme_logits_remain_finite(self) -> None:
        logits = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
        probabilities = softmax(logits)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        loss = CrossEntropyLoss()
        self.assertAlmostEqual(loss.forward(logits, np.array([1, 0])), 2000.0)
        self.assertTrue(np.isfinite(loss.backward()).all())

    def test_uniform_logits_have_expected_loss(self) -> None:
        loss = CrossEntropyLoss()
        value = loss.forward(np.zeros((3, 10)), np.array([0, 4, 9]))
        self.assertAlmostEqual(value, np.log(10))
        np.testing.assert_allclose(loss.backward().sum(axis=1), 0, atol=1e-12)

    def test_complete_network_matches_finite_differences(self) -> None:
        network = MLP((3, 4, 2), seed=7)
        inputs = np.array([[0.2, -0.7, 0.4], [0.8, 0.3, -0.5]], dtype=np.float32)
        targets = np.array([0, 1])
        loss = CrossEntropyLoss()
        loss.forward(network.forward(inputs, training=True), targets)
        network.backward(loss.backward())
        parameters = list(network.parameters())
        gradients = [parameter.grad.copy() for parameter in parameters]
        epsilon = 1e-3
        for parameter, gradient in zip(parameters, gradients):
            for index in np.ndindex(parameter.value.shape):
                original = float(parameter.value[index])
                parameter.value[index] = original + epsilon
                plus = loss.forward(network.forward(inputs), targets)
                parameter.value[index] = original - epsilon
                minus = loss.forward(network.forward(inputs), targets)
                parameter.value[index] = original
                numerical = (plus - minus) / (2 * epsilon)
                self.assertAlmostEqual(
                    float(gradient[index]), numerical, delta=2e-3,
                    msg=f"Gradient mismatch at parameter index {index}",
                )


class IntegrationTests(unittest.TestCase):
    """Exercise learning and the inference checkpoint format end to end."""

    def test_sgd_learns_separable_examples(self) -> None:
        network = MLP((2, 8, 2), seed=3)
        optimizer = SGD(learning_rate=0.1)
        loss = CrossEntropyLoss()
        inputs = np.array([[-1, -1], [-1, -0.5], [1, 1], [1, 0.5]], dtype=np.float32)
        targets = np.array([0, 0, 1, 1])
        initial = loss.forward(network.forward(inputs), targets)
        for _ in range(150):
            loss.forward(network.forward(inputs, training=True), targets)
            network.backward(loss.backward())
            optimizer.step(network.parameters())
        logits = network.forward(inputs)
        self.assertLess(loss.forward(logits, targets), initial / 5)
        self.assertEqual(accuracy(logits, targets), 1.0)

    def test_checkpoint_round_trip_preserves_predictions(self) -> None:
        network = MLP((3, 5, 2), seed=42)
        inputs = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.npz"
            network.save(checkpoint)
            restored = MLP.load(checkpoint)
            np.testing.assert_array_equal(network.forward(inputs), restored.forward(inputs))

    def test_invalid_input_dimensions_are_rejected(self) -> None:
        network = MLP((3, 2), seed=42)
        with self.assertRaises(ValueError):
            network.forward(np.zeros((2, 4), dtype=np.float32))

    def test_invalid_learning_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SGD(learning_rate=0)

    def test_invalid_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CrossEntropyLoss().forward(np.zeros((2, 3)), np.array([0, 3]))


if __name__ == "__main__":
    unittest.main()
