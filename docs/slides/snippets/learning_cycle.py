# Predict scores and compare them with the known labels.
logits = self.model.forward(x_train[indices], training=True)
criterion.forward(logits, y_train[indices])
# Compute gradients, then use them to update the weights.
self.model.backward(criterion.backward())
self.optimizer.step(self.model.parameters())
