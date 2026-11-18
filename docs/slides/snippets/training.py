# Visit one mini-batch at a time.
for start in range(0, len(order), self.batch_size):
    indices = order[start:start + self.batch_size]
    logits = self.model.forward(x_train[indices], training=True)
    criterion.forward(logits, y_train[indices])
    # Send gradients backward, then update the weights.
    self.model.backward(criterion.backward())
    self.optimizer.step(self.model.parameters())
