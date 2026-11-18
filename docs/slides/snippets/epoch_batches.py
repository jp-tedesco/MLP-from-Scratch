# Shuffle the examples, then visit one mini-batch at a time.
for epoch in range(1, epochs + 1):
    order = self.rng.permutation(len(x_train))
    for start in range(0, len(order), self.batch_size):
        indices = order[start:start + self.batch_size]
