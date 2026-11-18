# Split the training pool; keep the official test separate.
train_indices, val_indices = _split_indices(y, validation_size, seed)
