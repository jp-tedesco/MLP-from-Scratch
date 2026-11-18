# Stabilize scores, then normalize each image's row.
shifted = logits - np.max(logits, axis=1, keepdims=True)
scores = np.exp(shifted)
return scores / np.sum(scores, axis=1, keepdims=True)
