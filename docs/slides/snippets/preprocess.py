# One row per image; one column per pixel.
x = images.reshape(len(images), 784).astype(np.float32)
# Scale pixel brightness to the range [0, 1].
x /= np.float32(255.0)
