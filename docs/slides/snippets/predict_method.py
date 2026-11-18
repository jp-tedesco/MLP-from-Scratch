def predict(self, inputs: NDArray) -> NDArray[np.intp]:
    # Return the highest-score digit for each image.
    return self.forward(inputs).argmax(axis=1)
