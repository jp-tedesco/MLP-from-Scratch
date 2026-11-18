# The highest score determines the predicted digit.
return self.forward(inputs).argmax(axis=1)
