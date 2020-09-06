import numpy as np

class Label_transformer:
    def __init__(self, Y, r=1.5):
        self.mean = np.mean(Y)
        self.std = np.std(Y)
        self.r = r

    def transform(self, Y):
        return np.clip((Y-self.mean)/self.std/self.r, -1.0, 1.0)

    def save():
        np.save("model/label", np.array([self.mean, self.std]))

    def load():
        self.mean, self.std = np.load("model/label")