from random import randint
from abc import abstractmethod
import numpy as np


class AugmentFunction:
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def augment(self, arr):
        pass


class ShiftUp(AugmentFunction):
    def __init__(self) -> None:
        super().__init__("shift_up")

    def augment(self, arr):
        output = []
        offset = randint(1, 5)
        for a in arr:
            out = np.zeros(a.shape)
            out[:-offset] = a[offset:]
            output.append(out)
        return output


class ShiftDown(AugmentFunction):
    def __init__(self) -> None:
        super().__init__("shift_down")

    def augment(self, arr):
        output = []
        offset = randint(1, 5)
        for a in arr:
            out = np.zeros(a.shape)
            out[offset:] = a[:-offset]
            output.append(out)
        return output


class ShiftLeft(AugmentFunction):
    def __init__(self) -> None:
        super().__init__("shift_left")

    def augment(self, arr):
        output = []
        offset = randint(1, 5)
        for a in arr:
            out = np.zeros(a.shape)
            out[:, :-offset] = a[:, offset:]
            output.append(out)
        return output


class ShiftRight(AugmentFunction):
    def __init__(self) -> None:
        super().__init__("shift_right")

    def augment(self, arr):
        output = []
        offset = randint(1, 5)
        for a in arr:
            out = np.zeros(a.shape)
            out[:, offset:] = a[:, :-offset]
            output.append(out)
        return output