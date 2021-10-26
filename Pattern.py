import numpy as np
import math
from numpy.core.numeric import Inf

STD_KB_WIDTH = 1083
STD_KB_HEIGHT = 351
STD_KB_POS = {
    'q': (-474, 105),
    'w': (-370, 105),
    'e': (-265, 105),
    'r': (-161, 105),
    't': (-52, 105),
    'y': (51, 105),
    'u': (156, 105),
    'i': (262, 105),
    'o': (367, 105),
    'p': (469, 105),
    'a': (-446, 0),
    's': (-340, 0),
    'd': (-235, 0),
    'f': (-131, 0),
    'g': (-28, 0),
    'h': (78, 0),
    'j': (184, 0),
    'k': (292, 0),
    'l': (398, 0),
    'z': (-400, -105),
    'x': (-293, -105),
    'c': (-187, -105),
    'v': (-82, -105),
    'b': (23, -105),
    'n': (127, -105),
    'm': (232, -105)
}


def distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


def minimum_jerk(path, word):
    """
    description
    ---------
    uses mininum jerk algorithm to make segemnts of given path
    
    param
    -------
    path: array of (x, y)
    word: string of the word
    
    Returns
    -------
    segments: array of segments with coordinates
    
    """
    if len(path) < 5:
        print("too short for minimum jerk")
        return [i for i in range(len(path))]
    points: list = path[2:-2]
    keys = [STD_KB_POS[c] for c in word]
    segments = []
    for k in range(len(keys) - 1):
        if keys[k] == keys[k + 1]:
            continue
        segment = []
        d = Inf
        for p in points:
            d1 = distance(p, keys[k])
            if d1 < d or d1 < distance(p, keys[k + 1]):
                segment.append(p)
                d = d1
            else:
                break
    for p in segment:
        points.remove(p)
    segments.append(segment)

    segments[0].insert(0, path[0])
    segments[0].insert(1, path[1])

    points.append(path[-2])
    points.append(path[-1])
    segments.append(points)

    return segments


def generate_standard_pattern():
    pass


def compute_distance():
    pass


def get_user_path():
    pass


def main():
    pass


if __name__ == "__main__":
    main()