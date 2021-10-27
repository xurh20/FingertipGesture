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


def identity(pos: tuple[float, float]):
    return pos


def key_transform(pos: tuple[float, float]):
    return identity(pos)


def distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


def minimum_jerk(path, word: str):
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
    keys = [key_transform(STD_KB_POS[c]) for c in word]
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


def generate_standard_pattern(word: str, num_pieces: int):
    """
    description
    ---------
    generate **standard** pattern for the given word with num_pieces
    
    param
    -------
    word: target word
    num_pieces: number of pieces
    
    Returns
    -------
    pattern: array of coordinates
    
    """
    nodes = []
    pattern = []
    for c in word:
        nodes.append(key_transform(STD_KB_POS[c]))
    total_len = 0
    for i in range(len(nodes) - 1):
        total_len += distance(nodes[i], nodes[i + 1])
    used_pieces = 0
    for i in range(len(nodes) - 1):
        if i == len(nodes) - 1:
            p1 = total_len - used_pieces
        else:
            d1 = distance(nodes[i], nodes[i + 1])
            p1 = int(d1 * num_pieces / total_len)
        delta_x = (nodes[i + 1][0] - nodes[i][0]) / p1
        delta_y = (nodes[i + 1][1] - nodes[i][1]) / p1
        for j in range(0, p1):
            pattern.append(
                (nodes[i][0] + delta_x * j, nodes[i][1] + delta_y * j))
        used_pieces += p1
    pattern.append(nodes[-1])
    return pattern


def location_distance(path, pattern):
    """
    description
    ---------
    compute location distance of the user path and pattern
    
    param
    -------
    path: user path
    pattern: standard pattern
    
    Returns
    -------
    distance: float
    
    """
    assert len(path) == len(pattern)
    r = 1
    weights = [1 / len(path) for i in range(len(path))]
    d1 = 0
    for i, p in enumerate(path):
        min_d = Inf
        for p1 in pattern:
            min_d = min(min_d, distance(p, p1))
        d1 += max(min_d - r, 0)
    d2 = 0
    for i, p in enumerate(pattern):
        min_d = Inf
        for p1 in path:
            min_d = min(min_d, distance(p, p1))
        d2 += max(min_d - r, 0)
    ld = 0
    if d1 != 0 or d2 != 0:
        for i in range(len(path)):
            ld += weights[i] * distance(path[i], pattern[i])
    return ld


def get_user_path():
    pass


def main():
    pass


if __name__ == "__main__":
    main()