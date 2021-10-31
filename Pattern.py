import numpy as np
import math, random
from numpy.core.numeric import Inf
import matplotlib.pyplot as plt
from queue import PriorityQueue
from tqdm import tqdm, trange

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

WORD_SET = set()


def identity(pos: tuple[float, float]):
    return pos


def linear_rectangle(pos: tuple[float, float]):
    center = (0, 0)
    width = 8
    height = 18
    return (center[0] + pos[0] * width / STD_KB_WIDTH,
            center[1] + pos[1] * height / STD_KB_HEIGHT)


def key_transform(pos: tuple[float, float]):
    return linear_rectangle(pos)


def distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


def minimum_jerk(path, word: str):
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


def generate_standard_pattern(word: str, num_nodes: int):
    nodes = []
    pattern = []
    for i, c in enumerate(word):
        if (i > 0 and word[i] == word[i - 1]):
            continue
        nodes.append(key_transform(STD_KB_POS[c]))
    if len(nodes) == 1:
        return [nodes[0] for i in range(num_nodes)]
    total_len = 0
    for i in range(len(nodes) - 1):
        total_len += distance(nodes[i], nodes[i + 1])
    num_pieces = num_nodes - 1
    used_pieces = 0
    for i in range(len(nodes) - 1):
        if i == len(nodes) - 2:
            p1 = num_pieces - used_pieces
        else:
            d1 = distance(nodes[i], nodes[i + 1])
            p1 = int(d1 * num_pieces / total_len)
        if p1 == 0:
            continue
        delta_x = (nodes[i + 1][0] - nodes[i][0]) / p1
        delta_y = (nodes[i + 1][1] - nodes[i][1]) / p1
        for j in range(0, p1):
            pattern.append(
                (nodes[i][0] + delta_x * j, nodes[i][1] + delta_y * j))
        used_pieces += p1
    pattern.append(nodes[-1])
    assert len(pattern) == num_nodes
    return pattern


def location_distance(path, pattern):
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


def init_word_set():
    with open("data/voc.txt", 'r') as file:
        lines = file.readlines()
        for i in range(1, 80):
            line = lines[i - 1]
            for j in range(1, 12):
                try:
                    WORD_SET.add(line.replace('\n', '').split(' ')[j])
                except:
                    continue


def get_word(sen, wor):
    with open("data/voc.txt", 'r') as file:
        try:
            lines = file.readlines()
            line = lines[sen - 1]
            return line.replace('\n', '').split(' ')[wor]
        except:
            return None


def get_user_path(path_dir, sentence, piece):
    try:
        ox = np.load(path_dir + "/%d_%d_x.npy" % (sentence, piece))
        oy = np.load(path_dir + "/%d_%d_y.npy" % (sentence, piece))
    except:
        return None, None
    ox = [ox[i] - ox[0] for i in range(len(ox))]
    oy = [oy[i] - oy[0] for i in range(len(oy))]
    return (ox, oy)


def draw_mininum_jerk(path_dir, sentence, piece):
    path = list(zip(*get_user_path(path_dir, sentence, piece)))
    word = get_word(sentence, piece)
    segments = minimum_jerk(path, word)
    for segment in segments:
        x = [seg[0] for seg in segment]
        y = [seg[1] for seg in segment]
        c = [
            "#" +
            ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        ]
        plt.scatter(x, y, c=c)
    plt.show()


def check_top_k():
    total = 0
    top_1 = 0
    top_2 = 0
    top_3 = 0
    for i in trange(1, 80):
        for j in trange(1, 12):
            x, y = get_user_path("data/path_lyh", i, j)
            word = get_word(i, j)
            if word is not None and x is not None:
                nodes = len(x)
                total += 1
                q = PriorityQueue()
                for w1 in WORD_SET:
                    d = location_distance(
                        list(zip(x, y)),
                        generate_standard_pattern('g' + w1, nodes))
                    q.put((d, w1))
                top = []
                # top 1
                top.append(q.get()[1])
                if word in top:
                    top_1 += 1
                # top 2
                top.append(q.get()[1])
                if word in top:
                    top_2 += 1
                # top 3
                top.append(q.get()[1])
                if word in top:
                    top_3 += 1
                # print(word, top)
    print("total: %d" % total)
    print("top1 acc: %f" % (top_1 / total))
    print("top2 acc: %f" % (top_2 / total))
    print("top3 acc: %f" % (top_3 / total))


def show_difference():
    plt.axis('scaled')
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    # i = random.randint(1, 80)
    # j = random.randint(1, 5)
    i, j = 1, 1
    x, y = get_user_path("data/path_lyh", i, j)
    word = get_word(i, j)
    if word is not None and x is not None:
        print(word)
        pattern = generate_standard_pattern('g' + word, len(x))
        gx = [seg[0] for seg in pattern]
        gy = [seg[1] for seg in pattern]
        plt.scatter(x, y, c='r')
        plt.scatter(gx, gy, c='g')
        plt.show()


def main():
    while True:
        show_difference()
    # check_top_k()


if __name__ == "__main__":
    init_word_set()
    main()