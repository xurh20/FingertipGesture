import numpy as np
import math, random, json
from numpy.core.numeric import Inf
import matplotlib.pyplot as plt
from queue import PriorityQueue
from tqdm import tqdm, trange
from dtw import dtw
from scipy.special import erf

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


def aggregate(data):
    points_x = []
    points_y = []
    depths = []
    for frame in data:
        sum_force = 0
        x_average = 0
        y_average = 0
        for y_coordinate in range(len(frame)):
            for x_coordinate in range(len(frame[y_coordinate])):
                sum_force += frame[y_coordinate][x_coordinate]
        if (sum_force > 100):
            for y_coordinate in range(len(frame)):
                for x_coordinate in range(len(frame[y_coordinate])):
                    rate = frame[y_coordinate][x_coordinate] / sum_force
                    x_average += rate * x_coordinate
                    y_average += rate * y_coordinate * -1
            points_x.append(x_average)
            points_y.append(y_average)
            depths.append(sum_force)
    # Normalize depths
    if (len(depths) > 0):
        max_depth = max(depths)
        depths = [i / max_depth for i in depths]
    # Merge points
    merge_num = 1
    while merge_num > 0:
        merge_num = 0
        i = 0
        while i < len(points_x) - 1:
            if np.linalg.norm(
                    np.array([points_x[i], points_y[i]]) -
                    np.array([points_x[i + 1], points_y[i + 1]])) < 0.3:
                x = (points_x[i] + points_x[i + 1]) / 2
                y = (points_y[i] + points_y[i + 1]) / 2
                d = (depths[i] + depths[i + 1]) / 2
                del (points_x[i])
                del (points_x[i])
                del (points_y[i])
                del (points_y[i])
                del (depths[i])
                del (depths[i])
                points_x.insert(i, x)
                points_y.insert(i, y)
                depths.insert(i, d)
                merge_num += 1
            else:
                # print(np.linalg.norm(np.array([points_x[i], points_y[i]]) - np.array([points_x[i + 1], points_y[i + 1]])))
                # print([points_x[i], points_y[i]])
                # print([points_x[i + 1], points_y[i + 1]])
                i += 1

    points_x = [points_x[i] - points_x[0] for i in range(len(points_x))]
    points_y = [points_y[i] - points_y[0] for i in range(len(points_y))]
    return points_x, points_y


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


def generate_standard_pattern(word: str, num_nodes: int, distribute,
                              seg_index: list):
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
        delta_x = nodes[i + 1][0] - nodes[i][0]
        delta_y = nodes[i + 1][1] - nodes[i][1]
        for j in range(0, p1):
            pattern.append((nodes[i][0] + delta_x * distribute(j / p1),
                            nodes[i][1] + delta_y * distribute(j / p1)))
        used_pieces += p1
    pattern.append(nodes[-1])
    if len(pattern) != num_nodes:
        print(word, num_nodes)
        raise Exception()
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


def dynamic_time_warping(path, pattern):
    d, cost_matrix, acc_cost_matrix, pair = dtw(path, pattern, dist=distance)
    # plt.imshow(acc_cost_matrix.T,
    #            origin='lower',
    #            cmap='gray',
    #            interpolation='nearest')
    # plt.plot(pair[0], pair[1], 'w')
    # plt.show()
    return d


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


def get_user_path_json(path_dir, sentence, piece):
    try:
        file = open(
            'data/break_point/' + path_dir + '_%d_%d.txt' % (sentence, piece),
            'r')
        lines = file.readlines()
        path = json.loads(lines[0])
        ox = [p[0] for p in path]
        oy = [p[1] for p in path]
    except:
        return None, None
    ox = [ox[i] - ox[0] for i in range(len(ox))]
    oy = [oy[i] - oy[0] for i in range(len(oy))]
    return (ox, oy)


def get_user_path_original(path_dir, sentence, piece):
    try:
        data = np.load('data/alphabeta_data_' + path_dir + "/" +
                       str(sentence) + "_" + str(piece) + ".npy")
        return data
    except:
        return None


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
            # x, y = get_user_path_json(name, i, j)
            word = get_word(i, j)
            if word is not None and x is not None:
                nodes = len(x)
                if nodes == 0:
                    continue
                total += 1
                q = PriorityQueue()
                for w1 in WORD_SET:
                    # location_distance
                    # d = location_distance(
                    #     list(zip(x, y)),
                    #     generate_standard_pattern('g' + w1, nodes))
                    # basic dtw
                    d = dynamic_time_warping(
                        list(zip(x, y)),
                        generate_standard_pattern(
                            'g' + w1 + 'g', nodes,
                            lambda x: -2 * x**3 + 3 * x**2))
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
    i = random.randint(1, 80)
    j = random.randint(1, 5)
    # i, j = 1, 1
    # x, y = get_user_path("data/path_lyh", i, j)
    # x, y = get_user_path_json("lyh", i, j)
    data = get_user_path_original("lyh", i, j)
    if data is None:
        return
    x, y = aggregate(data)
    word = get_word(i, j)
    if word is not None:
        print(word)

        pattern = generate_standard_pattern('g' + word + 'g', len(x),
                                            lambda x: -2 * x**3 + 3 * x**2)
        gx = [seg[0] for seg in pattern]
        gy = [seg[1] for seg in pattern]
        plt.scatter(x, y, c='r')
        plt.scatter(gx, gy, c='g')
        for i in range(len(x)):
            plt.plot([x[i], gx[i]], [y[i], gy[i]], c='b')
        plt.show()
        # dynamic_time_warping(list(zip(x, y)), list(zip(gx, gy)))


def main():
    while True:
        show_difference()
    check_top_k()


if __name__ == "__main__":
    init_word_set()
    main()