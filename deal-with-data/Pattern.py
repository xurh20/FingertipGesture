import numpy as np
import math, random, json
from numpy.core.numeric import Inf
import matplotlib.pyplot as plt
from queue import PriorityQueue
from numpy.linalg.linalg import norm
from tqdm import tqdm, trange
from dtw import dtw
from scipy.special import erf
from match import genVectors, genPattern, a_dtw, genPointLabels
from cleanWords import clean
from scipy.spatial.distance import cdist
from math import sqrt
from fastdtw import fastdtw
from multiprocessing import Pool, Manager
from functools import partial, wraps

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

ANA_KB_WIDTH = 8
ANA_KB_HEIGHT = 8
ANA_KB_POS = {
    'a': (-3.60647363, -0.63834205),
    'b': (-0.36606605, -5.12443623),
    'c': (-1.47231184, -4.23960535),
    'd': (-1.9210401, -0.62426094),
    'e': (-2.39354782, 2.39488218),
    'f': (-1.79432936, -0.88401021),
    'g': (-1.43803671, 0.01432068),
    'h': (0.860294, -1.6729026),
    'i': (2.03130594, 2.3622898),
    'j': (1.00236296, -0.75739284),
    'k': (1.90240149, -0.31155476),
    'l': (2.49140859, -0.97244246),
    'm': (1.56049504, -4.53376199),
    'n': (0.45920625, -4.39645984),
    'o': (2.61070786, 1.83023801),
    'p': (3.07807985, 1.60002022),
    'q': (-3.61720778, 1.77148202),
    'r': (-0.8815241, 2.25594157),
    's': (-2.83912263, -0.91653282),
    't': (-0.84827485, 2.5858369),
    'u': (0.35521257, 2.05379052),
    'v': (-0.71334268, -4.39406909),
    'w': (-2.86616848, 1.76756366),
    'x': (-2.23156079, -3.50692019),
    'y': (-0.18780098, 2.51398159),
    'z': (-3.25208095, -4.35824647)
}

WORD_SET = set()
PERSON = ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]
global SENT_LIMIT, WORD_LIMIT
SENT_LIMIT = 0
WORD_LIMIT = []
global THETA_PATTERN, DIST_PATTERN
THETA_PATTERN = {}
DIST_PATTERN = {}
global WORD_PROB, BIWORD_PROB
WORD_PROB = {}
BIWORD_PROB = {}


def identity(pos: tuple([float, float])):
    return pos


def linear_rectangle(pos: tuple([float, float])):
    # center = (0, 0)
    center = (-1.43803671, 0.01432068)
    width = 8
    height = 8
    return np.array([
        center[0] + pos[0] * width / STD_KB_WIDTH,
        center[1] + pos[1] * height / STD_KB_HEIGHT
    ])


def key_transform(pos: tuple([float, float])):
    return linear_rectangle(pos)


def distance(p: tuple([float, float]), q: tuple([float, float])) -> float:
    return theta_distance(p, q, True)


def theta_distance(p: tuple([float, float]),
                   q: tuple([float, float]),
                   normalized=False) -> float:
    if normalized:
        return -np.array(p).dot(np.array(q)) + 0.9
    return -np.array(p).dot(np.array(q)) / np.linalg.norm(
        np.array(p)) / np.linalg.norm(np.array(q)) + 0.9


def modulus_distance(p: tuple([float, float]), q: tuple([float,
                                                         float])) -> float:
    return np.abs(np.linalg.norm(q) * 0.8 + 4 - np.linalg.norm(p))


def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
    # return np.linalg.norm(p - q)
    # return cdist(p, q)
    return sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


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

    # points_x = [points_x[i] - points_x[0] for i in range(len(points_x))]
    # points_y = [points_y[i] - points_y[0] for i in range(len(points_y))]
    return points_x, points_y, depths


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


def generate_standard_pattern(word: str, num_nodes: int, distribute):
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
        total_len += euclidean_distance(nodes[i], nodes[i + 1])
    num_pieces = num_nodes - 1
    used_pieces = 0
    for i in range(len(nodes) - 1):
        if i == len(nodes) - 2:
            p1 = num_pieces - used_pieces
        else:
            d1 = euclidean_distance(nodes[i], nodes[i + 1])
            p1 = int(d1 * num_pieces / total_len)
        if p1 == 0:
            continue
        delta_x = nodes[i + 1][0] - nodes[i][0]
        delta_y = nodes[i + 1][1] - nodes[i][1]
        for j in range(0, p1):
            pattern.append(
                np.array([
                    nodes[i][0] + delta_x * distribute(j / p1),
                    nodes[i][1] + delta_y * distribute(j / p1)
                ]))
        used_pieces += p1
    pattern.append(nodes[-1])
    if len(pattern) != num_nodes:
        print(word, num_nodes, len(pattern))
        raise Exception()
    return np.array(pattern)


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
    d, cost_matrix, acc_cost_matrix, pair = a_dtw(path, pattern, dist=distance)
    # plt.imshow(acc_cost_matrix.T,
    #            origin='lower',
    #            cmap='gray',
    #            interpolation='nearest')
    # plt.plot(pair[0], pair[1], 'w')
    # plt.show()
    return d, pair


def centerize(x, y):
    x_avg = np.average(np.array(x))
    y_avg = np.average(np.array(y))
    r_max = np.max(
        np.array([
            np.linalg.norm(np.array([x[i] - x_avg, y[i] - y_avg]))
            for i in range(len(x))
        ]))
    xx = [(x[i] - x_avg) / r_max for i in range(len(x))]
    yy = [(y[i] - y_avg) / r_max for i in range(len(y))]

    return xx, yy


def downsample(x, y, depths, step=2):
    pointLabels = genPointLabels(x, y, depths)
    xx = []
    yy = []
    if len(pointLabels) == 1:
        for i in range(0, len(x), step):
            xx.append(x[i])
            yy.append(y[i])
    else:
        for s, t in list(zip(pointLabels, pointLabels[1:])):
            for i in range(s, t, step):
                xx.append(x[i])
                yy.append(y[i])
        xx.append(x[-1])
        yy.append(y[-1])
    return np.array(xx), np.array(yy)


def init_word_set():
    global SENT_LIMIT, WORD_LIMIT, WORD_PROB
    with open("../data/words_10000.txt", 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            word = lines[i].lower().replace('\n', '').split('\t')[0]
            prob = float(lines[i].lower().replace('\n', '').split('\t')[2])
            WORD_SET.add(word)
            WORD_PROB[word] = prob

    with open("../data/voc.txt", 'r') as file:
        lines = file.readlines()
        SENT_LIMIT = len(lines) + 1
        WORD_LIMIT.append(0)
        for i in range(len(lines)):
            line = lines[i].replace('\n', '').split(' ')
            WORD_LIMIT.append(len(line))
            # for j in range(len(line)):
            #     if (line[j] not in WORD_SET):
            #         print(line[j])


def init_pattern_set():
    global THETA_PATTERN, DIST_PATTERN
    for w in WORD_SET:
        THETA_PATTERN[w] = np.array(genPattern(clean('g' + w)))

        dist_path = generate_standard_pattern(
            clean('g' + w), int((len(w) * 6.6457 + 4.2080) / 2),
            lambda x: -2 * x**3 + 3 * x**2)
        dist_path_x = [d[0] for d in dist_path]
        dist_path_y = [d[1] for d in dist_path]
        dist_path_x, dist_path_y = centerize(dist_path_x, dist_path_y)
        DIST_PATTERN[w] = list(zip(dist_path_x, dist_path_y))


def init_language_model():
    global WORD_PROB, BIWORD_PROB
    with open("../data/bigrams-written-katz.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.replace('\n', '').split(' ')
            if l[0] not in BIWORD_PROB:
                BIWORD_PROB[l[0]] = {}
            try:
                BIWORD_PROB[l[0]][l[1]] = float(l[2])
            except:
                continue


def get_word(sen, wor):
    with open("../data/voc.txt", 'r') as file:
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
            '../data/break_point/' + path_dir + '_%d_%d.txt' %
            (sentence, piece), 'r')
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
        data = np.load('../data/alphabeta_data_' + path_dir + "/" +
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


def get_top_k(person, k):
    top_k = [[]]
    for i in trange(1, SENT_LIMIT):
        top_k_i = []
        for j in range(0, WORD_LIMIT[i]):
            data = get_user_path_original(person, i, j)
            if data is None:
                continue
            x, y, depths = aggregate(data)
            user = genVectors(x, y, depths, False)
            word = get_word(i, j)

            if word is not None and x is not None:
                q = PriorityQueue()
                for w1 in WORD_SET:
                    pattern = genPattern(clean('g' + w1), False)
                    if (len(user) <= 0 or len(pattern) <= 0):
                        continue
                    d, cost_matrix, acc_cost_matrix, path = a_dtw(
                        user, pattern, dist=distance)
                    # import matplotlib.pyplot as plt

                    # plt.imshow(acc_cost_matrix.T,
                    #            origin='lower',
                    #            cmap='gray',
                    #            interpolation='nearest')
                    # plt.plot(path[0], path[1], 'w')
                    # plt.show()
                    q.put((d, w1))
                top = []
                for p in range(k):
                    try:
                        tmp = q.get_nowait()
                        top.append(tmp[1])
                        if word in top:
                            top_k[p] += 1
                    except:
                        break
                # print(word, top)
                top_k_i.append(top)
        top_k.append(top_k_i)
    return top_k


def check_top_k(person, k):

    total = 0
    top_k = [0] * k
    for i in trange(2, int(SENT_LIMIT / 2)):
        prev = None
        for j in range(0, WORD_LIMIT[i]):
            data = get_user_path_original(person, i, j)
            if data is None:
                continue
            x, y, depths = aggregate(data)
            user = genVectors(x, y, depths)
            word = get_word(i, j)

            if word in WORD_SET and x is not None:
                total += 1
                x, y = downsample(x, y, depths, 2)
                user_path_x, user_path_y = centerize(x, y)
                user_path = np.array(list(zip(user_path_x, user_path_y)))
                q = PriorityQueue()

                for w in WORD_SET:
                    # theta distance
                    pattern = THETA_PATTERN[w]
                    if (len(user) <= 0 or len(pattern) <= 0):
                        continue
                    d1, cost_matrix, acc_cost_matrix, path = a_dtw(
                        user, pattern, dist=distance)

                    # shape distance
                    pattern = DIST_PATTERN[w]
                    if (len(user_path) <= 0 or len(pattern) <= 0):
                        continue
                    d2, path = fastdtw(user_path,
                                       pattern,
                                       radius=1,
                                       dist=euclidean_distance)

                    # bigram probabilities
                    if prev in BIWORD_PROB and w in BIWORD_PROB[prev]:
                        d = BIWORD_PROB[prev][w]
                    elif w in WORD_PROB:
                        d = WORD_PROB[w]
                    else:
                        d = 10**(-10)

                    q.put((0.6 * d2 - 0.4 * 0.1 * np.log10(d), w))

                top = []
                for p in range(k):
                    try:
                        tmp = q.get_nowait()
                        top.append(tmp[1])
                        if word in top:
                            top_k[p] += 1
                    except:
                        break
                print(word, top[:10])
                input()
            prev = word
    print("total: %d" % total)
    for i in range(k):
        print("top%d acc: %f" % (i + 1, top_k[i] / total))


def show_difference(person):
    k = 50
    plt.axis("scaled")
    for i in trange(2, int(SENT_LIMIT / 2)):
        prev = None
        for j in range(0, WORD_LIMIT[i]):
            data = get_user_path_original(person, i, j)
            if data is None:
                continue
            x, y, depths = aggregate(data)
            user = genVectors(x, y, depths)
            word = get_word(i, j)

            if word in WORD_SET and x is not None:
                x, y = downsample(x, y, depths, 2)
                user_path_x, user_path_y = centerize(x, y)
                user_path = np.array(list(zip(user_path_x, user_path_y)))
                q = PriorityQueue()

                for w in WORD_SET:
                    # theta distance
                    pattern = THETA_PATTERN[w]
                    if (len(user) <= 0 or len(pattern) <= 0):
                        continue
                    d1, cost_matrix, acc_cost_matrix, path = a_dtw(
                        user, pattern, dist=distance)

                    # shape distance
                    pattern = DIST_PATTERN[w]
                    if (len(user_path) <= 0 or len(pattern) <= 0):
                        continue
                    d2, path = fastdtw(user_path,
                                       pattern,
                                       radius=1,
                                       dist=euclidean_distance)

                    # bigram probabilities
                    if prev in BIWORD_PROB and w in BIWORD_PROB[prev]:
                        d = BIWORD_PROB[prev][w]
                    elif w in WORD_PROB:
                        d = WORD_PROB[w]
                    else:
                        d = 10**(-10)

                    q.put((0.6 * d2 - 0.4 * 0.1 * np.log10(d), w))

                top = []
                for p in range(k):
                    try:
                        tmp = q.get_nowait()
                        top.append(tmp[1])
                    except:
                        break
                print(word, top[0])

                # black  - original user path (downsampled)
                # red    - vectors extracted from user path
                # green  - predicted word vectors
                # blue   - correct word vectors
                # yellow - correct word distance pattern
                plt.scatter(user_path_x, user_path_y, c='black')

                # ux, uy = 0, 0
                # uxs, uys = [0], [0]
                # for u in user:
                #     ux += u[0]
                #     uy += u[1]
                #     uxs.append(ux)
                #     uys.append(uy)
                # plt.plot(uxs, uys, color='red')

                # pred = top[0]
                # pred = genPattern(clean('g' + pred), False)
                # ux, uy = 0, 0
                # uxs, uys = [0], [0]
                # for u in pred:
                #     ux += u[0]
                #     uy += u[1]
                #     uxs.append(ux)
                #     uys.append(uy)
                # plt.plot(uxs, uys, color='green')

                # answ = word
                # answ = genPattern(clean('g' + answ), False)
                # ux, uy = 0, 0
                # uxs, uys = [0], [0]
                # for u in answ:
                #     ux += u[0]
                #     uy += u[1]
                #     uxs.append(ux)
                #     uys.append(uy)
                # plt.plot(uxs, uys, color='blue')

                s_pattern = DIST_PATTERN[word]
                sx = [s[0] for s in s_pattern]
                sy = [s[1] for s in s_pattern]
                sx, sy = centerize(sx, sy)
                plt.scatter(sx, sy, color='yellow')

                plt.show()


def cal_len_nodes():
    num_nodes = {}
    for person in PERSON:
        for i in trange(1, SENT_LIMIT):
            for j in range(0, WORD_LIMIT[i]):
                data = get_user_path_original(person, i, j)
                if data is None:
                    continue
                x, y, depths = aggregate(data)
                word = get_word(i, j)
                if len(word) not in num_nodes:
                    num_nodes[len(word)] = []
                num_nodes[len(word)].append(len(x))

    avg_num_nodes = []
    for i in num_nodes.keys():
        avg_num_nodes.append([i, np.average(np.array(num_nodes[i]))])

    import pandas as pd
    data = pd.DataFrame(avg_num_nodes, columns=['x', 'y'])
    from statsmodels.formula.api import ols
    from statsmodels.api import graphics
    formula = 'y ~ x'
    ols_results = ols(formula, data).fit()
    print(ols_results.summary())
    fig = plt.figure(figsize=(15, 8))
    fig = graphics.plot_regress_exog(ols_results, "x", fig=fig)
    plt.show()


def main():
    while True:
        show_difference("qlp")
    # check_top_k("qlp", 50)


if __name__ == "__main__":
    init_word_set()
    # cal_len_nodes()
    init_pattern_set()
    init_language_model()
    print("using set size: ", len(WORD_SET))
    # print(SENT_LIMIT, len(WORD_LIMIT))
    main()
