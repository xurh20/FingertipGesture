import numpy as np
import math, random, json
from numpy.core.numeric import Inf
import matplotlib.pyplot as plt
from queue import PriorityQueue
from numpy.linalg.linalg import norm
from tqdm import tqdm, trange
from dtw import dtw
from scipy.special import erf
from match import genVectors, genPattern, a_dtw
from cleanWords import clean

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


def identity(pos: tuple([float, float])):
    return pos


def linear_rectangle(pos: tuple([float, float])):
    # center = (0, 0)
    center = (-1.43803671, 0.01432068)
    width = 8
    height = 8
    return (center[0] + pos[0] * width / ANA_KB_WIDTH,
            center[1] + pos[1] * height / ANA_KB_HEIGHT)


def key_transform(pos: tuple([float, float])):
    return linear_rectangle(pos)


def distance(p: tuple([float, float]), q: tuple([float, float])) -> float:
    return theta_distance(p, q)


def theta_distance(p: tuple([float, float]), q: tuple([float,
                                                       float])) -> float:
    return -np.array(p).dot(np.array(q)) / np.linalg.norm(
        np.array(p)) / np.linalg.norm(np.array(q)) + 0.9


def modulus_distance(p: tuple([float, float]), q: tuple([float,
                                                         float])) -> float:
    return np.abs(np.linalg.norm(q) * 0.8 + 4 - np.linalg.norm(p))


def euclidian_distance(p: tuple([float, float]), q: tuple([float,
                                                           float])) -> float:
    return np.linalg.norm(np.array(p) - np.array(q))


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
        nodes.append(key_transform(ANA_KB_POS[c]))
    if len(nodes) == 1:
        return [nodes[0] for i in range(num_nodes)]
    total_len = 0
    for i in range(len(nodes) - 1):
        total_len += euclidian_distance(nodes[i], nodes[i + 1])
    num_pieces = num_nodes - 1
    used_pieces = 0
    for i in range(len(nodes) - 1):
        if i == len(nodes) - 2:
            p1 = num_pieces - used_pieces
        else:
            d1 = euclidian_distance(nodes[i], nodes[i + 1])
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
        print(word, num_nodes, len(pattern))
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
    d, cost_matrix, acc_cost_matrix, pair = a_dtw(path, pattern, dist=distance)
    # plt.imshow(acc_cost_matrix.T,
    #            origin='lower',
    #            cmap='gray',
    #            interpolation='nearest')
    # plt.plot(pair[0], pair[1], 'w')
    # plt.show()
    return d, pair


def centerize(x, y):
    xx = [x[i] - x[0] for i in range(len(x))]
    yy = [y[i] - y[0] for i in range(len(y))]
    return xx, yy


def init_word_set():
    global SENT_LIMIT, WORD_LIMIT
    with open("../data/phrases2.txt", 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i].lower().replace('\n', '').split(' ')
            for j in range(len(line)):
                try:
                    WORD_SET.add(line[j])
                except:
                    continue
    with open("../data/voc.txt", 'r') as file:
        lines = file.readlines()
        SENT_LIMIT = len(lines) + 1
        WORD_LIMIT.append(0)
        for i in range(len(lines)):
            line = lines[i].replace('\n', '').split(' ')
            WORD_LIMIT.append(len(line))
            for j in range(len(line)):
                if (line[j] not in WORD_SET):
                    print(line[j])


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
    for i in trange(1, SENT_LIMIT):
        for j in range(0, WORD_LIMIT[i]):
            data = get_user_path_original(person, i, j)
            if data is None:
                continue
            x, y, depths = aggregate(data)
            user = genVectors(x, y, depths, False)
            word = get_word(i, j)

            if word is not None and x is not None:
                total += 1
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
                        # if word in top:
                        #     top_k[p] += 1
                    except:
                        break
                # print(word, top)

                q1 = PriorityQueue()
                for w2 in top:
                    path = list(zip(x, y))
                    pattern = generate_standard_pattern(
                        clean('g' + w2), len(path),
                        lambda x: -2 * x**3 + 3 * x**2)
                    if (len(path) <= 0 or len(pattern) <= 0):
                        continue
                    d, cost_matrix, acc_cost_matrix, path = dtw(
                        path, pattern, dist=euclidian_distance)
                    # import matplotlib.pyplot as plt

                    # plt.imshow(acc_cost_matrix.T,
                    #            origin='lower',
                    #            cmap='gray',
                    #            interpolation='nearest')
                    # plt.plot(path[0], path[1], 'w')
                    # plt.show()
                    q1.put((d, w2))
                topp = []
                for p in range(k):
                    try:
                        tmp = q1.get_nowait()
                        topp.append(tmp[1])
                        if word in topp:
                            top_k[p] += 1
                    except:
                        break
                print(word, topp)
    print("total: %d" % total)
    for i in range(k):
        print("top%d acc: %f" % (i + 1, top_k[i] / total))


def show_difference():
    k = 50
    i = random.randint(1, 80)
    j = random.randint(1, 5)
    data = get_user_path_original("qlp", i, j)
    if data is None:
        return
    x, y, depths = aggregate(data)
    user = genVectors(x, y, depths, False)
    word = get_word(i, j)

    if word is not None and x is not None:
        q = PriorityQueue()
        for w1 in WORD_SET:
            pattern = genPattern(clean('g' + w1), False)
            if (len(user) <= 0 or len(pattern) <= 0):
                continue
            d, cost_matrix, acc_cost_matrix, path = a_dtw(user,
                                                          pattern,
                                                          dist=distance)
            # import matplotlib.pyplot as plt

            # plt.imshow(acc_cost_matrix.T,
            #            origin='lower',
            #            cmap='gray',
            #            interpolation='nearest')
            # plt.plot(path[0], path[1], 'w')
            # plt.show()
            q.put((d, w1))
        if (q.qsize() < 1):
            return
        top = []
        for p in range(k):
            try:
                tmp = q.get_nowait()
                top.append(tmp[1])
                # if word in top:
                #     top_k[p] += 1
            except:
                break

        q1 = PriorityQueue()
        user_path_x, user_path_y = centerize(x, y)
        user_path = list(zip(user_path_x, user_path_y))
        for w2 in top:
            pattern = generate_standard_pattern(clean('g' + w2),
                                                len(user_path),
                                                lambda x: -2 * x**3 + 3 * x**2)
            if (len(user_path) <= 0 or len(pattern) <= 0):
                continue
            d, cost_matrix, acc_cost_matrix, path = dtw(
                user_path, pattern, dist=euclidian_distance)
            # import matplotlib.pyplot as plt

            # plt.imshow(acc_cost_matrix.T,
            #            origin='lower',
            #            cmap='gray',
            #            interpolation='nearest')
            # plt.plot(path[0], path[1], 'w')
            # plt.show()
            q1.put((d, w2))
        topp = []
        for p in range(k):
            try:
                tmp = q1.get_nowait()
                topp.append(tmp[1])
            except:
                break
        print(word, topp[0])

        plt.scatter(user_path_x, user_path_y, c='black')

        # ux, uy = 0, 0
        # uxs, uys = [0], [0]
        # for u in user:
        #     ux += u[0]
        #     uy += u[1]
        #     uxs.append(ux)
        #     uys.append(uy)
        # plt.plot(uxs, uys, color='red')

        # pred = topp[0]
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

        s_pattern = generate_standard_pattern(clean('g' + word), len(x),
                                              lambda x: -2 * x**3 + 3 * x**2)
        sx = [s[0] for s in s_pattern]
        sy = [s[1] for s in s_pattern]
        plt.scatter(sx, sy, color='yellow')

        plt.show()


def main():
    while True:
        show_difference()
    # check_top_k("qlp", 50)


if __name__ == "__main__":
    init_word_set()
    print("using set size: ", len(WORD_SET))
    # print(SENT_LIMIT, len(WORD_LIMIT))
    main()
