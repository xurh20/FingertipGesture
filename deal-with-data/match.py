import math
import os
import json
from os import curdir, name
import numpy as np
from dtw import dtw
from numpy.lib.function_base import append

from plot import gatherCorner, calFirstCorner, calSecondCorner, loadData, calculatePoints
from cleanWords import cleanWords, lowerCase
from advancedDtw import a_dtw

BASE_DIR = "../data/alphabeta_data_"
SAVE_MP_DIR = "../data/match_points/"  # save match points
PERSON = ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]
STD_KB_WIDTH = 1083
STD_KB_HEIGHT = 351
STD_KB_POS = {
    'q': np.array([-474, 105]),
    'w': np.array([-370, 105]),
    'e': np.array([-265, 105]),
    'r': np.array([-161, 105]),
    't': np.array([-52, 105]),
    'y': np.array([51, 105]),
    'u': np.array([156, 105]),
    'i': np.array([262, 105]),
    'o': np.array([367, 105]),
    'p': np.array([469, 105]),
    'a': np.array([-446, 0]),
    's': np.array([-340, 0]),
    'd': np.array([-235, 0]),
    'f': np.array([-131, 0]),
    'g': np.array([-28, 0]),
    'h': np.array([78, 0]),
    'j': np.array([184, 0]),
    'k': np.array([292, 0]),
    'l': np.array([398, 0]),
    'z': np.array([-400, -105]),
    'x': np.array([-293, -105]),
    'c': np.array([-187, -105]),
    'v': np.array([-82, -105]),
    'b': np.array([23, -105]),
    'n': np.array([127, -105]),
    'm': np.array([232, -105])
}
allPattern = cleanWords()


def linear_rectangle(pos):
    center = (0, 0)
    width = 8
    height = 8
    return np.array([
        center[0] + pos[0] * width / STD_KB_WIDTH,
        center[1] + pos[1] * height / STD_KB_HEIGHT
    ])


def genPoints(points_x, points_y, depths):
    pointLabels = []
    centerPointGroups = gatherCorner(
        calFirstCorner(points_x, points_y, depths),
        calSecondCorner(points_x, points_y, depths))
    # print(centerPointGroups)
    for pointGroup in centerPointGroups:
        pointLabels.append(pointGroup[int(len(pointGroup) / 2)])
    points = [[points_x[i], points_y[i], depths[i]] for i in pointLabels]
    # print(pointLabels)
    # print(points)
    return points


def genVectors(points_x, points_y, depths):
    vectors = []
    points = genPoints(points_x, points_y, depths)
    # calculate vector
    if (len(points) == 1):
        print("error only one point")
    for i in range(1, len(points)):
        v = np.array(
            [points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]])
        v = v / np.linalg.norm(v)
        vectors.append(v)
    return vectors


def genPattern(sentence, word, normalized=True):
    word = allPattern[sentence - 1][word]
    if (len(word) == 1):
        print("error genpattern with only one word")
    patterns = []
    for i in range(len(word) - 1):
        v = linear_rectangle(STD_KB_POS[lowerCase(
            word[i + 1])]) - linear_rectangle(STD_KB_POS[lowerCase(word[i])])
        # print(v)
        if normalized:
            v = v / np.linalg.norm(v)
        patterns.append(v)
    # print(word)
    return patterns


def genPattern(word, normalized=True):
    if (len(word) == 1):
        print("error genpattern with only one word")
    patterns = []
    for i in range(len(word) - 1):
        v = linear_rectangle(STD_KB_POS[lowerCase(
            word[i + 1])]) - linear_rectangle(STD_KB_POS[lowerCase(word[i])])
        # print(v)
        if normalized:
            v = v / np.linalg.norm(v)
        patterns.append(v)
    # print(word)
    return patterns


def distance(p: np.array, q: np.array) -> float:
    return -p.dot(q)


def showDistPath(sentence, word, person):
    data = loadData(sentence, word, person)
    points_x, points_y, depths = calculatePoints(data)
    vectors = np.array(genVectors(points_x, points_y, depths)).reshape(-1, 2)
    patterns = np.array(genPattern(sentence, word)).reshape(-1, 2)
    print(len(vectors))
    print(len(patterns))
    print(vectors)
    print(patterns)
    d, cost_matrix, acc_cost_matrix, path = dtw(vectors,
                                                patterns,
                                                dist=distance)
    print(d)
    print(path)
    import matplotlib.pyplot as plt

    plt.imshow(acc_cost_matrix.T,
               origin='lower',
               cmap='gray',
               interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.show()


def genLabelPoints(sentence, word, person):
    data = loadData(sentence, word, person)
    points_x, points_y, depths = calculatePoints(data)
    vectors = np.array(genVectors(points_x, points_y, depths)).reshape(-1, 2)
    patterns = np.array(genPattern(sentence, word)).reshape(-1, 2)
    if (len(vectors) == 0 or len(patterns) == 0):
        print("error, blank patterns")
        return []
    d, cost_matrix, acc_cost_matrix, path = a_dtw(vectors,
                                                  patterns,
                                                  dist=distance)
    # print(path)
    # if (len(path[0]) < max(path[1])):
    #     print("error, points chosen less than expected")
    points = genPoints(points_x, points_y, depths)
    match_group = []
    cur_point = 0
    for i in range(len(patterns)):
        match_num = np.sum(path[1] == i)
        if (match_num == 0):
            print("error, some point has no match")
            break
        elif (match_num == 1):
            match_group.append([path[0][path[1].tolist().index(i)]])
        else:
            match_group.append([path[0][path[1].tolist().index(i)]])
            # if (i == 0):
            #     last_zero_pos = match_num # for example you have 2 zeros, then the first point is zero and last is two
            #     min_dist = 2
            #     chosen_point = 0
            #     for first_pos in range(last_zero_pos):
            #         test_vector = np.array([points[last_zero_pos][0] - points[first_pos][0], points[last_zero_pos][1] - points[first_pos][1]])
            #         test_vector = test_vector / np.linalg.norm(test_vector)
            #         if (distance(test_vector, patterns[0]) < min_dist):
            #             min_dist = distance(test_vector, patterns[0])
            #             chosen_point = first_pos
            #     match_group.append([p for p in range(chosen_point, match_num)])
            #     cur_point = match_num - 1
            # elif (i == len(patterns) - 1):
            #     first_num_pos = max(cur_point, path[0][len(path[1]) - match_num])
            #     min_dist = 2
            #     chosen_point = 0
            #     for last_pos in range(first_num_pos + 1, len(points)):
            #         test_vector = np.array([points[last_pos][0] - points[first_num_pos][0], points[last_pos][1] - points[first_num_pos][1]])
            #         test_vector = test_vector / np.linalg.norm(test_vector)
            #         if (distance(test_vector, patterns[-1]) < min_dist):
            #             min_dist = distance(test_vector, patterns[-1])
            #             chosen_point = last_pos
            #     match_group.append([p for p in range(first_num_pos, chosen_point)])
            # else:
            #     match_group.append([p for p in range(path[0][path_str.index(str(i))], path[0][path_str.index(str(i))] + match_num)])
            #     # match_group.append([path[0][path_str.index(str(i))] + match_num - 1])
            #     cur_point = path[0][path_str.index(str(i))] + match_num - 1
    # print(match_group)
    match_points = []
    for i in match_group:
        match_points.append(i[0])
    match_points.append(match_group[-1][-1] + 1)
    return match_points


if __name__ == "__main__":
    # showDistPath(73, 4, PERSON[-1])
    # match_points = genLabelPoints(73, 4, PERSON[-2])
    # print(match_points)
    for person in PERSON:
        for i in range(1, 82):
            for j in range(len(allPattern[i - 1])):
                if os.path.exists(BASE_DIR + person + "/" + str(i) + "_" +
                                  str(j) + ".npy"):
                    with open(
                            SAVE_MP_DIR + person + "_" + str(i) + "_" +
                            str(j) + ".txt", "w") as f:
                        f.write(
                            json.dumps(
                                np.array(genLabelPoints(i, j,
                                                        person)).tolist()))
                else:
                    print("error, lost data ", person, i, j,
                          allPattern[i - 1][j])
                    break
            print("done", i)