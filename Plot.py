from itertools import product
from math import atan2, sqrt
import numpy as np
import argparse
import os, re
import matplotlib.pyplot as plt
from dtaidistance import dtw_ndim, dtw
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from Calculate import calculatePoints, genPointLabels
from persistence1d import RunPersistence
from queue import PriorityQueue
import seaborn as sns
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison

BASE_DIR = 'new_data'
LETTER = [chr(y) for y in range(97, 123)]
EIGHT_DIRECTIONS = [
    -np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 3, np.pi / 2,
    3 * np.pi / 4
]
FOUR_DIRECTIONS = [i * np.pi / 2 for i in range(4)]
DIRECTIONS_MAP = {
    '4': [(i - 2) * np.pi / 2 for i in range(4)],
    '6': [(i - 3) * np.pi / 3 for i in range(6)],
    '8': [(i - 4) * np.pi / 4 for i in range(8)],
    '10': [(i - 5) * np.pi / 5 for i in range(10)],
    '12': [(i - 6) * np.pi / 6 for i in range(12)],
}
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
UP = np.array([0, 1])
DOWN = np.array([0, -1])
COLORS = [
    'red', 'chocolate', 'darkorange', 'yellow', 'lawngreen', 'green', 'cyan',
    'slategrey', 'blue', 'darkviolet', 'magenta', 'hotpink'
]
DIRECTION_PATTERN = {
    'a': np.array([LEFT, RIGHT, DOWN]),
    'b': np.array([DOWN, UP, RIGHT, LEFT]),
    'c': np.array([LEFT, RIGHT]),
    'd': np.array([LEFT, RIGHT, UP, DOWN]),
    'e': np.array([RIGHT, LEFT, DOWN, RIGHT]),
    'f': np.array([LEFT, DOWN, UP, RIGHT]),
    'g': np.array([LEFT, RIGHT, DOWN, LEFT]),
    'h': np.array([DOWN, UP, RIGHT, DOWN]),
    'i': np.array([DOWN, UP]),
    'j': np.array([DOWN, LEFT]),
    'k': np.array([DOWN, UP, RIGHT, LEFT, RIGHT]),
    'l': np.array([DOWN]),
    'm': np.array([DOWN, UP, DOWN, UP, DOWN]),
    'n': np.array([DOWN, UP, DOWN]),
    'o': np.array([LEFT, DOWN, RIGHT, UP]),
    'p': np.array([DOWN, UP, RIGHT, DOWN, LEFT]),
    'q': np.array([LEFT, DOWN, RIGHT, DOWN]),
    'r': np.array([DOWN, UP, RIGHT]),
    's': np.array([LEFT, RIGHT, LEFT]),
    't': np.array([RIGHT, UP, DOWN]),
    'u': np.array([DOWN, RIGHT, UP, DOWN]),
    'v': np.array([DOWN, UP, LEFT]),
    'w': np.array([DOWN, UP, DOWN, UP]),
    'x': np.array([RIGHT, DOWN, UP, LEFT, DOWN]),
    'y': np.array([DOWN, RIGHT, UP, DOWN, LEFT]),
    'z': np.array([RIGHT, LEFT, RIGHT]),
}
DIRECTION_PATTERN8 = {
    'a': np.array([1, 5, 2]),
    'b': np.array([2, 5, 2, 7]),
    'c': np.array([0, 4]),
    'd': np.array([1, 5, 6, 2]),
    'e': np.array([4, 0, 2, 5]),
    'f': np.array([1, 2, 7, 4]),
    'g': np.array([1, 5, 2, 7]),
    'h': np.array([2, 5, 2]),
    'i': np.array([2, 6]),
    'j': np.array([2, 1]),
    'k': np.array([2, 5, 1, 3]),
    'l': np.array([2]),
    'm': np.array([2, 6, 2, 6, 2]),
    'n': np.array([2, 6, 2]),
    'o': np.array([1, 3, 6]),
    'p': np.array([2, 6, 4, 0]),
    'q': np.array([7, 2, 5, 2]),
    'r': np.array([2, 6, 5]),
    's': np.array([0, 4, 0]),
    't': np.array([4, 7, 2]),
    'u': np.array([2, 4, 6, 2]),
    'v': np.array([2, 5]),
    'w': np.array([2, 6, 2, 6]),
    'x': np.array([3, 6, 1]),
    'y': np.array([3, 5, 1]),
    'z': np.array([4, 0, 4]),
}

allowed_dir = [[1, 3, 4], [0, 2, 3, 4, 5], [1, 4, 5], [0, 1, 4, 6, 7],
               [0, 1, 2, 3, 5, 6, 7, 8], [1, 2, 4, 7, 8], [3, 4, 7],
               [3, 4, 5, 6, 8], [4, 5, 7]]
allowed_coord = [
    np.array([-1, 1]),
    np.array([0, 1]),
    np.array([1, 1]),
    np.array([-1, 0]),
    np.array([0, 0]),
    np.array([1, 0]),
    np.array([-1, -1]),
    np.array([0, -1]),
    np.array([1, -1]),
]


def getAveragePath(path, align_to_first=True, integer=False):
    """
    description
    ---------
    Get average path from path containing matrices to x, y, depth array respectively
    
    param
    -------
    path: path containing matrices
    align_to_first: whether output begins with (0,0)
    integer: whether output coordinates are integers
    
    Returns
    -------
    (x, y, depth)
    
    """

    points_x = []
    points_y = []
    depths = []
    for frame in path:
        sum_force = 0
        x_average = 0
        y_average = 0
        for y_coordinate in range(len(frame)):
            for x_coordinate in range(len(frame[y_coordinate])):
                sum_force += frame[y_coordinate][x_coordinate]
        if (sum_force > 1000):
            for y_coordinate in range(len(frame)):
                for x_coordinate in range(len(frame[y_coordinate])):
                    rate = frame[y_coordinate][x_coordinate] / sum_force
                    x_average += rate * x_coordinate
                    y_average += rate * y_coordinate
            if integer:
                points_x.append(int(x_average * 10))
                points_y.append(int((35 - y_average) * 10))
            else:
                points_x.append(x_average)
                points_y.append(35 - y_average)
            depths.append(sum_force)
    if align_to_first:
        center_x = points_x[0]
        center_y = points_y[0]
        for i in range(len(points_x)):
            points_x[i] -= center_x
        for i in range(len(points_y)):
            points_y[i] -= center_y

    i = 1
    while i < len(depths) and depths[i] > depths[i - 1]:
        i += 1
    first_extrema = depths[i]
    while i >= 0 and depths[i] >= first_extrema * 0.3:
        i -= 1
    trunc_at_start = i
    i = len(depths) - 1
    while i > 1 and depths[i - 1] > depths[i]:
        i -= 1
    last_extrema = depths[i]
    while i < len(depths) and depths[i] >= last_extrema * 0.7:
        i += 1
    trunc_at_end = i
    return np.array(points_x[trunc_at_start:trunc_at_end]), np.array(
        points_y[trunc_at_start:trunc_at_end]), np.array(
            depths[trunc_at_start:trunc_at_end])


def getXYExtrema(path):
    THRESHOLD = (0.3, 0.3)
    x, y = getAveragePath(path)
    x = gaussian_filter1d(x, sigma=5)
    xExtrema, yExtrema = [], []
    for k, (v, l) in enumerate(list(zip([x, y], [xExtrema, yExtrema]))):
        for i, e in enumerate(
                sorted([t for t in RunPersistence(v) if t[1] > THRESHOLD[k]],
                       key=lambda x: x[1])):
            l.append(-1 if i % 2 == 0 else 1)
    return np.array(xExtrema, dtype=np.float64), np.array(yExtrema,
                                                          dtype=np.float64)


def getLongestDirection(path):
    """
    description
    ---------
    Get longest direction as the main direction for the input path
    
    param
    -------
    path: path containing matrices
    
    Returns
    -------
    (start, end) indicating the start and end of the identified direction
    
    """

    ANGLE_THRESHOLD = np.pi / 4

    def angleDiff(u, v):
        ang1 = np.arctan2(u[1], u[0])
        ang2 = np.arctan2(v[1], v[0])
        return 2 * np.pi - abs(ang1 -
                               ang2) if abs(ang1 -
                                            ang2) >= np.pi else abs(ang1 -
                                                                    ang2)

    # smooth the path
    x, y, d = getAveragePath(path)
    x = gaussian_filter1d(x, sigma=5)
    y = gaussian_filter1d(y, sigma=5)

    corners = getCorners(path)
    debug_dir = []
    debug_dir.append(corners[0])
    if len(corners) > 2:
        for i in range(1, len(corners) - 1):
            if angleDiff(
                (x[corners[i]] - x[corners[i - 1]],
                 y[corners[i]] - y[corners[i - 1]]),
                (x[corners[i + 1]] - x[corners[i]],
                 y[corners[i + 1]] - y[corners[i]])) >= ANGLE_THRESHOLD:
                debug_dir.append(corners[i])
    debug_dir.append(corners[-1])

    max_dis = -1
    start, end = 0, len(x) - 1
    for i in range(1, len(debug_dir)):
        if debug_dir[i] - debug_dir[i - 1] > max_dis:
            max_dis = debug_dir[i] - debug_dir[i - 1]
            start = debug_dir[i - 1]
            end = debug_dir[i]

    return start, end


def getCorners(path):
    """
    description
    ---------
    Get corners of path
    
    param
    -------
    path: path containing matrices
    
    Returns
    -------
    array containing indexes of corners
    
    """

    ANGLE_THRESHOLD = np.pi / 4
    MERGE_THRESHOLD = 10
    HV_THRESHOLD = 0.6
    HV_AVG_THRESHOLD = 1.1

    def angleDiff(u, v):
        ang1 = np.arctan2(u[1], u[0])
        ang2 = np.arctan2(v[1], v[0])
        return 2 * np.pi - abs(ang1 -
                               ang2) if abs(ang1 -
                                            ang2) >= np.pi else abs(ang1 -
                                                                    ang2)

    # smooth the path
    x, y, d = getAveragePath(path)
    x = gaussian_filter1d(x, sigma=5)
    y = gaussian_filter1d(y, sigma=5)

    # filter the turning points that seperates angles
    debug_dir = []
    debug_dir.append(0)
    i = 1
    while i < len(x):
        cur_v = (x[i] - x[i - 1], y[i] - y[i - 1])
        j = i + 1
        while j < len(x) and angleDiff(
            (x[j] - x[j - 1], y[j] - y[j - 1]),
                cur_v) < ANGLE_THRESHOLD and angleDiff(
                    (x[j] - x[j - 1], y[j] - y[j - 1]),
                    (x[i] - x[i - 1], y[i] - y[i - 1])) < ANGLE_THRESHOLD:
            cur_v = (x[j] - x[i - 1], y[j] - y[i - 1])
            j += 1
        debug_dir.append(i)
        i = j

    # merge points that are enough close to each other
    i = 0
    simplified_dir = []
    while i < len(debug_dir):
        t = []
        t.append(debug_dir[i])
        while i + 1 < len(debug_dir) and debug_dir[
                i + 1] - debug_dir[i] < MERGE_THRESHOLD:
            t.append(debug_dir[i + 1])
            i += 1
        i += 1
        simplified_dir.append(t[int(len(t) / 2)])

    simplified_dir.append(len(x) - 1)

    # calculate the average of x/y offsets
    x_avg, y_avg = 0, 0
    for u, v in list(zip(simplified_dir[:-1], simplified_dir[1:])):
        if abs(x[v] - x[u]) > abs(y[v] - y[u]):
            x_avg += abs(x[v] - x[u])
        else:
            y_avg += abs(y[v] - y[u])
    x_avg = x_avg / (len(simplified_dir) - 1)
    y_avg = y_avg / (len(simplified_dir) - 1)

    # filter the turning points that are far enough to each other
    filtered_dir = []
    for sd in simplified_dir:
        if len(filtered_dir) <= 0:
            filtered_dir.append(sd)
            continue
        x_off = abs(x[sd] - x[filtered_dir[-1]])
        y_off = abs(y[sd] - y[filtered_dir[-1]])
        if (x_off > x_avg * HV_AVG_THRESHOLD
                or y_off > y_avg * HV_AVG_THRESHOLD) and (
                    x_off > HV_THRESHOLD or y_off > HV_THRESHOLD):
            filtered_dir.append(sd)
    return filtered_dir


def getHVDirections(path):

    def chooseClosestDirection(v):
        # ang = np.arctan2(v[1], v[0])
        # if abs(abs(ang) - np.pi) < abs(abs(ang) - np.pi / 2):
        #     return LEFT
        # return [RIGHT, UP, DOWN][np.argmin(
        #     [abs(ang - 0),
        #      abs(ang - np.pi / 2),
        #      abs(ang + np.pi / 2)])]
        if abs(v[1]) > abs(v[0]):
            if v[1] > 0:
                return UP
            else:
                return DOWN
        else:
            if v[0] > 0:
                return RIGHT
            else:
                return LEFT

    x, y, d = getAveragePath(path)
    simplified_dir = getCorners(path)
    directions = []

    for u, v in list(zip(simplified_dir[:-1], simplified_dir[1:])):
        if len(directions) <= 0 or list(directions[-1]) != list(
                chooseClosestDirection((x[v] - x[u], y[v] - y[u]))):
            directions.append(
                chooseClosestDirection((x[v] - x[u], y[v] - y[u])))
    # plt.plot(x, y)
    # for i in simplified_dir:
    #     plt.scatter([x[i]], [y[i]], c='red')
    # plt.show()

    return np.array(directions)


def get8Directions(path):
    """
    description
    ---------
    Get all directions adopted to 8 directions
    
    param
    -------
    path: path containing matrices
    
    Returns
    -------
    arrays of directions
    
    """

    def chooseClosestDirection(v):
        ang = np.arctan2(v[1], v[0])
        if abs(ang) > 7 * np.pi / 8:
            return 0
        ans = np.argmin([abs(ang - std_ang) for std_ang in EIGHT_DIRECTIONS])
        return ans

    x, y, d = getAveragePath(path)
    simplified_dir = getCorners(path)
    directions = []

    for u, v in list(zip(simplified_dir[:-1], simplified_dir[1:])):
        if len(directions) <= 0 or directions[-1] != chooseClosestDirection(
            (x[v] - x[u], y[v] - y[u])):
            directions.append(
                chooseClosestDirection((x[v] - x[u], y[v] - y[u])))
    # plt.axis("scaled")
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    # plt.plot(x, y)
    # for i in simplified_dir:
    #     plt.scatter([x[i]], [y[i]], c='red')
    # print(directions)
    # plt.show()

    return np.array(directions)


def getConfidenceQueue(path):
    directions = getHVDirections(path)
    q = PriorityQueue()
    for ch in LETTER:
        q.put((dtw_ndim.distance(directions, DIRECTION_PATTERN[ch]), ch))
    return q


def getConfidenceQueue8(path):
    """
    description
    ---------
    Get confidence queue for eight-direction pattern
    
    param
    -------
    path: path containing matrices
    
    Returns
    -------
    queue containing (dist, letter)
    
    """

    directions = [
        np.array([np.cos(EIGHT_DIRECTIONS[i]),
                  np.sin(EIGHT_DIRECTIONS[i])]) for i in get8Directions(path)
    ]
    q = PriorityQueue()
    for ch in LETTER:
        pattern = [
            np.array(
                [np.cos(EIGHT_DIRECTIONS[i]),
                 np.sin(EIGHT_DIRECTIONS[i])]) for i in DIRECTION_PATTERN8[ch]
        ]
        q.put((dtw_ndim.distance(directions, pattern), ch))
    return q


def getPathAngles(x, y):
    angles = []
    for i in range(len(x)):
        if i == 0:
            continue
        angles.append(atan2(y[i] - y[i - 1], x[i] - x[i - 1]))
    return np.array(angles)


def plotAllLetters():
    dir = os.path.join(BASE_DIR,
                       'ch_data_' + args.person + '_' + str(args.index))
    for i, c in enumerate(LETTER):
        path = np.load(os.path.join(dir, c + '_2.npy'))
        x, y, d = getAveragePath(path, False)
        plt.subplot(3, 9, i + 1)
        plt.axis("scaled")
        plt.xlim(10, 17)
        plt.ylim(15, 25)
        plt.xlabel(LETTER[i])
        plt.scatter(x, y)
    plt.show()


def plotAllLettersBig():
    dir = os.path.join(BASE_DIR,
                       'ch_data_' + args.person + '_' + str(args.index))
    for i, c in enumerate(LETTER):
        path = np.load(os.path.join(dir, c + '_2.npy'))
        x, y, d = getAveragePath(path, False, True)
        plt.subplot(3, 9, i + 1)
        plt.axis("scaled")
        plt.xlim(100, 170)
        plt.ylim(150, 250)
        plt.xlabel(LETTER[i])
        plt.scatter(x, y)
    plt.show()


def plotAllLettersCorner():
    dir = os.path.join(BASE_DIR,
                       'ch_data_' + args.person + '_' + str(args.index))
    for i, c in enumerate(LETTER):
        path = np.load(os.path.join(dir, c + '_3.npy'))
        # x, y, d = getAveragePath(path, False)
        # corners = getCorners(path)
        # plt.subplot(3, 9, i + 1)
        # plt.axis("scaled")
        # plt.xlim(10, 17)
        # plt.ylim(15, 25)
        # plt.xlabel(LETTER[i])
        # plt.scatter(x, y)
        # for _, corner in enumerate(corners):
        #     plt.scatter([x[corner]], [y[corner]], c='red')
        # plt.text(x[corner], y[corner], str(_) + ',' + str(d[corner]))
        # print(getHVDirections(path))
        # print(d)
        try:
            get8Directions(path)
        except:
            pass
    plt.show()


def plotOneLettersCorner(path):
    x, y, d = getAveragePath(path, False)
    corners = getCorners(path)
    plt.axis("scaled")
    plt.xlim(10, 17)
    plt.ylim(15, 25)
    plt.scatter(x, y)
    for _, corner in enumerate(corners):
        plt.scatter([x[corner]], [y[corner]], c='red')
        plt.text(x[corner], y[corner], str(_))
    print(getHVDirections(path))
    # print(d)
    plt.show()


def plotOneLettersCorner8(path):
    """
    description
    ---------
    Plot one letter's corners adoopting to 8 directions
    
    param
    -------
    path: path containing matrices
    
    Returns
    -------
    None
    
    """

    x, y, d = getAveragePath(path, False)
    corners = getCorners(path)
    plt.axis("scaled")
    plt.xlim(10, 17)
    plt.ylim(15, 25)
    plt.scatter(x, y)
    for _, corner in enumerate(corners):
        plt.scatter([x[corner]], [y[corner]], c='red')
        plt.text(x[corner], y[corner], str(_))
    print(get8Directions(path))
    # print(d)
    plt.show()


def plotAllLettersAngle():
    dir = os.path.join(BASE_DIR,
                       'ch_data_' + args.person + '_' + str(args.index))
    for i, c in enumerate(LETTER):
        path = np.load(os.path.join(dir, c + '_2.npy'))
        x, y, d = getAveragePath(path, align_to_first=False, integer=False)
        angles = getPathAngles(x, y)
        angles = gaussian_filter1d(angles, sigma=5)
        indexs = list(range(len(angles)))
        plt.subplot(3, 9, i + 1)
        plt.ylim(-np.pi, np.pi)
        plt.xlabel(LETTER[i])
        plt.scatter(indexs, angles)
    plt.show()


def plotAllLettersX():
    dir = os.path.join(BASE_DIR,
                       'ch_data_' + args.person + '_' + str(args.index))
    for i, c in enumerate(LETTER):
        path = np.load(os.path.join(dir, c + '_2.npy'))
        x, y, d = getAveragePath(path, align_to_first=False, integer=False)
        x = gaussian_filter1d(x, sigma=5)
        indexs = list(range(len(x)))
        plt.subplot(3, 9, i + 1)
        plt.ylim(10, 17)
        plt.xlabel(LETTER[i])
        plt.plot(indexs, x)
        for i, e in enumerate(
                sorted([t for t in RunPersistence(x) if t[1] > 0.5],
                       key=lambda x: x[1])):
            if i % 2 == 0:
                plt.scatter([e[0]], [x[e[0]]], c='blue')
            else:
                plt.scatter([e[0]], [x[e[0]]], c='red')
    plt.show()


def plotAllLettersY():
    dir = os.path.join(BASE_DIR,
                       'ch_data_' + args.person + '_' + str(args.index))
    for i, c in enumerate(LETTER):
        path = np.load(os.path.join(dir, c + '_2.npy'))
        x, y, d = getAveragePath(path, align_to_first=False, integer=False)
        y = gaussian_filter1d(y, sigma=5)
        indexs = list(range(len(y)))
        plt.subplot(3, 9, i + 1)
        plt.ylim(15, 25)
        plt.xlabel(LETTER[i])
        plt.plot(indexs, y)
    plt.show()


def plotDirections():
    """
    description
    ---------
    Plot input directions with standard directions
    
    param
    -------
    None
    
    Returns
    -------
    None
    
    """

    avg_angles = []
    std_angles = []
    for dir in os.listdir(BASE_DIR):
        if not 'ch_data_' + str(args.direction) + '_dir_' in dir:
            continue
        for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
            for j in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir,
                                     str(i) + '_{}.npy'.format(j)))
                    x, y, d = getAveragePath(path,
                                             align_to_first=False,
                                             integer=False)
                    x = gaussian_filter1d(x, sigma=5)
                    y = gaussian_filter1d(y, sigma=5)
                    start, end = getLongestDirection(path)
                    # start, end = getStartAndEnd(args.direction, x, y, i)
                except:
                    continue
                if end < 0:
                    continue
                angle = np.arctan2(y[end] - y[start], x[end] - x[start])
                # if angle > (DIRECTIONS_MAP[args.direction][-1] + np.pi) / 2:
                if i == 0 and angle > 0:
                    angle -= 2 * np.pi
                # plt.axis("scaled")
                # plt.scatter([np.cos(angle)], [np.sin(angle)], c=COLORS[i])

                # if i == 0:
                #     print(
                #         os.path.join(BASE_DIR, dir,
                #                      str(i) + '_{}.npy'.format(j)))
                #     print(c, angle)
                #     plt.axis("scaled")
                #     plt.xlim(10, 17)
                #     plt.ylim(15, 25)
                #     plt.scatter(x, y, c='blue')
                #     plt.scatter([x[start], x[end]], [y[start], y[end]],
                #                 c='red')
                #     plt.show()
                avg_angles.append(angle)
                std_angles.append(c)
    plt.show()
    # plt.axis("scaled")
    # plt.xlim(-np.pi / 4, 2 * np.pi)
    # plt.ylim(-np.pi / 4, 2 * np.pi)
    # plt.scatter(std_angles, avg_angles)
    # plt.show()
    df = pd.DataFrame({'std_angle': std_angles, 'usr_angle': avg_angles})
    fig, axes = plt.subplots()
    sns.boxplot(x='std_angle', y='usr_angle', data=df, ax=axes)
    plt.show()


def plotAmplitude():
    """
    description
    ---------
    Plot amplitude of 8 directions
    
    param
    -------
    None
    
    Returns
    -------
    None
    
    """
    plt.axis("scaled")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    for dir in os.listdir(BASE_DIR):
        if not 'ch_data_8_dir_' in dir:
            continue
        for i, c in enumerate(DIRECTIONS_MAP['8']):
            for j in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir,
                                     str(i) + '_{}.npy'.format(j)))
                    x, y, d = getAveragePath(path,
                                             align_to_first=False,
                                             integer=False)
                    x = gaussian_filter1d(x, sigma=5)
                    y = gaussian_filter1d(y, sigma=5)
                    start, end = getLongestDirection(path)
                    # start, end = getStartAndEnd(args.direction, x, y, i)
                except:
                    continue
                if end < 0:
                    continue
                angle = np.arctan2(y[end] - y[start], x[end] - x[start])
                # if angle > (DIRECTIONS_MAP[args.direction][-1] + np.pi) / 2:
                if i == 0 and angle > 0:
                    angle -= 2 * np.pi
                # plt.axis("scaled")
                # plt.scatter([np.cos(angle)], [np.sin(angle)], c=COLORS[i])

                # if i == 0:
                #     print(
                #         os.path.join(BASE_DIR, dir,
                #                      str(i) + '_{}.npy'.format(j)))
                #     print(c, angle)
                #     plt.axis("scaled")
                #     plt.xlim(10, 17)
                #     plt.ylim(15, 25)
                #     plt.scatter(x, y, c='blue')
                #     plt.scatter([x[start], x[end]], [y[start], y[end]],
                #                 c='red')
                #     plt.show()
                plt.scatter([x[end] - x[start]], [y[end] - y[start]],
                            c=COLORS[i])
    plt.show()


def plotPressure():
    """
    description
    ---------
    Plot pressure of 8 directions
    
    param
    -------
    None
    
    Returns
    -------
    None
    
    """
    for dir in os.listdir(BASE_DIR):
        if not 'ch_data_8_dir_' in dir:
            continue
        for i, c in enumerate(DIRECTIONS_MAP['8']):
            for j in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir,
                                     str(i) + '_{}.npy'.format(j)))
                    x, y, d = getAveragePath(path,
                                             align_to_first=False,
                                             integer=False)
                    x = gaussian_filter1d(x, sigma=5)
                    y = gaussian_filter1d(y, sigma=5)
                    start, end = getLongestDirection(path)
                    # start, end = getStartAndEnd(args.direction, x, y, i)
                except:
                    continue
                if end < 0:
                    continue
                angle = np.arctan2(y[end] - y[start], x[end] - x[start])
                # if angle > (DIRECTIONS_MAP[args.direction][-1] + np.pi) / 2:
                if i == 0 and angle > 0:
                    angle -= 2 * np.pi
                # plt.axis("scaled")
                # plt.scatter([np.cos(angle)], [np.sin(angle)], c=COLORS[i])

                # if i == 0:
                #     print(
                #         os.path.join(BASE_DIR, dir,
                #                      str(i) + '_{}.npy'.format(j)))
                #     print(c, angle)
                #     plt.axis("scaled")
                #     plt.xlim(10, 17)
                #     plt.ylim(15, 25)
                #     plt.scatter(x, y, c='blue')
                #     plt.scatter([x[start], x[end]], [y[start], y[end]],
                #                 c='red')
                #     plt.show()
                plt.scatter([list(range(end - start + 1))], [d[start:end + 1]],
                            c=COLORS[i])
    plt.show()


def anovaDirections():
    """
    description
    ---------
    Analyze anova of distiguishability between given directions
    
    param
    -------
    None
    
    Returns
    -------
    None
    
    """
    avg_angles = []
    std_angles = []
    for dir in os.listdir(BASE_DIR):
        if not 'ch_data_' + str(args.direction) + '_dir_' in dir:
            continue
        for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
            for j in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir,
                                     str(i) + '_{}.npy'.format(j)))
                    x, y, d = getAveragePath(path,
                                             align_to_first=False,
                                             integer=False)
                    x = gaussian_filter1d(x, sigma=5)
                    y = gaussian_filter1d(y, sigma=5)
                    start, end = getLongestDirection(path)
                    # start, end = getStartAndEnd(args.direction, x, y, i)
                except:
                    continue
                if end < 0:
                    continue
                angle = np.arctan2(y[end] - y[start], x[end] - x[start])
                # if angle > (DIRECTIONS_MAP[args.direction][-1] + np.pi) / 2:
                if i == 0 and angle > 0:
                    angle -= 2 * np.pi

                avg_angles.append(angle)
                std_angles.append(c)

    df = pd.DataFrame({'std_angle': std_angles, 'usr_angle': avg_angles})
    # fig, axes = plt.subplots()
    # sns.boxplot(x='std_angle', y='usr_angle', data=df, ax=axes)
    # plt.show()
    model = ols('std_angle~usr_angle', data=df).fit()
    print(model.summary())
    model = ols('usr_angle~C(std_angle)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    print(anova_table)
    mc = MultiComparison(avg_angles, std_angles)
    print(mc.tukeyhsd())


def plotDoubleDirectionsCutToSingle():
    """
    description
    ---------
    Plot cutting directions adopting to 8 directions
    
    param
    -------
    None
    
    Returns
    -------
    None
    
    """

    def getAngle(x1, x2, y1, y2):
        angle = np.arctan2(y2 - y1, x2 - x1)
        if angle > (DIRECTIONS_MAP['8'][-1] + np.pi) / 2:
            angle -= 2 * np.pi
        return angle

    avg_angles = []
    std_angles = []
    orders = []
    for dir in os.listdir(BASE_DIR):
        if not 'dd_' in dir:
            continue
        for u in range(9):
            for v in allowed_dir[u]:
                for w in allowed_dir[v]:
                    path = np.load(
                        os.path.join(BASE_DIR, dir,
                                     "%d_%d_%d.npy" % (u, v, w)))

                    try:
                        x, y, d = getAveragePath(path, align_to_first=False)
                        corners = getCorners(path)
                        included_angle1 = 0
                        included_angle2 = 0
                        if len(corners) >= 3:
                            max_uvw = -1
                            for _u, _v, _w in list(
                                    zip(corners[:-2], corners[1:-1],
                                        corners[2:])):
                                if _w - _u > max_uvw:
                                    max_uvw = _w - _u
                                    included_angle1 = getAngle(
                                        x[_u], x[_v], y[_u], y[_v])
                                    included_angle2 = getAngle(
                                        x[_v], x[_w], y[_v], y[_w])

                        std_incl_angle1 = getAngle(
                            allowed_coord[u][0],
                            allowed_coord[v][0],
                            allowed_coord[u][1],
                            allowed_coord[v][1],
                        )
                        std_incl_angle2 = getAngle(
                            allowed_coord[v][0],
                            allowed_coord[w][0],
                            allowed_coord[v][1],
                            allowed_coord[w][1],
                        )
                        avg_angles.append(included_angle1)
                        std_angles.append(std_incl_angle1)
                        orders.append('First')
                        avg_angles.append(included_angle2)
                        std_angles.append(std_incl_angle2)
                        orders.append('Second')

                        # plt.axis("scaled")
                        # plt.xlim(10, 17)
                        # plt.ylim(15, 25)
                        # plt.scatter(x, y, c='blue')
                        # for corner in corners:
                        #     plt.scatter([x[corner]], [y[corner]], c='red')
                        # plt.show()
                    except Exception as e:
                        print(str(e))

    # plt.axis("scaled")
    # plt.xlim(-np.pi / 4, 2 * np.pi)
    # plt.ylim(-np.pi / 4, 2 * np.pi)
    # plt.scatter(std_angles, avg_angles)
    # plt.show()
    df = pd.DataFrame({
        'std_angle': std_angles,
        'usr_angle': avg_angles,
        'order': orders
    })
    fig, axes = plt.subplots()
    sns.boxplot(x='std_angle', y='usr_angle', hue='order', data=df, ax=axes)
    plt.show()


def plotDoubleDirections():
    """
    description
    ---------
    Plot included angles of user input directions with standard included angles
    
    param
    -------
    None
    
    Returns
    -------
    None
    
    """
    avg_angles = []
    std_angles = []
    for dir in os.listdir(BASE_DIR):
        if not 'dd_' in dir:
            continue
        for u in range(9):
            for v in allowed_dir[u]:
                for w in allowed_dir[v]:
                    path = np.load(
                        os.path.join(BASE_DIR, dir,
                                     "%d_%d_%d.npy" % (u, v, w)))

                    try:
                        x, y, d = getAveragePath(path, align_to_first=False)
                        corners = getCorners(path)
                        included_angle = 0
                        if len(corners) >= 3:
                            max_uvw = -1
                            for _u, _v, _w in list(
                                    zip(corners[:-2], corners[1:-1],
                                        corners[2:])):
                                if _w - _u > max_uvw:
                                    max_uvw = _w - _u
                                    v1 = np.array(
                                        [x[_u] - x[_v], y[_u] - y[_v]])
                                    v2 = np.array(
                                        [x[_v] - x[_w], y[_v] - y[_w]])
                                    included_angle = np.arccos(
                                        np.dot(v1, v2) / np.linalg.norm(v1) /
                                        np.linalg.norm(v2))
                        if v - u == w - v:
                            std_incl_angle = 0
                        elif u == w:
                            std_incl_angle = np.pi
                        else:
                            v1 = allowed_coord[u] - allowed_coord[v]
                            v2 = allowed_coord[v] - allowed_coord[w]

                            std_incl_angle = np.arccos(
                                np.dot(v1, v2) / np.linalg.norm(v1) /
                                np.linalg.norm(v2))
                        avg_angles.append(included_angle)
                        std_angles.append(std_incl_angle)

                        # plt.axis("scaled")
                        # plt.xlim(10, 17)
                        # plt.ylim(15, 25)
                        # plt.scatter(x, y, c='blue')
                        # for corner in corners:
                        #     plt.scatter([x[corner]], [y[corner]], c='red')
                        # plt.show()
                    except:
                        pass

    # plt.axis("scaled")
    # plt.xlim(-np.pi / 4, 2 * np.pi)
    # plt.ylim(-np.pi / 4, 2 * np.pi)
    # plt.scatter(std_angles, avg_angles)
    # plt.show()
    df = pd.DataFrame({'std_angle': std_angles, 'usr_angle': avg_angles})
    fig, axes = plt.subplots()
    sns.boxplot(x='std_angle', y='usr_angle', data=df, ax=axes)
    plt.show()


def plot42Directions():
    avg_angles = []
    std_angles = []
    for _ in range(2):
        dir = os.path.join(BASE_DIR, 'ch_data_42_dir_' + str(_))
        for i, c in enumerate(FOUR_DIRECTIONS):
            for a in range(2):
                for j in range(5):
                    path = np.load(
                        os.path.join(dir,
                                     LETTER[i * 2 + a] + '_{}.npy'.format(j)))
                    try:
                        x, y, d = getAveragePath(path,
                                                 align_to_first=False,
                                                 integer=False)
                        x = gaussian_filter1d(x, sigma=5)
                        y = gaussian_filter1d(y, sigma=5)
                        # start, end = getLongestDirection(path)
                        if i == 0:
                            start = np.argmin(x)
                            end = np.argmax(x)
                        elif i == 1:
                            start = np.argmin(y)
                            end = np.argmax(y)
                        elif i == 2:
                            start = np.argmax(x)
                            end = np.argmin(x)
                        elif i == 3:
                            start = np.argmax(y)
                            end = np.argmin(y)
                    except:
                        continue
                    dis = sqrt((x[end] - x[start])**2 + (y[end] - y[start])**2)
                    angle = np.arctan2(y[end] - y[start], x[end] - x[start])
                    plt.axis("scaled")
                    plt.scatter([x[end] - x[start]], [y[end] - y[start]],
                                c=COLORS[i],
                                alpha=(a + 1) / 2)
                    if angle < -np.pi / 4:
                        angle = angle + 2 * np.pi

                    # plt.axis("scaled")
                    # plt.xlim(10, 17)
                    # plt.ylim(15, 25)
                    # plt.scatter(x, y, c='blue')
                    # plt.scatter([x[start], x[end]], [y[start], y[end]],
                    #             c='red')
                    # plt.show()
                    avg_angles.append(angle)
                    std_angles.append(c)
    plt.show()
    plt.axis("scaled")
    plt.xlim(-np.pi / 4, 2 * np.pi)
    plt.ylim(-np.pi / 4, 2 * np.pi)
    plt.scatter(std_angles, avg_angles)
    plt.show()


def plot83Pressure():
    xx = []
    yy = []
    for _ in range(2):
        dir = os.path.join(BASE_DIR, 'ch_data_p8_' + str(_))
        for i, c in enumerate(EIGHT_DIRECTIONS):
            for k in range(3):
                for j in range(5):
                    path = np.load(
                        os.path.join(dir,
                                     LETTER[i * 3 + k] + '_{}.npy'.format(j)))
                    try:
                        x, y, d = getAveragePath(path,
                                                 align_to_first=False,
                                                 integer=False)
                        x = gaussian_filter1d(x, sigma=5)
                        y = gaussian_filter1d(y, sigma=5)
                        # start, end = getLongestDirection(path)
                        if i == 3 or i == 4 or i == 5:
                            start = np.argmin(x)
                            end = np.argmax(x)
                        elif i == 6:
                            start = np.argmin(y)
                            end = np.argmax(y)
                        elif i == 2:
                            start = np.argmax(y)
                            end = np.argmin(y)
                        elif i == 0 or i == 1 or i == 7:
                            start = np.argmax(x)
                            end = np.argmin(x)
                    except:
                        continue
                    angle = np.arctan2(y[end] - y[start], x[end] - x[start])
                    avg_pressure = np.average(d[start:end])
                    plt.axis("scaled")
                    plt.scatter([avg_pressure * np.cos(angle)],
                                [avg_pressure * np.sin(angle)],
                                c=COLORS[i],
                                alpha=(k + 1) / 3)
                    if angle < -np.pi / 8:
                        angle = angle + 2 * np.pi

                    # plt.axis("scaled")
                    # plt.xlim(10, 17)
                    # plt.ylim(15, 25)
                    # plt.scatter(x, y, c='blue')
                    # plt.scatter([x[start], x[end]], [y[start], y[end]], c='red')
                    xx.append(k)
                    yy.append(avg_pressure)
    plt.show()
    # plt.axis("scaled")
    # plt.xlim(-np.pi / 4, 2 * np.pi)
    # plt.ylim(-np.pi / 4, 2 * np.pi)
    plt.scatter(xx, yy)
    plt.show()


def plotStartDiff():
    dir = os.path.join(BASE_DIR, 'ch_data_sd_' + str(args.index))
    for i in range(0, 8, 2):
        for j in range(5):
            path1 = np.load(os.path.join(dir, LETTER[i] + '_%d.npy' % j))
            path2 = np.load(os.path.join(dir, LETTER[i + 1] + '_%d.npy' % j))
            x, y, d = getAveragePath(path1, False)
            plt.subplot(1, 2, 1)
            plt.axis("scaled")
            plt.xlim(10, 17)
            plt.ylim(15, 25)
            plt.xlabel('Free')
            plt.scatter(x, y)
            plt.scatter([np.average(x)], [np.average(y)], c='red')
            x, y, d = getAveragePath(path2, False)
            plt.subplot(1, 2, 2)
            plt.axis("scaled")
            plt.xlim(10, 17)
            plt.ylim(15, 25)
            plt.xlabel('Center')
            plt.scatter(x, y)
            plt.scatter([np.average(x)], [np.average(y)], c='red')
            plt.show()


def calCrossAcc(duel):
    x1, x2 = [], []
    for c in LETTER:
        for xi, dir_name, idx in list(zip([x1, x2], ['jjx_0', 'jjx_1'], duel)):
            # xi.append(
            #     getAveragePath(
            #         np.load(
            #             os.path.join(BASE_DIR, 'ch_data_' + dir_name,
            #                          c + '_2.npy'))))
            # xi.append(
            #     gaussian_filter1d(getPathAngles(*getAveragePath(
            #         np.load(
            #             os.path.join(BASE_DIR, 'ch_data_' + dir_name, c +
            #                          '_2.npy')))),
            #                       sigma=5))
            # xi.append(
            #     np.array(
            #         list(
            #             zip(*getAveragePath(
            #                 np.load(
            #                     os.path.join(BASE_DIR, 'ch_data_' +
            #                                  dir_name, c + '_' + str(idx) +
            #                                  '.npy')))))))
            xi.append(
                getXYExtrema(
                    np.load(
                        os.path.join(BASE_DIR, 'ch_data_' + dir_name,
                                     c + '_' + str(idx) + '.npy'))))
    print(x1, x2)
    x, y = [], []
    top_k = [0] * 3
    for i in range(26):
        distances = []
        for j in range(26):
            distances.append(
                # dtw_ndim.distance(x1[i], x2[j], window=10, use_c=True))
                # dtw.distance(x1[i], x2[j], window=10, use_c=True))
                dtw.distance(x1[i][0], x2[j][0], window=10, use_c=True) +
                dtw.distance(x1[i][1], x2[j][1], window=10, use_c=True))
        x.append(i)
        y.append(np.argmin(distances))
        sorted_index = np.argsort(distances)
        for k in range(3):
            if i in sorted_index[:(k + 1)]:
                top_k[k] += 1
    # plt.scatter(x, y)
    # plt.show()
    # for i in range(26):
    #     print(LETTER[i], LETTER[y[i]])
    print(top_k)


def calPatternAcc():
    dir = os.path.join(BASE_DIR,
                       'ch_data_' + args.person + '_' + str(args.index))
    top_k = [0] * 3
    total = 0
    for i, c in enumerate(LETTER):
        for j in range(5):
            path = np.load(os.path.join(dir, c + '_' + str(j) + '.npy'))
            try:
                path_directions = getHVDirections(path)
            except:
                print(c, j)
                continue
            candidates = []
            for ch in LETTER:
                candidates.append(
                    dtw_ndim.distance(path_directions, DIRECTION_PATTERN[ch]))
            sorted_index = np.argsort(candidates)
            for k in range(3):
                if i in sorted_index[:(k + 1)]:
                    top_k[k] += 1
            total += 1
            # print(path_directions)
            # print(candidates)
            # input()
            if i not in sorted_index[:3]:
                print(LETTER[i])
                plotOneLettersCorner(path)
    print(total, top_k)


def calPatternAcc8():
    dir = os.path.join(BASE_DIR,
                       'ch_data_' + args.person + '_' + str(args.index))
    top_k = [0] * 3
    total = 0
    for i, c in enumerate(LETTER):
        for j in range(5):
            path = np.load(os.path.join(dir, c + '_' + str(j) + '.npy'))
            try:
                path_directions = [
                    np.array([
                        np.cos(EIGHT_DIRECTIONS[i]),
                        np.sin(EIGHT_DIRECTIONS[i])
                    ]) for i in get8Directions(path)
                ]
            except:
                print(c, j)
                continue
            candidates = []
            for ch in LETTER:
                candidates.append(
                    dtw_ndim.distance(path_directions, [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[ch]
                    ]))
            sorted_index = np.argsort(candidates)
            for k in range(3):
                if i in sorted_index[:(k + 1)]:
                    top_k[k] += 1
            total += 1
            # print(path_directions)
            # print(candidates)
            # input()
            if i not in sorted_index[:3]:
                print(LETTER[i])
                plotOneLettersCorner8(path)
    print(total, top_k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--person',
                        help='specify the person you want to look into')
    parser.add_argument(
        '-i',
        '--index',
        default=0,
        help='specify the index you want to look into, default as 0')
    parser.add_argument(
        '-d',
        '--direction',
        default=0,
        help='specify the directions you want to look into, default as 0')
    args = parser.parse_args()

    # for duel in product(list(range(5)), list(range(5))):
    #     calCrossAcc(duel)
    # plotAllLettersCorner()
    # calPatternAcc()
    # calPatternAcc8()
    # plot8Directions()
    # plot83Pressure()
    # plotDirections()
    # anovaDirections()
    # plotStartDiff()
    # plotDoubleDirections()
    plotAmplitude()
    # plotPressure()
    # plotDoubleDirectionsCutToSingle()
