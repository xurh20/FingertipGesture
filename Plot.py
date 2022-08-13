import logging
from itertools import product
from math import atan2, sqrt
import numpy as np
import argparse
import os
import json
import re
import matplotlib.pyplot as plt
from dtaidistance import dtw_ndim
from dtw import dtw
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from Calculate import calculatePoints, genPointLabels
from persistence1d import RunPersistence
from queue import PriorityQueue
import seaborn as sns
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from mpl_toolkits.mplot3d import axes3d

BASE_DIR = 'study1'
LETTER = [chr(y) for y in range(97, 123)]
EIGHT_DIRECTIONS = [
    -np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 3, np.pi / 2,
            3 * np.pi / 4
]
FOUR_DIRECTIONS = [i * np.pi / 2 for i in range(4)]
# 各种方向类型的对应角度
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
ORDERS_STR = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th']
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
    'j': np.array([2, 1]),  # 计划修改为只有一笔[1]
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


def pointsNumber():
    # dir = os.path.join(BASE_DIR,
    #                    args.person, '0', str(args.direction))

    value =[]
    for person in os.listdir(BASE_DIR):
        try:
            dir = os.path.join(BASE_DIR, person, '0', str(args.direction))
            print(dir)
            avg_time = 0
            val = []
            for i in range(int(args.direction)):
                for j in range(5):
                    path = np.load(os.path.join(dir, str(i) + '_' + str(j) + '.npy'))
                    x, y, d = getAveragePath(path,
                                     align_to_first=False,
                                     integer=False)
                    x = gaussian_filter1d(x, sigma=5)
                    y = gaussian_filter1d(y, sigma=5)
                # print(len(x))
                # for dot in range(len(x)):
                #     plt.scatter(x[dot], y[dot])
                # plt.show()
                    avg_time = avg_time + len(x)
                avg_time = avg_time / 5
                val.append(avg_time)
            value.append(val)
            # print(avg_time)
        except:
            continue
    print(value)
    plt.style.use('tableau-colorblind10')
    i = 0
    feature = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

    angles = np.linspace(0, 2 * np.pi, len(value[0]), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    feature = np.concatenate((feature, [feature[0]]))  # 对labels进行封闭
    # 绘图
    fig = plt.figure()
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)
    for values in value:
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=str(i))
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, feature)
        i += 1
    ax.grid(True)

    plt.show()


def getAveragePath(path, align_to_first=True, integer=False, truncate=False):
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
    threshold = 0
    for frame in path:
        for y_coordinate in range(len(frame)):
            for x_coordinate in range(len(frame[y_coordinate])):
                threshold += frame[y_coordinate][x_coordinate]
    threshold /= len(path)
    # print(threshold)
    # 算一下压力的平均值只留下大于某一个数值的点
    for frame in path:  # 一次采集
        sum_force = 0
        x_average = 0
        y_average = 0
        for y_coordinate in range(len(frame)):
            for x_coordinate in range(len(frame[y_coordinate])):
                sum_force += frame[y_coordinate][x_coordinate]
        if sum_force > 2 * threshold:  # if(sum_force > 0)  有效采集的阈值
            for y_coordinate in range(len(frame)):
                for x_coordinate in range(len(frame[y_coordinate])):
                    rate = frame[y_coordinate][x_coordinate] / sum_force
                    x_average += rate * x_coordinate  # 取平均值作为该压力点坐标
                    y_average += rate * y_coordinate
            if integer:
                points_x.append(int(x_average * 10))
                points_y.append(int((35 - y_average) * 10))
            else:
                points_x.append(x_average)
                points_y.append(35 - y_average)
            depths.append(sum_force)
    if align_to_first:
        center_x = points_x[0]  # 起始点为(0, 0)，其他的都要减掉
        center_y = points_y[0]
        for i in range(len(points_x)):
            points_x[i] -= center_x
        for i in range(len(points_y)):
            points_y[i] -= center_y
    # 处理头尾问题
    # i = 1
    # while i < len(depths) and depths[i] > depths[i - 1]:  # 找到第一个压力极大值点
    #     i += 1
    # first_extrema = depths[i]
    # while i > 0 and depths[i] >= first_extrema * 0.3:  # >=0 有意忽略一些从一开始就压力很大的数据？？
    #     i -= 1
    # trunc_at_start = i
    # i = len(depths) - 1
    # while i > 1 and depths[i - 1] > depths[i]:  # 找到最后一个压力极小值点
    #     i -= 1
    # last_extrema = depths[i]
    # while i < len(depths) and depths[i] >= last_extrema * 0.7:
    #     i += 1
    # trunc_at_end = i
    # print(trunc_at_start, trunc_at_end)
    # # 尝试使用一些移动速度的差异
    # i = trunc_at_start
    # while i < trunc_at_end and abs(points_x[i+1] - points_x[i]) < 0.05:
    #     i += 1
    # trunc_at_start = i
    # i = trunc_at_end - 1
    # while i > trunc_at_start and abs(points_x[i] - points_x[i-1]) < 0.05:
    #     i -= 1
    # trunc_at_end = i
    # # 考虑取最后的方向
    # while i <= trunc_at_end and abs(points_x[i] - points_x[i-1]) >= 0.05:
    #     i -= 1
    # trunc_at_start = i
    # print(trunc_at_start, trunc_at_end)

    # # 尝试使用一些角度的差异
    # i = 1
    # while i < len(points_x):
    # xlen = 0
    # ylen = 0
    # length = len(points_x)
    # for i in range(1, length):
    #     xlen += abs(points_x[i] - points_x[i - 1])
    #     ylen += abs(points_y[i] - points_y[i - 1])
    # xlen_ave = xlen / (length - 1)
    # ylen_ave = ylen / (length - 1)
    # print("xlen_ave:")
    # print(xlen_ave)
    # print("ylen_ave:")
    # print(ylen_ave)
    # trunc_at_start = 0
    # trunc_at_end = length
    # for i in range(1, length):
    # print("x:")
    # print(abs(points_x[i] - points_x[i - 1]))
    # print("y:")
    # print(abs(points_y[i] - points_y[i - 1]))
    #     if abs(points_x[i] - points_x[i - 1]) < xlen_ave or abs(points_y[i] - points_y[i - 1]) < ylen_ave:
    #         trunc_at_start += 1
    #     else:
    #         break
    # for i in range(length - 1, 0, -1):
    # print("y:")
    # print(abs(points_y[i] - points_y[i - 1]))
    #     if abs(points_x[i] - points_x[i - 1]) < 2*xlen_ave or abs(points_y[i] - points_y[i - 1]) < 2*ylen_ave:
    #         trunc_at_end -= 1
    #     else:
    #         break
    # print(length)
    # print(trunc_at_start, trunc_at_end)
    # 用来看完整图形
    # trunc_at_start = 0
    # trunc_at_end = len(points_x) - 1
    if truncate:
        d = np.array(depths)
        clamped_d = (d - np.min(d)) / (np.max(d) - np.min(d))
        pressure_persistence_pairs = sorted(
            [t for t in RunPersistence(clamped_d) if t[1] > 0.05],
            key=lambda x: x[0])
        if len(pressure_persistence_pairs) <= 2:
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
        elif len(pressure_persistence_pairs) == 3:
            smallest_extrema = depths[pressure_persistence_pairs[1][0]]
            i = 1
            while i < len(depths) and depths[i] < smallest_extrema * 0.3:
                i += 1
            trunc_at_start = i
            i = len(depths) - 1
            while i > trunc_at_start and depths[i] < smallest_extrema * 0.3:
                i -= 1
            trunc_at_end = i
        else:
            ppp = pressure_persistence_pairs[1:-1]
            smallest_extrema = np.min([depths[_[0]] for _ in ppp])
            i = 1
            while i < len(depths) and depths[i] < smallest_extrema * 0.6:
                i += 1
            trunc_at_start = i
            i = len(depths) - 1
            while i > trunc_at_start and depths[i] < smallest_extrema:
                i -= 1
            trunc_at_end = i
        if len(points_x[trunc_at_start:trunc_at_end]) <= 0:
            logging.warning('Empty path extracted!')
    # return np.array(points_x[trunc_at_start:trunc_at_end]), np.array(
    #         points_y[trunc_at_start:trunc_at_end]), np.array(
    #         depths[trunc_at_start:trunc_at_end])
    return np.array(points_x), np.array(points_y), np.array(depths)


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
    ANGLE_THRESHOLD_2 = np.pi / 10
    MERGE_THRESHOLD = 10
    MERGE_DIST_THRESHOLD = 1
    HV_THRESHOLD = 0.6
    HV_AVG_THRESHOLD = 0.5

    def angleDiff(u, v):
        ang1 = np.arctan2(u[1], u[0])
        ang2 = np.arctan2(v[1], v[0])
        return 2 * np.pi - abs(ang1 -
                               ang2) if abs(ang1 -
                                            ang2) >= np.pi else abs(ang1 -
                                                                    ang2)

    # smooth the path
    x, y, d = getAveragePath(path, False)
    x = gaussian_filter1d(x, sigma=8)
    y = gaussian_filter1d(y, sigma=8)
    d = gaussian_filter1d(d, sigma=8)

    # collecting extrema points of pressure
    d = np.array(d)
    clamped_d = (d - np.min(d)) / (np.max(d) - np.min(d))
    pressure_persistence_pairs = sorted(
        [t for t in RunPersistence(clamped_d) if t[1] > 0.05],
        key=lambda x: x[0])
    pre_collected_corners = [int(ppp[0]) for ppp in pressure_persistence_pairs]
    if pre_collected_corners[0] - 0 > MERGE_THRESHOLD:
        pre_collected_corners.insert(0, 0)
    if len(x) - 1 - pre_collected_corners[-1] > MERGE_THRESHOLD:
        pre_collected_corners.append(len(x) - 1)

    # filter the turning points that seperates angles
    debug_dir = []
    debug_dir.append(pre_collected_corners[0])
    for u, v in list(zip(pre_collected_corners[:-1],
                         pre_collected_corners[1:])):
        i = u + 1
        while i < v:
            cur_v = (x[i] - x[i - 1], y[i] - y[i - 1])
            j = i
            while j < v and angleDiff(
                    (x[j] - x[j - 1], y[j] - y[j - 1]),
                    cur_v) < ANGLE_THRESHOLD and angleDiff(
                (x[j] - x[j - 1], y[j] - y[j - 1]),
                (x[i] - x[i - 1], y[i] - y[i - 1])) < ANGLE_THRESHOLD:
                cur_v = (x[j] - x[i - 1], y[j] - y[i - 1])
                j += 1
            debug_dir.append(j - 1)
            i = j

    # merge redundant points caused by overreact
    real_debug_dir = []
    for dd in debug_dir:
        if len(real_debug_dir) <= 1:
            real_debug_dir.append(dd)
            continue
        if angleDiff((x[real_debug_dir[-1]] - x[real_debug_dir[-2]],
                      y[real_debug_dir[-1]] - y[real_debug_dir[-2]]),
                     (x[dd] - x[real_debug_dir[-1]],
                      y[dd] - y[real_debug_dir[-1]])) <= ANGLE_THRESHOLD_2:
            real_debug_dir[-1] = dd
        else:
            real_debug_dir.append(dd)
    debug_dir = real_debug_dir

    # i = 1
    # while i < len(x):
    #     cur_v = (x[i] - x[i - 1], y[i] - y[i - 1])
    #     j = i
    #     while j < len(x) and angleDiff(
    #         (x[j] - x[j - 1], y[j] - y[j - 1]),
    #             cur_v) < ANGLE_THRESHOLD and angleDiff(
    #                 (x[j] - x[j - 1], y[j] - y[j - 1]),
    #                 (x[i] - x[i - 1], y[i] - y[i - 1])) < ANGLE_THRESHOLD:
    #         cur_v = (x[j] - x[i - 1], y[j] - y[i - 1])
    #         j += 1
    #     debug_dir.append(j-1)
    #     i = j

    # merge points that are enough close to each other
    i = 0
    simplified_dir = []
    while i < len(debug_dir):
        t = []
        lat = debug_dir[i]
        while i < len(debug_dir) and (
                debug_dir[i] - lat < MERGE_THRESHOLD or
                (x[debug_dir[i]] - x[lat]) ** 2 +
                (y[debug_dir[i]] - y[lat]) ** 2 < MERGE_DIST_THRESHOLD):
            t.append(debug_dir[i])
            lat = debug_dir[i]
            i += 1
        simplified_dir.append(int(np.mean(t)))

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
        if x_off > y_off and x_off > x_avg * HV_AVG_THRESHOLD and x_off > HV_THRESHOLD:
            filtered_dir.append(sd)
        elif y_off > x_off and y_off > y_avg * HV_AVG_THRESHOLD and y_off > HV_THRESHOLD:
            filtered_dir.append(sd)
        elif x_off == y_off and (
                (x_off > x_avg * HV_AVG_THRESHOLD and x_off > HV_THRESHOLD) or
                (y_off > y_avg * HV_AVG_THRESHOLD and y_off > HV_THRESHOLD)):
            filtered_dir.append(sd)

    return simplified_dir


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


def getSingleDirectionConfidenceList(v):
    gauss_dict = {}
    with open('gauss_direction_mix.json', 'r') as file:
        gauss_dict = json.load(file)

    ang = np.arctan2(v[1], v[0])
    # if abs(ang) > 7 * np.pi / 8:
    #     return 0
    # ans = np.argmin([abs(ang - std_ang) for std_ang in EIGHT_DIRECTIONS])
    confidence_list = []
    for ix in range(8):
        if ix == 0:
            val = np.exp(
                -((ang - gauss_dict['0'][1]) / gauss_dict['0'][2]) ** 2 / 2)
            val_adj = np.exp(-(
                                      (ang - 2 * np.pi - gauss_dict['0'][1]) / gauss_dict['0'][2]) ** 2
                             / 2)
            confidence_list.append((gauss_dict['0'][0], max(val, val_adj)))
        elif ix == 7:
            val = np.exp(
                -((ang - gauss_dict['7'][1]) / gauss_dict['7'][2]) ** 2 / 2)
            val_adj = np.exp(-(
                                      (ang + 2 * np.pi - gauss_dict['7'][1]) / gauss_dict['7'][2]) ** 2
                             / 2)
            confidence_list.append((gauss_dict['7'][0], max(val, val_adj)))
        else:
            confidence_list.append((gauss_dict[str(ix)][0],
                                    np.exp(-((ang - gauss_dict[str(ix)][1]) /
                                             gauss_dict[str(ix)][2]) ** 2 / 2)))
    return sorted(confidence_list, key=lambda t: t[1], reverse=True)


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
    Tuple of directions indexes, arrays of directions, weights of directions

    """

    x, y, d = getAveragePath(path)
    if (len(x) <= 0):
        return [], np.array([2]), []
    simplified_dir = getCorners(path)
    directions_index = []
    directions = []
    directions_weights = []

    singleDCList = []
    for u, v in list(zip(simplified_dir[:-1], simplified_dir[1:])):
        singleDCList.append(
            getSingleDirectionConfidenceList((x[v] - x[u], y[v] - y[u])))

    PROBABILITY_THRESHOLD = 0.1
    # TODO: Return a confidence list
    for i, (u,
            v) in enumerate(list(zip(simplified_dir[:-1],
                                     simplified_dir[1:]))):
        closest_direction = singleDCList[i][0][0]
        if len(directions) <= 0 or closest_direction != directions[-1]:
            directions_index.append((u, v))
            directions.append(closest_direction)
            directions_weights.append((x[v] - x[u]) ** 2 + (y[v] - y[u]) ** 2)
        elif len(directions) > 0 and directions[-1] == closest_direction:
            directions_index[-1] = (directions_index[-1][0], v)
            directions_weights[-1] = (x[v] - x[directions_index[-1][0]]) ** 2 + (
                    y[v] - y[directions_index[-1][0]]) ** 2
    directions_weights = np.array(directions_weights) / np.sum(
        directions_weights)

    return directions_index, np.array(directions), directions_weights


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
            plotOneLettersCorner8(path)
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
    fig, axes = plt.subplots(1, 2)
    axes[0].axis("scaled")
    axes[0].set_xlim(10, 17)
    axes[0].set_ylim(15, 25)
    axes[0].scatter(x, y, c='blue')
    axes[1].scatter(list(range(len(d))), d)
    for _, corner in enumerate(corners):
        axes[0].scatter([x[corner]], [y[corner]], c='red')
        axes[0].text(x[corner], y[corner], str(corner))
        axes[1].scatter([corner], [d[corner]], c='red')
        axes[1].text(corner, d[corner], str(_))

    clamped_d = (d - np.min(d)) / (np.max(d) - np.min(d))
    pressure_persistence_pairs = sorted(
        [t for t in RunPersistence(clamped_d) if t[1] > 0.1],
        key=lambda x: x[0])
    for (pressure_ex, persistence) in pressure_persistence_pairs:
        axes[1].scatter([pressure_ex], [d[pressure_ex]], c='blue')
        axes[1].text(pressure_ex, d[pressure_ex], str(persistence))
    print(get8Directions(path))
    # print(np.mean(d))
    plt.show()
    # from IPython.terminal import embed, pt_inputhooks
    # shell = embed.InteractiveShellEmbed.instance()
    # shell()


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


def plotElapsed():
    """
    description
    ---------
    Plot elapsed time of given directions

    param
    -------
    None

    Returns
    -------
    None

    """
    num_d = int(args.direction)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(111, polar=True)
    angles = DIRECTIONS_MAP[args.direction]
    angles = np.concatenate((angles, [angles[0]]))
    for dir in os.listdir(BASE_DIR):
        elapsed = [[] for _ in range(num_d)]
        if not 'ch_data_' + args.direction + '_dir_' in dir:
            continue
        for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
            for j in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir,
                                     str(i) + '_{}.npy'.format(j)))
                    x, y, d = getAveragePath(path,
                                             align_to_first=False,
                                             integer=False,
                                             truncate=False)
                except:
                    continue

                elapsed[i].append(len(x))
        print(elapsed)
        avg_ela = []
        for i in range(num_d):
            avg_ela.append(np.average(elapsed[i]))
        avg_ela = np.concatenate((avg_ela, [avg_ela[0]]))
        ax.plot(angles, avg_ela)
        ax.set_theta_zero_location('E')
    plt.show()


def plotTendency():
    """
    description
    ---------
    Plot tendency preference of given directions

    param
    -------
    None

    Returns
    -------
    None

    """
    num_d = int(args.direction)
    fig, ax = plt.subplots(2, 4)
    for dir in os.listdir(BASE_DIR):
        if not 'ch_data_' + args.direction + '_dir_' in dir:
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
                ax[i // 4][i % 4].scatter(np.array(x[start:end]) - x[start], np.array(y[start:end]) - y[start],
                                          c=COLORS[i])
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
    num_d = int(args.direction)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(111, polar=True)
    angles = DIRECTIONS_MAP[args.direction]
    angles = np.concatenate((angles, [angles[0]]))
    plt.style.use('tableau-colorblind10')
    # 计算同一方向不同的人的标准差
    print("共有"+str(args.direction)+"个方向")
    for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
        amp = []
        for j in range(5):
            for dir in os.listdir(BASE_DIR):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, '0', str(args.direction),
                                     str(i) + '_' + str(j) + '.npy'))
                    x, y, d = getAveragePath(path,
                                             align_to_first=False,
                                             integer=False)
                    x = gaussian_filter1d(x, sigma=5)
                    y = gaussian_filter1d(y, sigma=5)
                    start, end = getLongestDirection(path)
                except:
                    continue
                if end < 0:
                    continue
                angle = np.arctan2(y[end] - y[start], x[end] - x[start])
                # if angle > (DIRECTIONS_MAP[args.direction][-1] + np.pi) / 2:
                if i == 0 and angle > 0:
                    angle -= 2 * np.pi
                amp.append(np.sqrt((x[end] - x[start]) ** 2 + (y[end] - y[start]) ** 2))
        print('方向为'+str(i)+'标准差为'+str(np.array(amp).std()))
    # for dir in os.listdir(BASE_DIR):
    #     amplitude = [[] for _ in range(num_d)]
    #     print(dir)
    #     for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
    #         for j in range(5):
    #             try:
    #                 path = np.load(
    #                     os.path.join(BASE_DIR, dir, '0', str(args.direction),
    #                                  str(i) + '_' + str(j) + '.npy'))
    #                 x, y, d = getAveragePath(path,
    #                                          align_to_first=False,
    #                                          integer=False)
    #                 x = gaussian_filter1d(x, sigma=5)
    #                 y = gaussian_filter1d(y, sigma=5)
    #                 start, end = getLongestDirection(path)
    #                 # start, end = getStartAndEnd(args.direction, x, y, i)
    #             except:
    #                 continue
    #             if end < 0:
    #                 continue
    #             angle = np.arctan2(y[end] - y[start], x[end] - x[start])
    #             # if angle > (DIRECTIONS_MAP[args.direction][-1] + np.pi) / 2:
    #             if i == 0 and angle > 0:
    #                 angle -= 2 * np.pi
    #             # plt.axis("scaled")
    #             # plt.scatter([np.cos(angle)], [np.sin(angle)], c=COLORS[i])
    #
    #             # if i == 0:
    #             #     print(
    #             #         os.path.join(BASE_DIR, dir,
    #             #                      str(i) + '_{}.npy'.format(j)))
    #             #     print(c, angle)
    #             #     plt.axis("scaled")
    #             #     plt.xlim(10, 17)
    #             #     plt.ylim(15, 25)
    #             #     plt.scatter(x, y, c='blue')
    #             #     plt.scatter([x[start], x[end]], [y[start], y[end]],
    #             #                 c='red')
    #             #     plt.show()
    #             # plt.scatter([x[end] - x[start]], [y[end] - y[start]],
    #             #             c=COLORS[i])
    #             amplitude[i].append(np.sqrt((x[end] - x[start]) ** 2 + (y[end] - y[start]) ** 2))
    #     avg_amp = []
    #     for i in range(num_d):
    #         avg_amp.append(np.average(amplitude[i]))
    #     avg_amp = np.concatenate((avg_amp, [avg_amp[0]]))
    #     ax.plot(angles, avg_amp)
    #     ax.set_theta_zero_location('W')
    # plt.show()


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
    num_d = int(args.direction)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax1 = plt.subplot(121, polar=True)
    ax2 = plt.subplot(122, polar=True)
    angles = DIRECTIONS_MAP[args.direction]
    angles = np.concatenate((angles, [angles[0]]))
    max_p = []
    avg_p = []
    dir_p = []
    exp_p = []
    # 计算同一方向不同的人的标准差
    print("共有" + str(args.direction) + "个方向")
    for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
        press = []
        for j in range(5):
            for dir in os.listdir(BASE_DIR):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, '0', str(args.direction),
                                     str(i) + '_' + str(j) + '.npy'))
                    x, y, d = getAveragePath(path,
                                             align_to_first=False,
                                             integer=False)
                    x = gaussian_filter1d(x, sigma=5)
                    y = gaussian_filter1d(y, sigma=5)
                    start, end = getLongestDirection(path)
                except:
                    continue
                if end < 0:
                    continue
                angle = np.arctan2(y[end] - y[start], x[end] - x[start])
                # if angle > (DIRECTIONS_MAP[args.direction][-1] + np.pi) / 2:
                if i == 0 and angle > 0:
                    angle -= 2 * np.pi
                press.append(np.average(d))
                # print(press)
        print('方向为' + str(i) + '标准差为' + str(np.array(press).std()))
    for dir in os.listdir(BASE_DIR):
        print(dir)
        pressure_m = [[] for _ in range(num_d)]
        pressure_a = [[] for _ in range(num_d)]
        for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
            for j in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, '0', str(args.direction),
                                     str(i) + '_' + str(j) + '.npy'))
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
                clamp_d = (d - np.min(d)) / (np.max(d) - np.min(d))
                plt.scatter([list(range(end - start + 1))],
                            [clamp_d[start:end + 1]],
                            c=COLORS[i])
                max_p.append(np.max(d))
                avg_p.append(np.average(d))
                dir_p.append(i)
                exp_p.append(dir)
                pressure_m[i].append(np.max(d))
                pressure_a[i].append(np.average(d))
        pma, paa = [], []
        for i in range(num_d):
            pma.append(np.average(pressure_m[i]))
            paa.append(np.average(pressure_a[i]))
        pma = np.concatenate((pma, [pma[0]]))
        paa = np.concatenate((paa, [paa[0]]))
        ax1.plot(angles, pma)
        ax2.plot(angles, paa)
        ax1.set_theta_zero_location('E')
        ax2.set_theta_zero_location('E')
    plt.show()

    df = pd.DataFrame({
        'max_pressure': max_p,
        'average_pressure': avg_p,
        'direction': dir_p,
        'experiment': exp_p
    })
    fig, axes = plt.subplots()
    sns.boxplot(x='direction',
                y='max_pressure',
                hue='experiment',
                data=df,
                ax=axes)
    plt.show()


def anovaDirections():
    """
    description
    ---------
    Analyze anova of distinguishibility between given directions
    
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
        print(dir)
        for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
            for j in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, '0', str(args.direction),
                                     str(i) + '_' + str(j) + '.npy'))
                    # print(path)
                    x, y, d = getAveragePath(path,
                                             align_to_first=False,
                                             integer=False)
                    x = gaussian_filter1d(x, sigma=5)
                    y = gaussian_filter1d(y, sigma=5)
                    # print("fuck")
                    # gc = plt.gca()
                    # gc.xaxis.set_major_locator(plt.MultipleLocator(1))
                    # gc.yaxis.set_major_locator(plt.MultipleLocator(1))
                    # plt.scatter(x, y)
                    # plt.title(c)
                    # plt.show()
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
    print(mc.tukeyhsd())  # 两两比较看差异


def plotChar():
    for i in LETTER:
        for j in range(0, 5):
            path = np.load(
                os.path.join(BASE_DIR, "ch_data_letter_dir_3",
                             '{}_{}.npy'.format(i, j)))
            x, y, d = getAveragePath(path,
                                     align_to_first=False,
                                     integer=False)
            x = gaussian_filter1d(x, sigma=5)
            y = gaussian_filter1d(y, sigma=5)
            # x_even, y_even, x_sin, y_sin = [], [], [], []
            # for k in range(len(x)):
            #     if k%2 == 0:
            #         x_even.append(x[k])
            #         y_even.append(y[k])
            #     else:
            #         x_sin.append(x[k])
            #         y_sin.append(y[k])
            # plt.scatter(x_even, y_even, c='r')
            # plt.scatter(x_sin, y_sin, c='g')
            for dot in range(len(x)):
                plt.scatter(x[dot], y[dot])
                plt.annotate(dot, (x[dot], y[dot]))
            # plt.scatter(x, y)
            plt.title(i)
            plt.show()


def gaussianDirections():
    """
    description
    ---------
    Fit gaussian for every direction

    param
    -------
    None

    Returns
    -------
    None

    """

    avg_angles = [[] for _ in range(8)]
    for dir in os.listdir(BASE_DIR):
        if not 'ch_data_' + str('8') + '_dir_' in dir:
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

                avg_angles[i].append(angle)

    with open('gauss_direction.json', 'w') as output:
        gauss_dict = {}
        for ix in range(8):
            gauss_dict[ix] = (ix, np.mean(avg_angles[ix]),
                              np.std(avg_angles[ix]))
        json.dump(gauss_dict, output)


def gaussianDirectionsMultiple():
    """
    description
    ---------
    Fit gaussian for every direction

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

    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    avg_angles = [[] for _ in range(8)]
    for dir in os.listdir(BASE_DIR):
        if not 'letter_' in dir or not os.path.isdir(
                os.path.join(BASE_DIR, dir)):
            continue
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, "%s_%d.npy" % (t_l, rep)))

                    x, y, d = getAveragePath(path, align_to_first=False)
                    corners = getCorners(path)
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    identified_directions_index = []

                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in redundant_8directions
                    ]
                    std_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[t_l]
                    ]
                    paths = dtw_ndim.warping_path(path_directions,
                                                  std_directions)
                    idx = 0
                    while idx < len(paths):
                        match_list = []
                        current_std_idx = paths[idx][1]
                        while idx < len(
                                paths) and paths[idx][1] == current_std_idx:
                            match_list.append(paths[idx][0])
                            idx += 1
                        identified_directions_index.append(
                            directions_index[match_list[np.argmin([
                                angleDist(path_directions[m_l],
                                          std_directions[current_std_idx])
                                for m_l in match_list
                            ])]])

                    std_angles_set = [i for i in DIRECTION_PATTERN8[t_l]]
                    for ix, (iu, iv) in enumerate(identified_directions_index):
                        avg_angles[std_angles_set[ix]].append(
                            getAngle(x[iu], x[iv], y[iu], y[iv]))
                except Exception as e:
                    print(str(e))

    with open('gauss_direction_multiple.json', 'w') as output:
        gauss_dict = {}
        for ix in range(8):
            gauss_dict[ix] = (ix, np.mean(avg_angles[ix]),
                              np.std(avg_angles[ix]))
        json.dump(gauss_dict, output)


def visualizeGaussianDirections():
    thetas = np.linspace(0, 2 * np.pi, 1000)
    plt.axis("scaled")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    for theta in thetas:
        direction, confidence = getSingleDirectionConfidenceList(
            (np.cos(theta), np.sin(theta)))[0]
        plt.scatter([np.cos(theta)], [np.sin(theta)],
                    c=COLORS[direction],
                    alpha=confidence)
    plt.show()


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


def getIdentifiedGodPath(path, t_l):
    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    directions_index, redundant_8directions, weights = get8Directions(path)
    identified_directions_index = []

    path_directions = [
        np.array([np.cos(EIGHT_DIRECTIONS[i]),
                  np.sin(EIGHT_DIRECTIONS[i])]) for i in redundant_8directions
    ]
    std_directions = [
        np.array([np.cos(EIGHT_DIRECTIONS[i]),
                  np.sin(EIGHT_DIRECTIONS[i])])
        for i in DIRECTION_PATTERN8[t_l]
    ]
    paths = dtw_ndim.warping_path(path_directions, std_directions)
    idx = 0
    while idx < len(paths):
        match_list = []
        current_std_idx = paths[idx][1]
        while idx < len(paths) and paths[idx][1] == current_std_idx:
            match_list.append(paths[idx][0])
            idx += 1
        identified_directions_index.append(
            directions_index[match_list[np.argmin([
                angleDist(path_directions[m_l],
                          std_directions[current_std_idx])
                for m_l in match_list
            ])]])
    return identified_directions_index


def plotMultipleDirectionsCutToSingle():
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

    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    avg_angles = []
    std_angles = []
    orders = []
    for dir in os.listdir(BASE_DIR):
        if not 'letter_' in dir or not os.path.isdir(
                os.path.join(BASE_DIR, dir)):
            continue
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, "%s_%d.npy" % (t_l, rep)))

                    x, y, d = getAveragePath(path, align_to_first=False)
                    corners = getCorners(path)
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    identified_directions_index = []

                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in redundant_8directions
                    ]
                    std_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[t_l]
                    ]
                    paths = dtw_ndim.warping_path(path_directions,
                                                  std_directions)
                    idx = 0
                    while idx < len(paths):
                        match_list = []
                        current_std_idx = paths[idx][1]
                        while idx < len(
                                paths) and paths[idx][1] == current_std_idx:
                            match_list.append(paths[idx][0])
                            idx += 1
                        identified_directions_index.append(
                            directions_index[match_list[np.argmin([
                                angleDist(path_directions[m_l],
                                          std_directions[current_std_idx])
                                for m_l in match_list
                            ])]])

                    # plt.axis("scaled")
                    # plt.xlim(10, 17)
                    # plt.ylim(15, 25)
                    # plt.scatter(x, y, c='blue')
                    # for _, corner in enumerate(corners):
                    #     plt.scatter([x[corner]], [y[corner]], c='red')
                    #     plt.text(x[corner], y[corner], str(_))
                    # for iu, iv in identified_directions_index:
                    #     plt.plot([x[iu], x[iv]], [y[iu], y[iv]], c='green')
                    # plt.show()

                    std_angles_set = [
                        EIGHT_DIRECTIONS[i] for i in DIRECTION_PATTERN8[t_l]
                    ]
                    for ix, (iu, iv) in enumerate(identified_directions_index):
                        std_angles.append(std_angles_set[ix])
                        avg_angles.append(getAngle(x[iu], x[iv], y[iu], y[iv]))
                        orders.append(ORDERS_STR[ix])
                except Exception as e:
                    print(str(e))

    # plt.axis("scaled")
    # plt.xlim(-np.pi / 4, 2 * np.pi)
    # plt.ylim(-np.pi / 4, 2 * np.pi)
    # plt.scatter(std_angles, avg_angles)
    # plt.show()
    for direction in EIGHT_DIRECTIONS:
        _indexes = np.where(np.array(std_angles) == direction)[0]
        _usr_angles = [avg_angles[idxx] for idxx in _indexes]
        _orders = [orders[idxx] for idxx in _indexes]
        print(direction)
        df = pd.DataFrame({'usr_angle': _usr_angles, 'order': _orders})
        model = ols('usr_angle~C(order)', data=df).fit()
        anova_table = anova_lm(model, typ=2)
        print(anova_table)
        mc = MultiComparison(_usr_angles, _orders)
        print(mc.tukeyhsd())

    df = pd.DataFrame({
        'std_angle': std_angles,
        'usr_angle': avg_angles,
        'order': orders
    })
    fig, axes = plt.subplots()
    # sns.boxplot(x='std_angle', y='usr_angle', hue='order', data=df, ax=axes)
    sns.boxplot(x='std_angle', y='usr_angle', data=df, ax=axes)
    plt.show()
    model = ols('usr_angle~C(std_angle)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    print(anova_table)
    mc = MultiComparison(avg_angles, std_angles)
    print(mc.tukeyhsd())


def plotMultipleDirectionsCutToSingleIncludedAngles():
    """
    description
    ---------
    Plot included angles cutting directions adopting to 8 directions

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

    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    usr_angles = []
    std_angles = []
    for dir in os.listdir(BASE_DIR):
        if not 'letter_' in dir or not os.path.isdir(
                os.path.join(BASE_DIR, dir)):
            continue
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, "%s_%d.npy" % (t_l, rep)))

                    x, y, d = getAveragePath(path, align_to_first=False)
                    corners = getCorners(path)
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    identified_directions_index = []

                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in redundant_8directions
                    ]
                    std_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[t_l]
                    ]
                    paths = dtw_ndim.warping_path(path_directions,
                                                  std_directions)
                    idx = 0
                    while idx < len(paths):
                        match_list = []
                        current_std_idx = paths[idx][1]
                        while idx < len(
                                paths) and paths[idx][1] == current_std_idx:
                            match_list.append(paths[idx][0])
                            idx += 1
                        identified_directions_index.append(
                            directions_index[match_list[np.argmin([
                                angleDist(path_directions[m_l],
                                          std_directions[current_std_idx])
                                for m_l in match_list
                            ])]])

                    # plt.axis("scaled")
                    # plt.xlim(10, 17)
                    # plt.ylim(15, 25)
                    # plt.scatter(x, y, c='blue')
                    # for _, corner in enumerate(corners):
                    #     plt.scatter([x[corner]], [y[corner]], c='red')
                    #     plt.text(x[corner], y[corner], str(_))
                    # for iu, iv in identified_directions_index:
                    #     plt.plot([x[iu], x[iv]], [y[iu], y[iv]], c='green')
                    # plt.show()

                    std_angles_set = [
                        EIGHT_DIRECTIONS[i] for i in DIRECTION_PATTERN8[t_l]
                    ]
                    if (len(std_angles_set) < 2):
                        continue
                    for (usr, std) in list(
                            zip(
                                zip(identified_directions_index[1:],
                                    identified_directions_index[:-1]),
                                zip(std_angles_set[1:], std_angles_set[:-1]))):
                        def angleDiff(ang1, ang2):
                            return 2 * np.pi - abs(ang1 - ang2) if abs(
                                ang1 - ang2) >= np.pi else abs(ang1 - ang2)

                        def included_angle(x1, y1, x2, y2):
                            u = np.array([x1, y1])
                            v = np.array([x2, y2])
                            return np.arccos(
                                np.dot(u, v) / np.linalg.norm(u) /
                                np.linalg.norm(v))

                        std_angles.append(angleDiff(std[0], std[1]))
                        usr_angles.append(
                            included_angle(x[usr[0][1]] - x[usr[0][0]],
                                           y[usr[0][1]] - y[usr[0][0]],
                                           x[usr[1][1]] - x[usr[1][0]],
                                           y[usr[1][1]] - y[usr[1][0]]))
                except Exception as e:
                    print(str(e))

    df = pd.DataFrame({'std_angle': std_angles, 'usr_angle': usr_angles})
    fig, axes = plt.subplots()
    # sns.boxplot(x='std_angle', y='usr_angle', hue='order', data=df, ax=axes)
    sns.boxplot(x='std_angle', y='usr_angle', data=df, ax=axes)
    plt.show()
    model = ols('usr_angle~C(std_angle)', data=df).fit()
    anova_table = anova_lm(model, typ=2)
    print(anova_table)
    mc = MultiComparison(usr_angles, std_angles)
    print(mc.tukeyhsd())


def plotMultipleDirectionsCutToSingleAmplitude():
    """
    description
    ---------
    Plot Amplitude of cutting directions adopting to 8 directions

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

    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    plt.axis("scaled")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    for dir in os.listdir(BASE_DIR):
        if not 'letter_' in dir or not os.path.isdir(
                os.path.join(BASE_DIR, dir)):
            continue
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, "%s_%d.npy" % (t_l, rep)))

                    x, y, d = getAveragePath(path, align_to_first=False)
                    corners = getCorners(path)
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    identified_directions_index = []

                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in redundant_8directions
                    ]
                    std_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[t_l]
                    ]
                    paths = dtw_ndim.warping_path(path_directions,
                                                  std_directions)
                    idx = 0
                    while idx < len(paths):
                        match_list = []
                        current_std_idx = paths[idx][1]
                        while idx < len(
                                paths) and paths[idx][1] == current_std_idx:
                            match_list.append(paths[idx][0])
                            idx += 1
                        identified_directions_index.append(
                            directions_index[match_list[np.argmin([
                                angleDist(path_directions[m_l],
                                          std_directions[current_std_idx])
                                for m_l in match_list
                            ])]])

                    # plt.axis("scaled")
                    # plt.xlim(10, 17)
                    # plt.ylim(15, 25)
                    for ix, (iu, iv) in enumerate(identified_directions_index):
                        plt.scatter([x[iv] - x[iu]], [y[iv] - y[iu]],
                                    c=COLORS[DIRECTION_PATTERN8[t_l][ix]])
                except Exception as e:
                    print(str(e))
    plt.show()


def plotMultipleDirectionsCutToSinglePressure():
    """
    description
    ---------
    Plot presssure of cutting directions adopting to 8 directions

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

    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    # plt.axis("scaled")
    # plt.xlim(10, 17)
    # plt.ylim(15, 25)
    pressure_dict = [[[[] for ___ in range(8)] for _ in range(5)]
                     for __ in range(5)]

    for dir in os.listdir(BASE_DIR):
        if not 'letter_' in dir or not os.path.isdir(
                os.path.join(BASE_DIR, dir)):
            continue
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, "%s_%d.npy" % (t_l, rep)))

                    x, y, d = getAveragePath(path, align_to_first=False)
                    corners = getCorners(path)
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    identified_directions_index = []

                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in redundant_8directions
                    ]
                    std_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[t_l]
                    ]
                    paths = dtw_ndim.warping_path(path_directions,
                                                  std_directions)
                    idx = 0
                    while idx < len(paths):
                        match_list = []
                        current_std_idx = paths[idx][1]
                        while idx < len(
                                paths) and paths[idx][1] == current_std_idx:
                            match_list.append(paths[idx][0])
                            idx += 1
                        identified_directions_index.append(
                            directions_index[match_list[np.argmin([
                                angleDist(path_directions[m_l],
                                          std_directions[current_std_idx])
                                for m_l in match_list
                            ])]])

                    # plt.axis("scaled")
                    # plt.xlim(10, 17)
                    # plt.ylim(15, 25)
                    for ix, (iu, iv) in enumerate(identified_directions_index):
                        pressure_list = d[iu:iv]
                        pressure_dict[len(std_directions) - 1][ix][
                            DIRECTION_PATTERN8[t_l][ix]].append(pressure_list)
                except Exception as e:
                    print(str(e))

    # plot for every direction
    fig, axes = plt.subplots(5, 5)
    for total_stroke_num in range(5):
        for stroke_idx in range(5):
            for stroke_dir in range(8):
                pressure_list = pressure_dict[total_stroke_num][stroke_idx][
                    stroke_dir]
                if len(pressure_list) <= 0:
                    continue
                for p_l in pressure_list:
                    axes[total_stroke_num][stroke_idx].plot(
                        list(range(len(p_l))), p_l, c=COLORS[stroke_dir])
    plt.show()


def plotMultipleDirectionsCutToSinglePressureExtrema():
    """
    description
    ---------
    Plot cutting directions adopting to 8 directions with pressure Extrema

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

    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    total_corners = 0
    corners_in_pressure_ex = [0] * 10
    total_pressure_ex = 0
    pressure_ex_in_corners = [0] * 10
    for dir in os.listdir(BASE_DIR):
        if not 'letter_dir_1' in dir or not os.path.isdir(
                os.path.join(BASE_DIR, dir)):
            continue
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, "%s_%d.npy" % (t_l, rep)))

                    x, y, d = getAveragePath(path, align_to_first=False)
                    corners = getCorners(path)
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    identified_directions_index = []

                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in redundant_8directions
                    ]
                    std_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[t_l]
                    ]
                    paths = dtw_ndim.warping_path(path_directions,
                                                  std_directions)

                    idx = 0
                    while idx < len(paths):
                        match_list = []
                        current_std_idx = paths[idx][1]
                        while idx < len(
                                paths) and paths[idx][1] == current_std_idx:
                            match_list.append(paths[idx][0])
                            idx += 1
                        identified_directions_index.append(
                            directions_index[match_list[np.argmin([
                                angleDist(path_directions[m_l],
                                          std_directions[current_std_idx])
                                for m_l in match_list
                            ])]])

                    fig, axes = plt.subplots(1, 2)
                    axes[0].axis("scaled")
                    axes[0].set_xlim(10, 17)
                    axes[0].set_ylim(15, 25)
                    axes[0].scatter(x, y, c='blue')
                    axes[1].scatter(list(range(len(d))), d)
                    for _, corner in enumerate(corners):
                        axes[0].scatter([x[corner]], [y[corner]], c='red')
                        axes[0].text(x[corner], y[corner], str(_))
                        axes[1].scatter([corner], [d[corner]], c='red')
                        axes[1].text(corner, d[corner], str(_))
                    for iu, iv in identified_directions_index:
                        axes[0].plot([x[iu], x[iv]], [y[iu], y[iv]], c='green')

                    clamped_d = (d - np.min(d)) / (np.max(d) - np.min(d))
                    pressure_persistence_pairs = sorted(
                        [t for t in RunPersistence(clamped_d) if t[1] > 0.1],
                        key=lambda x: x[0])
                    for (pressure_ex,
                         persistence) in pressure_persistence_pairs:
                        axes[1].scatter([pressure_ex], [d[pressure_ex]],
                                        c='blue')
                        axes[1].text(pressure_ex, d[pressure_ex],
                                     str(persistence))

                    plt.show()

                    for corn in corners:
                        min_dist = np.min([
                            abs(corn - ppp[0])
                            for ppp in pressure_persistence_pairs
                        ])
                        total_corners += 1
                        if min_dist < 10:
                            corners_in_pressure_ex[min_dist] += 1
                    for (pressure_ex,
                         persistence) in pressure_persistence_pairs:
                        min_dist = np.min(
                            [abs(pressure_ex - corn) for corn in corners])
                        total_pressure_ex += 1
                        if min_dist < 10:
                            pressure_ex_in_corners[min_dist] += 1
                except Exception as e:
                    print(str(e))

    print(total_corners, corners_in_pressure_ex)
    print(total_pressure_ex, pressure_ex_in_corners)


def migrateSingleAndMultiple():
    def getAngle(x1, x2, y1, y2):
        angle = np.arctan2(y2 - y1, x2 - x1)
        if angle > (DIRECTIONS_MAP['8'][-1] + np.pi) / 2:
            angle -= 2 * np.pi
        return angle

    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    avg_angles = []
    std_angles = []
    cat = []
    usr_amp = []
    usr_pres = []
    max_p = []
    avg_p = []
    dir_p = []
    for dir in os.listdir(BASE_DIR):
        if not 'ch_data_' + '8' + '_dir_' in dir:
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

                avg_angles.append(angle)
                std_angles.append(c)
                usr_amp.append(
                    np.linalg.norm((x[end] - x[start], y[end] - y[start])))
                usr_pres.append(d[start:end])
                max_p.append(np.max(d[start:end]))
                avg_p.append(np.average(d[start:end]))
                dir_p.append(i)
                cat.append('Single')
    for dir in os.listdir(BASE_DIR):
        if not 'letter_' in dir or not os.path.isdir(
                os.path.join(BASE_DIR, dir)):
            continue
        for t_l_i, t_l in enumerate(LETTER):
            for rep in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, "%s_%d.npy" % (t_l, rep)))

                    x, y, d = getAveragePath(path, align_to_first=False)
                    corners = getCorners(path)
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    identified_directions_index = []

                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in redundant_8directions
                    ]
                    std_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[t_l]
                    ]
                    paths = dtw_ndim.warping_path(path_directions,
                                                  std_directions)
                    idx = 0
                    while idx < len(paths):
                        match_list = []
                        current_std_idx = paths[idx][1]
                        while idx < len(
                                paths) and paths[idx][1] == current_std_idx:
                            match_list.append(paths[idx][0])
                            idx += 1
                        identified_directions_index.append(
                            directions_index[match_list[np.argmin([
                                angleDist(path_directions[m_l],
                                          std_directions[current_std_idx])
                                for m_l in match_list
                            ])]])

                    # plt.axis("scaled")
                    # plt.xlim(10, 17)
                    # plt.ylim(15, 25)
                    # plt.scatter(x, y, c='blue')
                    # for _, corner in enumerate(corners):
                    #     plt.scatter([x[corner]], [y[corner]], c='red')
                    #     plt.text(x[corner], y[corner], str(_))
                    # for iu, iv in identified_directions_index:
                    #     plt.plot([x[iu], x[iv]], [y[iu], y[iv]], c='green')
                    # plt.show()

                    std_angles_set = [
                        EIGHT_DIRECTIONS[i] for i in DIRECTION_PATTERN8[t_l]
                    ]
                    for ix, (iu, iv) in enumerate(identified_directions_index):
                        std_angles.append(std_angles_set[ix])
                        avg_angles.append(getAngle(x[iu], x[iv], y[iu], y[iv]))
                        usr_amp.append(
                            np.linalg.norm((x[iv] - x[iu], y[iv] - y[iu])))
                        usr_pres.append(d[iu:iv])
                        max_p.append(np.max(d[iu:iv]))
                        avg_p.append(np.average(d[iu:iv]))
                        dir_p.append(DIRECTION_PATTERN8[t_l][ix])
                        cat.append('Multiple')
                except Exception as e:
                    print(str(e))

    # # direction
    # for direction in EIGHT_DIRECTIONS:
    #     _indexes = np.where(np.array(std_angles) == direction)[0]
    #     _usr_angles = [avg_angles[idxx] for idxx in _indexes]
    #     _cat = [cat[idxx] for idxx in _indexes]
    #     print(direction)
    #     df = pd.DataFrame({'usr_angle': _usr_angles, 'cat': _cat})
    #     model = ols('usr_angle~C(cat)', data=df).fit()
    #     anova_table = anova_lm(model, typ=2)
    #     print(anova_table)
    #     mc = MultiComparison(_usr_angles, _cat)
    #     print(mc.tukeyhsd())

    # df = pd.DataFrame({
    #     'std_angle': std_angles,
    #     'usr_angle': avg_angles,
    #     'cat': cat
    # })
    # fig, axes = plt.subplots()
    # sns.boxplot(x='std_angle', y='usr_angle', hue='cat', data=df, ax=axes)
    # # sns.boxplot(x='std_angle', y='usr_angle', data=df, ax=axes)
    # plt.show()
    # model = ols('usr_angle~C(std_angle)', data=df).fit()
    # anova_table = anova_lm(model, typ=2)
    # print(anova_table)
    # mc = MultiComparison(avg_angles, std_angles)
    # print(mc.tukeyhsd())

    # # amplitude
    # for direction in EIGHT_DIRECTIONS:
    #     _indexes = np.where(np.array(std_angles) == direction)[0]
    #     _usr_amp = [usr_amp[idxx] for idxx in _indexes]
    #     _cat = [cat[idxx] for idxx in _indexes]
    #     print(direction)
    #     df = pd.DataFrame({'usr_amp': _usr_amp, 'cat': _cat})
    #     model = ols('usr_amp~C(cat)', data=df).fit()
    #     anova_table = anova_lm(model, typ=2)
    #     print(anova_table)
    #     mc = MultiComparison(_usr_amp, _cat)
    #     print(mc.tukeyhsd())

    for i, usr_a in enumerate(usr_amp):
        plt.scatter([usr_a * np.cos(avg_angles[i])],
                    [usr_a * np.sin(avg_angles[i])],
                    c=COLORS[dir_p[i]],
                    alpha=0.3 if cat[i] == 'Single' else 0.7)
    plt.show()

    df = pd.DataFrame({
        'std_angle': std_angles,
        'usr_amp': usr_amp,
        'cat': cat
    })
    fig, axes = plt.subplots()
    sns.boxplot(x='std_angle', y='usr_amp', hue='cat', data=df, ax=axes)
    plt.show()

    # pressure
    # fig, axes = plt.subplots(2, 4)
    # for ix, direction in enumerate(EIGHT_DIRECTIONS):
    #     _indexes = np.where(np.array(std_angles) == direction)[0]
    #     single_max, single_num = 0, 0
    #     multiple_max, multiple_num = 0, 0
    #     for idxx in _indexes:
    #         if cat[idxx] == 'Single':
    #             single_max = max(single_max, len(usr_pres[idxx]))
    #             single_num += 1
    #         else:
    #             multiple_max = max(multiple_max, len(usr_pres[idxx]))
    #             multiple_num += 1
    #     single_arr = np.ma.empty((single_max, single_num))
    #     single_arr.mask = True
    #     multiple_arr = np.ma.empty((multiple_max, multiple_num))
    #     multiple_arr.mask = True
    #     single_i, multiple_i = 0, 0
    #     for idxx in _indexes:
    #         if cat[idxx] == 'Single':
    #             single_arr[:len(usr_pres[idxx]),
    #                        single_i] = np.array(usr_pres[idxx])
    #             single_i += 1
    #         else:
    #             multiple_arr[:len(usr_pres[idxx]),
    #                          multiple_i] = np.array(usr_pres[idxx])
    #             multiple_i += 1
    #     axes[ix // 4][ix % 4].scatter(list(range(single_max)),
    #                                   np.ma.mean(single_arr, axis=1),
    #                                   c='red')
    #     axes[ix // 4][ix % 4].scatter(list(range(multiple_max)),
    #                                   np.ma.mean(multiple_arr, axis=1),
    #                                   c='blue')
    df = pd.DataFrame({
        'max_pressure': max_p,
        'average_pressure': avg_p,
        'direction': dir_p,
        'category': cat
    })
    fig, axes = plt.subplots()
    sns.boxplot(x='direction',
                y='max_pressure',
                hue='category',
                data=df,
                ax=axes)
    plt.show()
    fig, axes = plt.subplots()
    sns.boxplot(x='direction',
                y='average_pressure',
                hue='category',
                data=df,
                ax=axes)
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
                    dis = sqrt((x[end] - x[start]) ** 2 + (y[end] - y[start]) ** 2)
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


def calSingleAcc():
    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    top_1 = 0
    total = 0

    for dir in os.listdir(BASE_DIR):
        print(dir)
        for i, c in enumerate(DIRECTIONS_MAP[args.direction]):
            for j in range(5):
                try:
                    path = np.load(
                        os.path.join(BASE_DIR, dir, '0', str(args.direction),
                                     str(i) + '_' + str(j) + '.npy'))
                except:
                    continue
                x, y, d = getAveragePath(path)
                if (len(x) <= 0):
                    continue
                try:
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    usr_direction = redundant_8directions[np.argmax(weights)]
                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[usr_direction]),
                            np.sin(EIGHT_DIRECTIONS[usr_direction])
                        ])
                    ]
                except Exception as e:
                    print(i, j)
                    print(str(e))
                    continue
                candidates = []
                for std_dir in EIGHT_DIRECTIONS:
                    candidates.append(
                        dtw(path_directions,
                            [np.array([np.cos(std_dir),
                                       np.sin(std_dir)])],
                            dist=angleDist)[0])
                sorted_index = np.argsort(candidates)
                dist_min = candidates[sorted_index[0]]
                dist_min_indexes = []
                for s_i in sorted_index:
                    if candidates[s_i] == dist_min:
                        dist_min_indexes.append(s_i)
                    elif candidates[s_i] > dist_min:
                        break
                total += 1
                if i in dist_min_indexes:
                    top_1 += 1
                else:
                    print(i, dist_min_indexes)
                    # plotOneLettersCorner8(path)

    print(total, top_1, top_1 / total)


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
    def angleDist(ang1, ang2):
        return 1 - (np.dot(ang1, ang2) / np.linalg.norm(ang1) /
                    np.linalg.norm(ang2))

    top_1 = 0
    total = 0
    for dir in os.listdir(BASE_DIR):
        if not 'ch_data_letter_dir_' in dir or not os.path.isdir(
                os.path.join(BASE_DIR, dir)):
            continue
        for i, c in enumerate(LETTER):
            for j in range(5):
                path = np.load(
                    os.path.join(BASE_DIR, dir, c + '_' + str(j) + '.npy'))
                try:
                    directions_index, redundant_8directions, weights = get8Directions(
                        path)
                    path_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[_]),
                            np.sin(EIGHT_DIRECTIONS[_])
                        ]) for _ in redundant_8directions
                    ]
                except:
                    print(c, j)
                    continue
                if (len(weights) <= 0):
                    continue
                candidates = []
                for ch in LETTER:
                    std_directions = [
                        np.array([
                            np.cos(EIGHT_DIRECTIONS[i]),
                            np.sin(EIGHT_DIRECTIONS[i])
                        ]) for i in DIRECTION_PATTERN8[ch]
                    ]
                    d, cost_matrix, acc_cost_matrix, warping_path = dtw(
                        path_directions,
                        std_directions,
                        dist=angleDist,
                        warp=3,
                        s=0.5)
                    adj_dist = 0
                    path_warp = np.array(list(warping_path[0]),
                                         dtype=np.uint16)
                    std_warp = np.array(list(warping_path[1]), dtype=np.uint16)
                    for path_i in range(len(path_warp)):
                        adj_dist += weights[path_warp[path_i]] * angleDist(
                            path_directions[path_warp[path_i]],
                            std_directions[std_warp[path_i]])
                    candidates.append(adj_dist)
                sorted_index = np.argsort(candidates)
                dist_min = candidates[sorted_index[0]]
                dist_min_indexes = []
                for s_i in sorted_index:
                    if candidates[s_i] == dist_min:
                        dist_min_indexes.append(s_i)
                    elif candidates[s_i] > dist_min:
                        break
                total += 1
                if i in dist_min_indexes:
                    top_1 += 1
                # print(path_directions)
                # print(candidates)
                # input()
                if i not in dist_min_indexes:
                    print(LETTER[i], j,
                          [LETTER[dmi] for dmi in dist_min_indexes])
                    # plotOneLettersCorner8(path)
    print(total, top_1, top_1 / total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--person',
                        help='specify the person you want to look into')
    parser.add_argument(
        '-i',
        '--index',
        default='0',
        help='specify the index you want to look into, default as 0')
    parser.add_argument(
        '-d',
        '--direction',
        default='0',
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
    # plotElapsed()
    # plotTendency()
    # plotAmplitude()
    plotPressure()
    # plotDoubleDirectionsCutToSingle()
    # plotMultipleDirectionsCutToSingle()
    # plotMultipleDirectionsCutToSingleIncludedAngles()
    # plotMultipleDirectionsCutToSingleAmplitude()
    # plotMultipleDirectionsCutToSinglePressure()
    # plotMultipleDirectionsCutToSinglePressureExtrema()
    # gaussianDirections()
    # gaussianDirectionsMultiple()
    # visualizeGaussianDirections()
    # calSingleAcc()
    # migrateSingleAndMultiple()
    # pointsNumber()
