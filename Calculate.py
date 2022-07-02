import math
import os
import json
from os import curdir, name
import numpy as np
from dtw import dtw
from numpy.lib.function_base import append

MIN_DISTANCE = 0  # 小于则合并
MIN_CORNER_ANGLE_FIRST = 90
MIN_CORNER_ANGLE_MIDDLE = 60
MIN_CORNER_ANGLE_SECOND = 45
MIN_LINE_LENGTH = 0.2  # just experience
MAX_LINE_LENGTH = 1.2


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


def genPointLabels(points_x, points_y, depths):
    pointLabels = []
    # centerPointGroups = gatherCorner(
    #     calFirstCorner(points_x, points_y, depths),
    #     calSecondCorner(points_x, points_y, depths))
    centerPointGroups = calFirstCorner(points_x, points_y, depths)
    # print(centerPointGroups)
    for pointGroup in centerPointGroups:
        pointLabels.append(pointGroup[int(len(pointGroup) / 2)])
    # print(pointLabels)
    # print(points)
    return pointLabels


def genVectors(points_x, points_y, depths, normalized=True):
    vectors = []
    points = genPoints(points_x, points_y, depths)
    # calculate vector
    # if (len(points) == 1):
    #     print("error only one point")
    for i in range(1, len(points)):
        v = np.array(
            [points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]])
        if (normalized):
            v = v / np.linalg.norm(v)
        vectors.append(v)
    return vectors


def calculatePoints(data):  # len(data) > 0
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
    # Move relatively to center
    center_x = points_x[0]
    center_y = points_y[0]
    for i in range(len(points_x)):
        points_x[i] -= center_x
    for i in range(len(points_y)):
        points_y[i] -= center_y

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
                    np.array([points_x[i + 1], points_y[i +
                                                        1]])) < MIN_DISTANCE:
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
                i += 1
    return points_x, points_y, depths


def calFirstCorner(points_x, points_y, depths):
    points = [[points_x[i], points_y[i]] for i in range(len(points_x))]
    centerPointGroups = []
    next_point = 0
    if (len(points) > 3):
        i = 1
        min_pressure_rate = max((depths[0] + depths[1] + depths[2]) / 3,
                                (depths[-1] + depths[-2] + depths[-3]) / 3)
        while (i < len(points_x) - 1):
            if (next_point >= len(points_x) - 1):
                break
            if (i < next_point):
                i = next_point
            centerPointSingle = []
            vector_1 = np.array(points[i]) - np.array(points[i - 1])
            vector_2 = np.array(points[i + 1]) - np.array(points[i])
            angle = calAngle(vector_1, vector_2)
            if angle >= MIN_CORNER_ANGLE_FIRST or i < 2 or i > len(
                    points_x) - 3:
                # search before
                j = i
                while (j >= 0):
                    if (np.linalg.norm(
                            np.array(points[i]) - np.array(points[j])) <
                            MIN_LINE_LENGTH):
                        centerPointSingle.append(j)
                        j = j - 1
                    else:
                        break
                centerPointSingle.reverse()
                # search after
                j = i + 1
                while (j < len(points_x)):
                    if (np.linalg.norm(
                            np.array(points[i]) - np.array(points[j])) <
                            MIN_LINE_LENGTH):
                        centerPointSingle.append(j)
                        j = j + 1
                    else:
                        next_point = j
                        break
                # if (len(centerPointSingle) > 1):
                centerPointGroups.append(centerPointSingle)
            i = i + 1
        try:
            # clean single element group
            cleanedCenterPointGroups = []
            allCenterPointGroups = []
            for i in centerPointGroups:
                for j in i:
                    allCenterPointGroups.append(j)
            allCenterPointGroups = list(set(allCenterPointGroups))
            allCenterPointGroups.sort()
            curr_num = allCenterPointGroups[0] - 1
            cleanedCenterPointSingle = []
            for i in allCenterPointGroups:
                if i == curr_num + 1:
                    cleanedCenterPointSingle.append(i)
                else:
                    cleanedCenterPointGroups.append(cleanedCenterPointSingle)
                    cleanedCenterPointSingle = [i]
                curr_num = i
            cleanedCenterPointGroups.append(cleanedCenterPointSingle)
            return cleanedCenterPointGroups
        except:
            return []
    else:
        print("too short data")
        return []


def calSecondCorner(points_x, points_y, depths):
    points = [[points_x[i], points_y[i]] for i in range(len(points_x))]
    centerPointGroups = []
    next_point = 0
    if (len(points) > 5):
        min_pressure_rate = max((depths[0] + depths[1] + depths[2]) / 3,
                                (depths[-1] + depths[-2] + depths[-3]) / 3)
        i = 2
        while (i < len(points_x) - 2):
            if (next_point >= len(points_x) - 2):
                break
            if (i < next_point):
                i = next_point
            centerPointSingle = []
            vector_1 = np.array(points[i - 1]) - np.array(points[i - 2])
            vector_2 = np.array(points[i + 2]) - np.array(points[i + 1])
            vector_3 = np.array(points[i]) - np.array(points[i - 1])
            vector_4 = np.array(points[i + 1]) - np.array(points[i])
            angle = calAngle(vector_1, vector_2)
            angle_1 = calAngle(vector_1, vector_4)
            angle_2 = calAngle(vector_3, vector_2)
            if ((angle >= MIN_CORNER_ANGLE_SECOND
                 or angle_1 >= MIN_CORNER_ANGLE_MIDDLE
                 or angle_2 >= MIN_CORNER_ANGLE_MIDDLE) and
                (i / len(points_x) > 0.1 or depths[i] > min_pressure_rate)):
                # print(i)
                # print(calRadius(np.array(points[i - 1]), np.array(points[i]), np.array(points[i + 1])))
                # if calRadius(np.array(points[i - 1]), np.array(points[i]), np.array(points[i + 1])) > 12:
                #     i = i + 1
                #     continue
                if np.linalg.norm(
                        np.array(points[i]) -
                        np.array(points[i - 1])) > MAX_LINE_LENGTH:
                    if np.linalg.norm(
                            np.array(points[i]) -
                            np.array(points[i + 1])) > MAX_LINE_LENGTH:
                        i = i + 1
                        continue
                # search before
                j = i
                while (j >= 0):
                    if np.linalg.norm(
                            np.array(points[i]) -
                            np.array(points[j])) < MIN_LINE_LENGTH and (
                                i / len(points_x) > 0.1
                                or depths[j] > min_pressure_rate):
                        centerPointSingle.append(j)
                        j = j - 1
                    else:
                        break
                centerPointSingle.reverse()
                # search after
                j = i + 1
                while (j < len(points_x)):
                    if np.linalg.norm(
                            np.array(points[i]) -
                            np.array(points[j])) < MIN_LINE_LENGTH and (
                                i / len(points_x) > 0.1
                                or depths[j] > min_pressure_rate):
                        centerPointSingle.append(j)
                        j = j + 1
                    else:
                        next_point = j
                        break
                # if (len(centerPointSingle) > 1):
                centerPointGroups.append(centerPointSingle)
            i = i + 1
        # clean single element group
        try:
            cleanedCenterPointGroups = []
            allCenterPointGroups = []
            for i in centerPointGroups:
                for j in i:
                    allCenterPointGroups.append(j)
            allCenterPointGroups = list(set(allCenterPointGroups))
            allCenterPointGroups.sort()
            curr_num = allCenterPointGroups[0] - 1
            cleanedCenterPointSingle = []
            for i in allCenterPointGroups:
                if i == curr_num + 1:
                    cleanedCenterPointSingle.append(i)
                else:
                    cleanedCenterPointGroups.append(cleanedCenterPointSingle)
                    cleanedCenterPointSingle = [i]
                curr_num = i
            cleanedCenterPointGroups.append(cleanedCenterPointSingle)
            return cleanedCenterPointGroups
        except:
            return []
    else:
        print("too short data")
        return []


def gatherCorner(centerPointGroupsFirst, centerPointGroupsSecond):
    cleanedCenterPointGroups = []
    allCenterPointGroups = []
    for i in centerPointGroupsFirst:
        for j in i:
            allCenterPointGroups.append(j)
    for i in centerPointGroupsSecond:
        for j in i:
            allCenterPointGroups.append(j)
    allCenterPointGroups = list(set(allCenterPointGroups))
    allCenterPointGroups.sort()
    try:
        curr_num = allCenterPointGroups[0] - 1
        cleanedCenterPointSingle = []
        for i in allCenterPointGroups:
            if i == curr_num + 1:
                cleanedCenterPointSingle.append(i)
            else:
                cleanedCenterPointGroups.append(cleanedCenterPointSingle)
                cleanedCenterPointSingle = [i]
            curr_num = i
        cleanedCenterPointGroups.append(cleanedCenterPointSingle)
        # break long edge and remove far points
        # for i in range(len(cleanedCenterPointGroups)):
        #     for j in range(len(cleanedCenterPointGroups[i]) - 1):
        #         point_label = cleanedCenterPointGroups[i][j]
        #         point_1 = np.array(points_x[point_label], points_y[point_label])
        #         point_2 = np.array(points_x[point_label + 1], points_y[point_label + 1])
        #         # if (np.linalg.norm(point_1 - point_2) > )
        #         print(np.linalg.norm(point_1 - point_2))
        return cleanedCenterPointGroups
    except:
        return []


def calAngle(vector_1, vector_2):  # vector should be np array return degree
    try:
        vector_1 = vector_1 / np.linalg.norm(vector_1)
        vector_2 = vector_2 / np.linalg.norm(vector_2)
        if (np.array_equal(vector_1, vector_2)):
            return 0.0
        elif (np.array_equal(vector_1, -1 * vector_2)):
            return 180.0
        else:
            cosangle = vector_1.dot(vector_2) / (np.linalg.norm(vector_1) *
                                                 np.linalg.norm(vector_2))
            angle = np.arccos(cosangle)
            return math.degrees(angle)
    except:
        print(vector_1)
        print(vector_2)


def calRadius(p1, p2, p3):
    x21 = p2[0] - p1[0]
    y21 = p2[1] - p1[1]
    x32 = p3[0] - p2[0]
    y32 = p3[1] - p2[1]
    if (x21 * y32 - x32 * y21 == 0):
        return None
    xy21 = p2[0] * p2[0] - p1[0] * p1[0] + p2[1] * p2[1] - p1[1] * p1[1]
    xy32 = p3[0] * p3[0] - p2[0] * p2[0] + p3[1] * p3[1] - p2[1] * p2[1]
    y0 = (x32 * xy21 - x21 * xy32) / 2 * (y21 * x32 - y32 * x21)
    x0 = (xy21 - 2 * y0 * y21) / (2.0 * x21)
    R = ((p1[0] - x0)**2 + (p1[1] - y0)**2)**0.5
    return R