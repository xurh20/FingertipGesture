import numpy as np
import argparse
import os
import imageio
import math
import random
import matplotlib.pyplot as plt
from itertools import chain

from numpy.core.numeric import allclose

PERSON = "cwh"
BASE_DIR = "../data_gt/" + PERSON + "/"
SAVE_PICTURE_DIR = "../pictures_red/"
candidates = [chr(y) for y in range(97, 123)]
LEFT_BOUND = 79
UP_BOUND = 34
RIGHT_BOUND = 106
DOWN_BOUND = 68
WIDTH = RIGHT_BOUND - LEFT_BOUND + 1
HEIGHT = DOWN_BOUND - UP_BOUND + 1
MIN_CORNER_ANGLE = 60
MIN_LINE_LENGTH = 0.5 # just experience

SENTENCE = 2
WORD = 0

def loadData():
    data = np.load(BASE_DIR + candidates[SENTENCE] + "_" + str(WORD) + ".npy")
    # print(data)
    return data

def calculatePoints(data):
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
    return points_x, points_y, depths

def showPicture(points_x, points_y, depths):
    max_depth = max(depths)
    colors = list(map(lambda x : '#' + str(hex(255 - int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, depths))
    centerPointGroups = calCorner(points_x, points_y, depths)
    for group in range(len(centerPointGroups)):
        for point in centerPointGroups[group]:
            colors[point] = "#FF0000"
            plt.text(points_x[point], points_y[point], group)
    # plt.axis('equal')
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(points_x, points_y, c=colors)
    # print(colors)
    x_bound = (min(points_x) - 1, max(points_x) + 1)
    y_bound = (min(points_y) - 1, max(points_y) + 1)
    plt.xlim((-7, 6))
    plt.ylim((-4, 8))
    plt.plot(points_x, points_y, linestyle='--')
    plt.show()

def saveAllPicures():
    for i in range(20):
        for j in range(8):
            SENTENCE = i
            WORD = j
            if os.path.exists(BASE_DIR + candidates[SENTENCE] + "_" + str(WORD) + ".npy"):
                data = np.load(BASE_DIR + candidates[SENTENCE] + "_" + str(WORD) + ".npy")
                points_x, points_y, depths = calculatePoints(data)
                if (len(points_x) == 0):
                    continue
                centerPointGroups = calCorner(points_x, points_y, depths)
                points_x = list(map(lambda x : x - WIDTH / 2, points_x))
                points_y = list(map(lambda x : x + HEIGHT / 2, points_y))
                max_depth = max(depths)
                colors = list(map(lambda x : '#' + str(hex(255 - int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, depths))
                for group in range(len(centerPointGroups)):
                    for point in centerPointGroups[group]:
                        colors[point] = "#FF0000"
                        plt.text(points_x[point], points_y[point], group)
                plt.axis('scaled')
                plt.scatter(points_x, points_y, c=colors)
                x_bound = (min(points_x) - 1, max(points_x) + 1)
                y_bound = (min(points_y) - 1, max(points_y) + 1)
                plt.xlim(x_bound)
                plt.ylim(y_bound)
                plt.plot(points_x, points_y, linestyle='--')
                if (not os.path.exists(SAVE_PICTURE_DIR + PERSON)):
                    os.makedirs(SAVE_PICTURE_DIR + PERSON)
                plt.savefig(os.path.join(SAVE_PICTURE_DIR + PERSON, "{}_{}.png".format(i, j)))
                plt.close()

def saveVideo(points_x, points_y, depths):
    max_depth = max(depths)
    colors = list(map(lambda x : '#' + str(hex(255 - int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, depths))
    x_bound = (min(points_x) - 1, max(points_x) + 1)
    y_bound = (min(points_y) - 1, max(points_y) + 1)
    for i in range(len(points_x)):
        plt.axis('scaled')
        plt.scatter(points_x[:i+1], points_y[:i+1], c=colors[:i+1])
        plt.plot(points_x[:i+1], points_y[:i+1], linestyle='--', color="darkcyan")
        plt.xlim(x_bound)
        plt.ylim(y_bound)
        plt.savefig(os.path.join(SAVE_PICTURE_DIR, "{}.png".format(i)))

    images = []
    for i in range(len(points_x)):
        images.append(imageio.imread(os.path.join(SAVE_PICTURE_DIR, "{}.png".format(i))))
    imageio.mimsave('{}.mp4'.format("plot_" + str(SENTENCE) + "_" + str(WORD)), images)

def calAngle(vector_1, vector_2): # vector should be np array return degree
    cosangle = vector_1.dot(vector_2)/(np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    angle = np.arccos(cosangle)
    # print(math.degrees(angle))
    return math.degrees(angle)

def calCorner(points_x, points_y, depths):
    points = [[points_x[i], points_y[i]] for i in range(len(points_x))]
    centerPointGroups = []
    next_point = 0
    if (len(points) > 5):
        # print(depths)
        for i in range(2, len(points_x) - 2):
            # if (depths[i - 2] < 5000 or depths[i - 1] < 5000 or depths[i] < 5000 or depths[i + 1] < 5000 or depths[i + 2] < 5000):
            #     continue
            # print(i)
            # print(points[i])
            if (next_point >= len(points_x) - 2):
                break
            if (i < next_point):
                i = next_point
            centerPointSingle = []
            vector_1 = np.array(points[i - 1]) - np.array(points[i - 2])
            vector_2 = np.array(points[i + 2]) - np.array(points[i + 1])
            angle = calAngle(vector_1, vector_2)
            if (angle >= MIN_CORNER_ANGLE and depths[i] > 3000):
                # search before
                j = i
                while(j >= 0):
                    if np.linalg.norm(np.array(points[i]) - np.array(points[j])) < MIN_LINE_LENGTH and depths[j] > 3000:
                        centerPointSingle.append(j)
                        j = j - 1
                    else:
                        break
                centerPointSingle.reverse()
                # search after
                j = i + 1
                while(j < len(points_x)):
                    if np.linalg.norm(np.array(points[i]) - np.array(points[j])) < MIN_LINE_LENGTH and depths[j] > 3000:
                        centerPointSingle.append(j)
                        j = j + 1
                    else:
                        next_point = j
                        break
                # if (len(centerPointSingle) > 1):
                centerPointGroups.append(centerPointSingle)
        # clean single element group
        print(centerPointGroups)
        cleanedCenterPointGroups = []
        allCenterPointGroups = []
        for i in centerPointGroups:
            for j in i:
                allCenterPointGroups.append(j)
        allCenterPointGroups = list(set(allCenterPointGroups))
        allCenterPointGroups.sort()
        # allCenterPointGroups = list(set(chain.from_iterable(centerPointGroups)))
        print(allCenterPointGroups)
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
        print(cleanedCenterPointGroups)
        # break long edge and remove far points
        newCleanedPG = []
        for i in cleanedCenterPointGroups:
            cleanedCenterPointSingle = [i[0]]
            for j in range(1, len(i)):
                k = i[j]
                # TODO change to relative
                if np.linalg.norm(np.array(points[k - 1]) - np.array(points[k])) >= 2 * MIN_LINE_LENGTH:
                    # break
                    newCleanedPG.append(cleanedCenterPointSingle)
                    cleanedCenterPointSingle = [k]
                else:
                    cleanedCenterPointSingle.append(k)
                # if np.linalg.norm(np.array(points[i[int(len(i) / 2)]]) - np.array(points[j])) < MIN_LINE_LENGTH:
                #     cleanedCenterPointSingle.append(j)
            newCleanedPG.append(cleanedCenterPointSingle)
        print(newCleanedPG)
        return newCleanedPG
    else:
        print("too short data")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        "--show",
                        action="store_true",
                        help="show picture")
    parser.add_argument("-a",
                        "--all",
                        action="store_true",
                        help="all pictures")
    parser.add_argument("-v",
                        "--video",
                        action="store_true",
                        help="save video")
    args = parser.parse_args()

    data = loadData()
    points_x, points_y, depths = calculatePoints(data)
    points_x = list(map(lambda x : x - WIDTH / 2, points_x))
    points_y = list(map(lambda x : x + HEIGHT / 2, points_y))
    
    if args.show:
        showPicture(points_x, points_y, depths)
    elif args.all:
        saveAllPicures()
    elif args.video:
        saveVideo(points_x, points_y, depths)
    else:
        centerPointGroups = calCorner(points_x, points_y, depths)
        print(centerPointGroups)