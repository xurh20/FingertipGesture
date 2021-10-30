import numpy as np
import argparse
import os
import imageio
import math
import json
import random
import matplotlib.pyplot as plt
from itertools import chain

from numpy.core.numeric import allclose
from calBreak import loadRawNum, checkBreak

PERSON = "tty"
BASE_DIR = "../data/alphabeta_data_"
SAVE_DIR = "../data/saved_data_" + PERSON + "/"
SAVE_CB_DIR = "../data/break_num/"
VIDEO_DIR = "../data/videos/"
SAVE_PICTURE_DIR = "../pictures/"
candidates = [chr(y) for y in range(97, 123)]
LEFT_BOUND = 79
UP_BOUND = 34
RIGHT_BOUND = 106
DOWN_BOUND = 68
WIDTH = RIGHT_BOUND - LEFT_BOUND + 1
HEIGHT = DOWN_BOUND - UP_BOUND + 1
MIN_CORNER_ANGLE_FIRST = 90
MIN_CORNER_ANGLE_SECOND = 60
MIN_LINE_LENGTH = 0.5 # just experience
MAX_LINE_LENGTH = 0.5
MIN_ANGLE = 20 # 低于且两端距离过长就算行进间
MIN_PRESSURE_RATE = 0.4

MIN_DISTANCE = 0.3 # 小于则合并

SENTENCE = 65
WORD = 3

def loadData(sentence, word, person):
    data = np.load(BASE_DIR + person + "/" + str(sentence) + "_" + str(word) + ".npy")
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
    # Normalize depths
    if (len(depths) > 0):
        max_depth = max(depths)
        depths = [i / max_depth for i in depths]
    return points_x, points_y, depths

def showPicture(points_x, points_y, depths):
    max_depth = max(depths)
    colors = list(map(lambda x : '#' + str(hex(255 - int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, depths))

    centerPointGroups = gatherCorner(calFirstCorner(points_x, points_y, depths), calSecondCorner(points_x, points_y, depths))
    for group in range(len(centerPointGroups)):
        for point in centerPointGroups[group]:
            colors[point] = "#FF0000"
            plt.text(points_x[point], points_y[point], group)

    plt.axis('equal')
    plt.scatter(points_x, points_y, c=colors)
    x_bound = (min(points_x) - 1, max(points_x) + 1)
    y_bound = (min(points_y) - 1, max(points_y) + 1)
    plt.plot(points_x, points_y, linestyle='--')
    plt.show()

def showDistance(points_x, points_y):
    prePoint = [points_x[0], points_y[0]]
    for i in range(1, len(points_x)):
        nowPoint = [points_x[i], points_y[i]]
        print(np.linalg.norm(np.array(prePoint) - np.array(nowPoint)))
        prePoint = nowPoint

def plotAllLocus(points_x, points_y, depths, person):
    max_depth = max(depths)
    colors = list(map(lambda x : '#' + str(hex(255 - int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, depths))

    plt.axis('equal')
    plt.scatter(points_x, points_y, c=colors)
    x_bound = (min(points_x) - 1, max(points_x) + 1)
    y_bound = (min(points_y) - 1, max(points_y) + 1)
    plt.plot(points_x, points_y, linestyle='--')
    if (not os.path.exists(SAVE_PICTURE_DIR + person)):
        os.makedirs(SAVE_PICTURE_DIR + person)
    plt.savefig(os.path.join(SAVE_PICTURE_DIR + person, "all.png"))
    print("done")
    plt.cla()
    plt.close()
    # plt.show()

def saveAllPicures():
    for i in range(82):
        for j in range(20):
            SENTENCE = i
            WORD = j
            if os.path.exists(BASE_DIR + PERSON + "/" + str(SENTENCE) + "_" + str(WORD) + ".npy"):
                data = np.load(BASE_DIR + PERSON + "/" + str(SENTENCE) + "_" + str(WORD) + ".npy")
                points_x, points_y, depths = calculatePoints(data)
                if (len(points_x) == 0):
                    continue
                # centerPointGroups = calFirstCorner(points_x, points_y, depths)
                points_x = list(map(lambda x : x - WIDTH / 2, points_x))
                points_y = list(map(lambda x : x + HEIGHT / 2, points_y))
                max_depth = max(depths)
                colors = list(map(lambda x : '#' + str(hex(255 - int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, depths))
                # for group in range(len(centerPointGroups)):
                #     for point in centerPointGroups[group]:
                #         colors[point] = "#FF0000"
                #         plt.text(points_x[point], points_y[point], group)
                plt.axis('scaled')
                plt.scatter(points_x, points_y, c=colors)
                x_bound = (min(points_x) - 1, max(points_x) + 1)
                y_bound = (min(points_y) - 1, max(points_y) + 1)
                # plt.xlim(x_bound)
                # plt.ylim(y_bound)
                plt.xlim((-7, 8))
                plt.ylim((-4, 10))
                plt.plot(points_x, points_y, linestyle='--')
                if (not os.path.exists(SAVE_PICTURE_DIR + PERSON)):
                    os.makedirs(SAVE_PICTURE_DIR + PERSON)
                plt.savefig(os.path.join(SAVE_PICTURE_DIR + PERSON, "{}_{}.png".format(i, j)))
                plt.close()

def savePremativeData():
    if (not os.path.exists(SAVE_DIR)):
        os.makedirs(SAVE_DIR)
    for i in range(82):
        for j in range(20):
            SENTENCE = i
            WORD = j
            if os.path.exists(BASE_DIR + PERSON + "/" + str(SENTENCE) + "_" + str(WORD) + ".npy"):
                data = np.load(BASE_DIR + PERSON + "/" + str(SENTENCE) + "_" + str(WORD) + ".npy")
                points_x, points_y, depths = calculatePoints(data)
                np.save(SAVE_DIR + str(SENTENCE) + "_" + str(WORD) + "_x" + ".npy", points_x)
                np.save(SAVE_DIR + str(SENTENCE) + "_" + str(WORD) + "_y" + ".npy", points_y)
                np.save(SAVE_DIR + str(SENTENCE) + "_" + str(WORD) + "_d" + ".npy", depths)

def saveVideo(points_x, points_y, depths, PERSON):
    max_depth = max(depths)
    colors = list(map(lambda x : '#' + str(hex(255 - int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, depths))
    x_bound = (min(points_x) - 1, max(points_x) + 1)
    y_bound = (min(points_y) - 1, max(points_y) + 1)
    for i in range(len(points_x)):
        plt.axis('scaled')
        plt.scatter(points_x[:i+1], points_y[:i+1], c=colors[:i+1])
        plt.plot(points_x[:i+1], points_y[:i+1], linestyle='--', color="darkcyan")
        # plt.xlim(x_bound)
        # plt.ylim(y_bound)
        plt.xlim((-7, 8))
        plt.ylim((-4, 10))
        plt.savefig(os.path.join(SAVE_PICTURE_DIR, "{}.png".format(i)))

    images = []
    for i in range(len(points_x)):
        images.append(imageio.imread(os.path.join(SAVE_PICTURE_DIR, "{}.png".format(i))))
    imageio.mimsave('{}.mp4'.format(VIDEO_DIR + PERSON + "_plot_" + str(SENTENCE) + "_" + str(WORD)), images)

def calAngle(vector_1, vector_2): # vector should be np array return degree
    try:
        cosangle = vector_1.dot(vector_2)/(np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        angle = np.arccos(cosangle)
    except:
        print(vector_1)
        print(vector_2)
    # print(math.degrees(angle))
    return math.degrees(angle)

def calFirstCorner(points_x, points_y, depths):
    points = [[points_x[i], points_y[i]] for i in range(len(points_x))]
    centerPointGroups = []
    next_point = 0
    if (len(points) > 3):
        i = 1
        min_pressure_rate = max((depths[0] + depths[1] + depths[2]) / 3, (depths[-1] + depths[-2] + depths[-3]) / 3)
        while (i < len(points_x) - 1):
            if (next_point >= len(points_x) - 1):
                break
            if (i < next_point):
                i = next_point
            centerPointSingle = []
            vector_1 = np.array(points[i]) - np.array(points[i - 1])
            vector_2 = np.array(points[i + 1]) - np.array(points[i])
            angle = calAngle(vector_1, vector_2)
            if (angle >= MIN_CORNER_ANGLE_FIRST and (i / len(points_x) > 0.1 or depths[i] > min_pressure_rate)):
                # search before
                j = i
                while(j >= 0):
                    if np.linalg.norm(np.array(points[i]) - np.array(points[j])) < (j / len(points_x) > 0.1 or MIN_LINE_LENGTH and depths[j] > min_pressure_rate):
                        centerPointSingle.append(j)
                        j = j - 1
                    else:
                        break
                centerPointSingle.reverse()
                # search after
                j = i + 1
                while(j < len(points_x)):
                    if np.linalg.norm(np.array(points[i]) - np.array(points[j])) < (i / len(points_x) > 0.1 or MIN_LINE_LENGTH and depths[j] > min_pressure_rate):
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
            # break long edge and remove far points
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
        min_pressure_rate = max((depths[0] + depths[1] + depths[2]) / 3, (depths[-1] + depths[-2] + depths[-3]) / 3)
        i = 2
        while (i < len(points_x) - 2):
            if (next_point >= len(points_x) - 2):
                break
            if (i < next_point):
                i = next_point
            centerPointSingle = []
            vector_1 = np.array(points[i - 1]) - np.array(points[i - 2])
            vector_2 = np.array(points[i + 2]) - np.array(points[i + 1])
            angle = calAngle(vector_1, vector_2)
            if (angle >= MIN_CORNER_ANGLE_SECOND and (i / len(points_x) > 0.1 or depths[i] > min_pressure_rate)):
                if np.linalg.norm(np.array(points[i]) - np.array(points[i - 1])) > MAX_LINE_LENGTH:
                    if np.linalg.norm(np.array(points[i]) - np.array(points[i + 1])) > MAX_LINE_LENGTH:
                        if calAngle(np.array(points[i]) - np.array(points[i - 1]), np.array(points[i + 1]) - np.array(points[i])) < MIN_ANGLE:
                            i = i + 1
                            continue
                # search before
                j = i
                while(j >= 0):
                    if np.linalg.norm(np.array(points[i]) - np.array(points[j])) < MIN_LINE_LENGTH and (i / len(points_x) > 0.1 or depths[j] > min_pressure_rate):
                        centerPointSingle.append(j)
                        j = j - 1
                    else:
                        break
                centerPointSingle.reverse()
                # search after
                j = i + 1
                while(j < len(points_x)):
                    if np.linalg.norm(np.array(points[i]) - np.array(points[j])) < MIN_LINE_LENGTH and (i / len(points_x) > 0.1 or depths[j] > min_pressure_rate):
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
        return cleanedCenterPointGroups
    except:
        return []


def calBreakNum(points_x, points_y, depths):
    centerPointGroups = gatherCorner(calFirstCorner(points_x, points_y, depths), calSecondCorner(points_x, points_y, depths))
    return len(centerPointGroups)

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
    parser.add_argument("-d",
                        "--savedata",
                        action="store_true",
                        help="save data")
    parser.add_argument("-pa",
                        "--plotall",
                        action="store_true",
                        help="plot all line")
    parser.add_argument("-cb",
                        "--calbreak",
                        action="store_true",
                        help="calculate break points")
    parser.add_argument("-cn",
                        "--checkbreak",
                        action="store_true",
                        help="check break points")
    parser.add_argument("-sd",
                        "--showdist",
                        action="store_true",
                        help="show distance")
    args = parser.parse_args()

    
    if args.show:
        data = loadData(SENTENCE, WORD, PERSON)
        points_x, points_y, depths = calculatePoints(data)
        points_x = list(map(lambda x : x - WIDTH / 2, points_x))
        points_y = list(map(lambda x : x + HEIGHT / 2, points_y))
        showPicture(points_x, points_y, depths)

    elif args.all:
        saveAllPicures()

    elif args.video:
        for PERSON in ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]:
            for i in range(1, 82):
                for j in range(20):
                    SENTENCE = i
                    WORD = j
                    BASE_DIR = "../data/alphabeta_data_"
                    if os.path.exists(BASE_DIR + PERSON + "/" + str(SENTENCE) + "_" + str(WORD) + ".npy"):
                        data = np.load(BASE_DIR + PERSON + "/" + str(SENTENCE) + "_" + str(WORD) + ".npy")
                        points_x, points_y, depths = calculatePoints(data)
                        points_x = list(map(lambda x : x - WIDTH / 2, points_x))
                        points_y = list(map(lambda x : x + HEIGHT / 2, points_y))
                        saveVideo(points_x, points_y, depths, PERSON)
                        print("done" + "_" + str(i) + "_" + str(j))

    elif args.savedata:
        savePremativeData()

    elif args.plotall:
        for person in ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]:
            allpoints_x = []
            allpoints_y = []
            alldepths = []
            for i in range(1, 82):
                for j in range(20):
                    if os.path.exists(BASE_DIR + person + "/" + str(i) + "_" + str(j) + ".npy"):
                        points_x, points_y, depths = calculatePoints(loadData(i, j, person))
                        try:
                            start_x = points_x[0]
                            start_y = points_y[0]
                        except:
                            continue
                        points_x = list(map(lambda x : x - start_x, points_x))
                        points_y = list(map(lambda x : x - start_y, points_y))
                        for t in points_x:
                            allpoints_x.append(t)
                        for t in points_y:
                            allpoints_y.append(t)
                        for t in depths:
                            alldepths.append(t)
            plotAllLocus(allpoints_x, allpoints_y, alldepths, person)


    elif args.calbreak:
        for person in ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]:
            allBreakNums = []
            for i in range(1, 82):
                sentenceBreakNums = []
                for j in range(20):
                    if os.path.exists(BASE_DIR + person + "/" + str(i) + "_" + str(j) + ".npy"):
                        points_x, points_y, depths = calculatePoints(loadData(i, j, person))
                        points_x = list(map(lambda x : x - WIDTH / 2, points_x))
                        points_y = list(map(lambda x : x + HEIGHT / 2, points_y))
                        breakNum = calBreakNum(points_x, points_y, depths)
                        sentenceBreakNums.append(breakNum)
                allBreakNums.append(sentenceBreakNums)
            with open(SAVE_CB_DIR + person + ".txt", "w") as f:
                f.write(json.dumps(allBreakNums))

    elif args.checkbreak:
        for person in ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]:
            print(checkBreak(person))
    
    elif args.showdist:
        data = loadData(SENTENCE, WORD, PERSON)
        points_x, points_y, depths = calculatePoints(data)
        points_x = list(map(lambda x : x - WIDTH / 2, points_x))
        points_y = list(map(lambda x : x + HEIGHT / 2, points_y))
        showDistance(points_x, points_y)
    
    else:
        # centerPointGroups = calFirstCorner(points_x, points_y, depths)
        # print(centerPointGroups)
        print("null inst")