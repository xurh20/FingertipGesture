import numpy as np
import argparse
import os
import imageio
import math
import json
import random
import matplotlib.pyplot as plt
from plot import gatherCorner, calFirstCorner, calSecondCorner, loadData, calculatePoints
from cleanWords import cleanWords, lowerCase
from match import genPoints
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import norm

PERSON = ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]
LETTER = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
SAVE_MP_DIR = "../data/match_points/"  # save match points
SAVE_LP_DIR = "../data/letter_points/"  # save points sort by letter

allPattern = cleanWords()


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x,
                                                        scale_y).translate(
                                                            mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def loadKeyPointLabel(person, sentence, word):
    with open(
            SAVE_MP_DIR + person + "_" + str(sentence) + "_" + str(word) +
            ".txt", "r") as f:
        data = json.loads(f.read())
        return data


def genKeyPoint(person, sentence, word):
    # generate points which had been selected where letters exist, one for each letter
    # if there are one point for two letters, throw exception, return empty list
    data = loadData(sentence, word, person)
    if (len(data) == 0):
        return []
    points_x, points_y, depths = calculatePoints(data)
    points = genPoints(points_x, points_y, depths)
    key_point_label = loadKeyPointLabel(person, sentence, word)
    try:
        if (len(key_point_label) > 0
                and len(key_point_label) == len(set(key_point_label))):
            return [points[i] for i in key_point_label]
        else:
            return []
    except:
        print(key_point_label, points)


def saveLetterPoints():
    skip = 0
    total = 0
    all_chosen_points_x = []
    all_chosen_points_y = []
    all_chosen_points_d = []
    for person in PERSON:
        # person = "qlp"
        for letter in LETTER:
            for i in range(1, 82):
                for j in range(len(allPattern[i - 1])):
                    key_point = genKeyPoint(person, i, j)
                    key_point_letter = allPattern[i - 1][j]
                    total += 1
                    if (key_point is None or len(key_point) == 0):
                        skip += 1
                        continue
                    else:
                        for letter_id in range(len(key_point_letter)):
                            if (key_point_letter[letter_id] == letter
                                    or key_point_letter[letter_id]
                                    == lowerCase(letter)):
                                # print(key_point, i)
                                all_chosen_points_x.append(
                                    key_point[letter_id][0])
                                all_chosen_points_y.append(
                                    key_point[letter_id][1])
                                all_chosen_points_d.append(
                                    key_point[letter_id][2])
                print(i, "done")  # sentence done
            print(person, "          done           ")  # person done
            with open(SAVE_LP_DIR + person + "_" + letter + ".txt", "w") as f:
                f.write(
                    json.dumps([
                        all_chosen_points_x, all_chosen_points_y,
                        all_chosen_points_d
                    ]))

            # clean for next person
            all_chosen_points_x = []
            all_chosen_points_y = []
            all_chosen_points_d = []

    return skip, total


def drawPointCloudLetter(person, letters):  # letter should be list of capitalized letters
    for letter in letters:
        with open(SAVE_LP_DIR + person + "_" + letter + ".txt", "r") as f:
            data = json.loads(f.read())
            points_x = data[0]
            points_y = data[1]
            points_d = data[2]
        max_depth = max(points_d)
        colors = list(
            map(
                lambda x: '#' + str(hex(255 - int(x / max_depth * 255)))[2:].
                rjust(2, '0') * 3, points_d))
        for point in range(len(points_x)):
            plt.text(points_x[point], points_y[point], letter)
        plt.axis('equal')
        plt.scatter(points_x, points_y, c=colors)
    plt.show()


def drawSinglePointClouds():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("scaled")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    letterPositions = []
    for letter in LETTER:
        lp = []
        for person in PERSON:
            with open(SAVE_LP_DIR + person + "_" + letter + ".txt", "r") as f:
                data = json.loads(f.read())
                lp.append((np.average(data[0]), np.average(data[1])))
        letterPositions.append(lp)
    averaged = [np.average(lp, axis=0) for lp in letterPositions]

    for i, lp in enumerate(letterPositions):
        confidence_ellipse(np.array([l[0] for l in lp]),
                           np.array([l[1] for l in lp]),
                           ax,
                           n_std=3,
                           alpha=0.5,
                           facecolor='pink',
                           edgecolor='purple',
                           zorder=0)
    for i, letter in enumerate(LETTER):
        plt.text(averaged[i][0], averaged[i][1], letter)
    plt.scatter([a[0] for a in averaged], [a[1] for a in averaged])
    plt.show()


def drawSinglePointRec():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("scaled")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    letterPositions = []
    recPoints = [] # [[(,), (,)]]
    for letter in LETTER:
        lp = []
        min_x, min_y = 1000000, 1000000
        max_x, max_y = -1000000, -1000000
        for person in PERSON:
            with open(SAVE_LP_DIR + person + "_" + letter + ".txt", "r") as f:
                data = json.loads(f.read())
                lp.append((np.average(data[0]), np.average(data[1])))
                min_x = min(min(data[0]), min_x)
                min_y = min(min(data[1]), min_y)
                max_x = max(max(data[0]), max_x)
                max_y = max(max(data[1]), max_y)
        recPoints.append([(min_x, min_y), (max_x, max_y)])
        letterPositions.append(lp)
    averaged = [np.average(lp, axis=0) for lp in letterPositions]

    for points in recPoints:
        rect=plt.Rectangle(
            points[0],  # (x,y)矩形左下角
            points[1][0] - points[0][0],  # width长
            points[1][1] - points[0][1],  # height宽
            color='maroon', 
            alpha=0.05)
        ax.add_patch(rect)
    for i, letter in enumerate(LETTER):
        plt.text(averaged[i][0], averaged[i][1], letter)
    plt.scatter([a[0] for a in averaged], [a[1] for a in averaged])
    plt.show()

def drawGaussion(gaussion_list): # gaussion_list is a list of 1-d points
    x = np.array(gaussion_list)
    n, bins, patches = plt.hist(x, 20, density=1, facecolor='blue', alpha=0.75)  #第二个参数是直方图柱子的数量
    mu =np.mean(x) #计算均值 
    sigma =np.std(x) 
    num_bins = 30 #直方图柱子的数量 
    n, bins, patches = plt.hist(x, num_bins,density=1, alpha=0.75) 
    #直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象 
    # print(bins)
    y = norm.pdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y 
    # print(y)
    
    plt.grid(True)
    plt.plot(bins, y, 'r--') #绘制y的曲线 
    plt.xlabel('values') #绘制x轴 
    plt.ylabel('Probability') #绘制y轴 
    plt.title('Histogram : $\mu$=' + str(round(mu,2)) + ' $\sigma=$'+str(round(sigma,2)))  #中文标题 u'xxx' 
    #plt.subplots_adjust(left=0.15)#左边距 
    plt.show()

def drawLetterGaussion(letter, axis = "x"): # axis is "x" or "y"
    point_x = []
    point_y = []
    for person in PERSON:
        with open(SAVE_LP_DIR + person + "_" + letter + ".txt", "r") as f:
            data = json.loads(f.read())
            point_x += data[0]
            point_y += data[1]
    if (axis == "x"):
        drawGaussion(point_x)
    else:
        drawGaussion(point_y)

if __name__ == "__main__":
    # skip, total = saveLetterPoints()
    # print(skip, total)
    # drawPointCloudLetter("xq", ["A", "B", "P", "Y"])
    # drawSinglePointClouds()
    # drawSinglePointRec()
    drawLetterGaussion("L", "y")