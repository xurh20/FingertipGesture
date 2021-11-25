from os import name
import numpy as np
from dtw import dtw
from numpy.lib.function_base import append

from plot import gatherCorner, calFirstCorner, calSecondCorner, loadData, calculatePoints

PERSON = ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]


def genVectors(points_x, points_y, depths):
    pointLabels = []
    vectors = []
    centerPointGroups = gatherCorner(calFirstCorner(
        points_x, points_y, depths), calSecondCorner(points_x, points_y, depths))
    for pointGroup in centerPointGroups:
        pointLabels.append(pointGroup[int(len(pointGroup) / 2)])
    points = [[points_x[i], points_y[i], depths[i]] for i in pointLabels]

    # calculate vector
    if (len(points) == 1):
        print("error only one point")
    for i in range(1, len(points)):
        v = np.array([points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]])
        v = v / np.linalg.norm(v)
        vectors.append(v)
    return vectors

if __name__ == "__main__":
    sentence = 1
    word = 0
    person = PERSON[0]
    data = loadData(sentence, word, person)
    points_x, points_y, depths = calculatePoints(data)
    genVectors(points_x, points_y, depths)
