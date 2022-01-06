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

PERSON = ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]
SAVE_MP_DIR = "../data/match_points/"  # save match points
SAVE_LP_DIR = "../data/letter_points/"  # save points sort by letter

allPattern = cleanWords()


def loadKeyPointLabel(person, sentence, word):
    with open(SAVE_MP_DIR + person + "_" + str(sentence) + "_" + str(word) + ".txt", "r") as f:
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
        if (len(key_point_label) > 0 and len(key_point_label) == len(set(key_point_label))):
            return [points[i] for i in key_point_label]
        else:
            return []
    except:
        print(key_point_label, points)


def saveLetterPoints(letter):
    all_chosen_points_x = []
    all_chosen_points_y = []
    all_chosen_points_d = []
    for person in PERSON:
        # person = "qlp"
        for i in range(1, 82):
            for j in range(len(allPattern[i - 1])):
                key_point = genKeyPoint(person, i, j)
                key_point_letter = allPattern[i - 1][j]
                if (len(key_point) == 0):
                    continue
                else:
                    for letter_id in range(len(key_point_letter)):
                        if(key_point_letter[letter_id] == letter or key_point_letter[letter_id] == lowerCase(letter)):
                            # print(key_point, i)
                            all_chosen_points_x.append(key_point[letter_id][0])
                            all_chosen_points_y.append(key_point[letter_id][1])
                            all_chosen_points_d.append(key_point[letter_id][2])
            print(i, "done")  # sentence done
        print(person, "          done           ")  # person done
        with open(SAVE_LP_DIR + person + "_" + letter + ".txt", "w") as f:
            f.write(json.dumps(
                [all_chosen_points_x, all_chosen_points_y, all_chosen_points_d]))

        # clean for next person
        all_chosen_points_x = []
        all_chosen_points_y = []
        all_chosen_points_d = []


def drawPointCloudLetter(person, letter):  # letter should be capitalized
    with open(SAVE_LP_DIR + person + "_" + letter + ".txt", "r") as f:
        data = json.loads(f.read())
        points_x = data[0]
        points_y = data[1]
        points_d = data[2]
        max_depth = max(points_d)
        colors = list(map(lambda x: '#' + str(hex(255 -
                                                  int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, points_d))
        for point in range(len(points_x)):
            plt.text(points_x[point], points_y[point], letter)
        plt.axis('equal')
        plt.scatter(points_x, points_y, c=colors)
        plt.show()


if __name__ == "__main__":
    # saveLetterPoints("Q")
    drawPointCloudLetter("xq", "Q")
