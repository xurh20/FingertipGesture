import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "../alphabet_data/"
candidates = [chr(y) for y in range(97, 123)]
LEFT_BOUND = 79
UP_BOUND = 34
RIGHT_BOUND = 106
DOWN_BOUND = 68
WIDTH = RIGHT_BOUND - LEFT_BOUND + 1
HEIGHT = DOWN_BOUND - UP_BOUND + 1

def loadData():
    data = np.load(BASE_DIR + candidates[5] + "_" + str(0) + ".npy")
    # print(data)
    return data

def plotData(data):
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
        if (sum_force > 0):
            for y_coordinate in range(len(frame)):
                for x_coordinate in range(len(frame[y_coordinate])):
                    rate = frame[y_coordinate][x_coordinate] / sum_force
                    x_average += rate * x_coordinate
                    y_average += rate * y_coordinate * -1
            points_x.append(x_average)
            points_y.append(y_average)
            depths.append(sum_force)
    return points_x, points_y, depths

if __name__ == "__main__":
    data = loadData()
    points_x, points_y, depths = plotData(data)
    points_x = list(map(lambda x : x - WIDTH / 2, points_x))
    points_y = list(map(lambda x : x + HEIGHT / 2, points_y))
    max_depth = max(depths)
    colors = list(map(lambda x : '#' + str(hex(255 - int(x / max_depth * 255)))[2:].rjust(2, '0') * 3, depths))
    # print(colors)
    plt.scatter(points_x, points_y, c=colors)
    # plt.xlim((-2, 2))
    # plt.ylim((0, 6))
    plt.show()