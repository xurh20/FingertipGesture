import json
import matplotlib.pyplot as plt
import numpy as np
from draw import genKeyPoint
from cleanWords import cleanWords, lowerCase

SAVE_PRS_DIR = "../data/letter_pressure/"  # save letter pressure
PERSON = ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]
LETTER = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
LOWERLETTER = [chr(i) for i in range(ord('a'), ord('z') + 1)]
allPattern = cleanWords()

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

def letterToNumber(letter):
    """
    convert a letter (lowercase or uppercase) to number from 0 to 25
    """
    letter = lowerCase(letter)
    return LOWERLETTER.index(letter)

def letterToDistance(letter):
    """
    convert a letter (lowercase or uppercase) to distance from G
    """
    letter = lowerCase(letter)
    return np.linalg.norm(STD_KB_POS[letter] - STD_KB_POS['g'])

def calPressure():
    total = 0
    skip = 0
    all_pressure = []
    for i in range(26):
        all_pressure.append([])
    print(all_pressure)
    for person in PERSON:
        # person = "qlp"
        print(person)
        for i in range(2, 82):
            for j in range(len(allPattern[i - 1])):
                key_point = genKeyPoint(person, i, j)
                key_point_letter = allPattern[i - 1][j]
                total += 1
                if (key_point is None or len(key_point) == 0):
                    skip += 1
                    continue
                else:
                    for letter_id in range(len(key_point_letter)):
                        all_pressure[letterToNumber(key_point_letter[letter_id])].append(key_point[letter_id][2])
            print("done", i)
        with open(SAVE_PRS_DIR + person + ".txt", "w") as f:
            f.write(json.dumps(all_pressure))
            all_pressure = []
            for i in range(26):
                all_pressure.append([])

def drawPressureLetterDependency():
    for person in PERSON:
        with open(SAVE_PRS_DIR + person + ".txt", "r") as f:
            data = json.loads(f.read())
            for i in range(26):
                plt.scatter([LETTER[i]] * len(data[i]), data[i])
        plt.show()
                
def drawPressureDistDependency():
    for person in PERSON:
        with open(SAVE_PRS_DIR + person + ".txt", "r") as f:
            data = json.loads(f.read())
            for i in range(26):
                plt.scatter([letterToDistance(LETTER[i])] * len(data[i]), data[i])
        plt.show()

def drawPressureDistDependencySingle():
    for person in PERSON:
        with open(SAVE_PRS_DIR + person + ".txt", "r") as f:
            data = json.loads(f.read())
            for i in range(26):
                plt.scatter([LETTER[i]], sum(data[i]) / len(data[i]))
        plt.show()

if __name__ == "__main__":
    calPressure()
