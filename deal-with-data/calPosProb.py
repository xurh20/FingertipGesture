import json
import numpy as np
from scipy.stats import norm
from cleanWords import cleanWords, lowerCase
from draw import genKeyPoint

PERSON = ["gww", "hxz", "ljh", "lyh", "lzp", "qlp", "tty", "xq"]
LETTER = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
SAVE_LP_DIR = "../data/letter_points/"  # save points sort by letter
SAVE_NON_CENTER_LP_DIR = "../data/non_center_letter_points/" # not move to G as first point
SAVE_WRONG_LETTER_DIR = "../data/wrong_letter/"  # save wrong letter info

allPattern = cleanWords()
letterPosParas = {}

def calLetterPosProb():
    for letter in LETTER:
        point_x = []
        point_y = []
        for person in PERSON:
            with open(SAVE_NON_CENTER_LP_DIR + person + "_" + letter + ".txt", "r") as f:
                data = json.loads(f.read())
                point_x += data[0]
                point_y += data[1]
        x = np.array(point_x)
        mu_x =np.mean(x)
        sigma_x =np.std(x)
        y = np.array(point_y)
        mu_y =np.mean(y)
        sigma_y =np.std(y)
        letterPosParas[lowerCase(letter)] = [[mu_x, sigma_x], [mu_y, sigma_y]]
    print(letterPosParas)

def calWordPosProb():
    skip = 0
    total = 0
    for person in PERSON:
        # person = "qlp"
        for i in range(1, 82):
            for j in range(len(allPattern[i - 1])):
                key_point = genKeyPoint(person, i, j)
                key_point_word = allPattern[i - 1][j]
                total += 1
                if (key_point is None or len(key_point) == 0):
                    skip += 1
                    continue
                else:
                    for letter_id, letter in enumerate(key_point_word):
                        # x_prob = 2 * norm.cdf([key_point[letter_id][0]], letterPosParas[lowerCase(letter)][0][0], letterPosParas[lowerCase(letter)][0][1])
                        # if x_prob > 1:
                        #     x_prob = 2 - x_prob
                        # if x_prob < 0.1:
                        #     print("not good pos, letter is", letter, ", word is", key_point_word, ", person is", person, ", sentence number is", i, ", position is", key_point[letter_id])
                        # print(x_prob)
                        y_prob = 2 * norm.cdf([key_point[letter_id][1]], letterPosParas[lowerCase(letter)][1][0], letterPosParas[lowerCase(letter)][1][1])
                        if y_prob > 1:
                            y_prob = 2 - y_prob
                        if y_prob < 0.05:
                            print("not good pos, letter is", letter, ", word is", key_point_word, ", person is", person, ", sentence number is", i, ", position is", key_point[letter_id])
                        print(y_prob)
            print(i, "done")  # sentence done
        print(person, "          done           ")  # person done

    return skip, total

def saveWordPosProb():
    skip = 0
    total = 0
    for person in PERSON:
        # person = "qlp"
        wrong_word_num = 0
        for i in range(1, 82):
            for j in range(len(allPattern[i - 1])):
                key_point = genKeyPoint(person, i, j)
                key_point_word = allPattern[i - 1][j]
                total += 1
                if (key_point is None or len(key_point) == 0):
                    skip += 1
                    continue
                else:
                    wrong_word = False
                    for letter_id, letter in enumerate(key_point_word):
                        x_prob = 2 * norm.cdf([key_point[letter_id][0]], letterPosParas[lowerCase(letter)][0][0], letterPosParas[lowerCase(letter)][0][1])
                        if x_prob > 1:
                            x_prob = 2 - x_prob
                        # if x_prob < 0.1:
                        #     print("not good pos, letter is", letter, ", word is", key_point_word, ", person is", person, ", sentence number is", i, ", position is", key_point[letter_id])
                        # print(x_prob)
                        y_prob = 2 * norm.cdf([key_point[letter_id][1]], letterPosParas[lowerCase(letter)][1][0], letterPosParas[lowerCase(letter)][1][1])
                        if y_prob > 1:
                            y_prob = 2 - y_prob
                        if y_prob < 0.01 or x_prob < 0.01:
                            if (wrong_word == False):
                                wrong_word = True
                                wrong_word_num += 1
                            # with open(SAVE_WRONG_LETTER_DIR + person + ".txt", "a") as f:
                            #     f.write("not good pos, letter is" + letter + ", word is" + key_point_word + ", person is" + person + ", sentence number is" + str(i) + ", position is" + str(key_point[letter_id]))
                            #     f.write("\n")
                            #     f.write(str(y_prob))
                            #     f.write("\n")
            print(i, "done")  # sentence done
        print(person, "          done           ", "wrong word num", wrong_word_num)  # person done

    return skip, total

if __name__ == "__main__":
    calLetterPosProb()
    # calWordPosProb()
    saveWordPosProb()