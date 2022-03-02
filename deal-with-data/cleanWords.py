import numpy as np
import json

from plot import calAngle

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


def lowerCase(letter):
    if (ord(letter) >= ord("A") and ord(letter) <= ord("Z")):
        return chr(ord(letter) - ord("A") + ord("a"))
    else:
        return letter


def cleanWords():  # start from zero
    with open("../data/voc.txt", "r") as f:
        data = f.read().split("\n")
        for i in range(len(data)):
            newSentence = []
            words = data[i].split(" ")
            for word in words:
                # clean same letter
                newWord = word[0]
                if (len(word) > 1):
                    lastLetter = word[0]
                    for j in range(1, len(word)):
                        if (lastLetter != word[j] and word[j] != '’'):
                            newWord += word[j]
                            lastLetter = word[j]

                # clean straight letter
                newWord_1 = newWord[0]
                if (len(newWord) > 2):
                    for j in range(1, len(newWord) - 1):
                        lastVector = STD_KB_POS[lowerCase(
                            newWord[j])] - STD_KB_POS[lowerCase(
                                newWord[j - 1])]
                        nowVector = STD_KB_POS[lowerCase(
                            newWord[j + 1])] - STD_KB_POS[lowerCase(
                                newWord[j])]
                        if (lastVector[1] == 0 and nowVector[1] == 0
                                and lastVector[0] * nowVector[0] > 0):
                            continue
                        else:
                            newWord_1 += newWord[j]
                            lastVector = nowVector
                    newWord_1 += newWord[-1]
                elif (len(newWord) == 2):
                    newWord_1 = newWord[0]
                    newWord_1 += newWord[1]
                newSentence.append(newWord_1)
            data[i] = newSentence
        return data


def clean(word):
    newWord = word[0]
    if (len(word) > 1):
        lastLetter = word[0]
        for j in range(1, len(word)):
            if (lastLetter != word[j] and word[j] != '’'):
                newWord += word[j]
                lastLetter = word[j]

    # clean straight letter
    newWord_1 = newWord[0]
    if (len(newWord) > 2):
        for j in range(1, len(newWord) - 1):
            lastVector = STD_KB_POS[str.lower(
                newWord[j])] - STD_KB_POS[str.lower(newWord[j - 1])]
            nowVector = STD_KB_POS[str.lower(
                newWord[j + 1])] - STD_KB_POS[str.lower(newWord[j])]
            if (lastVector[1] == 0 and nowVector[1] == 0
                    and lastVector[0] * nowVector[0] > 0):
                continue
            else:
                newWord_1 += newWord[j]
                lastVector = nowVector
        newWord_1 += newWord[-1]
    elif (len(newWord) == 2):
        newWord_1 = newWord[0]
        newWord_1 += newWord[1]
    return newWord_1


if __name__ == "__main__":
    print(cleanWords())