import json
SAVE_CB_DIR = "../data/break_num/"

def loadRawNum():
    nums = []
    with open("../alphabet_data_from_jjx/lyh.txt", "r") as f:
        data = f.read().split("\n")
        for i in data:
            num = []
            words = i.split(" ")
            for word in words:
                num.append(len(word))
            nums.append(num)
    return nums

def checkBreak(person):
    rawNums = loadRawNum()
    with open(SAVE_CB_DIR + person + ".txt", "r") as f:
        readNums = json.loads(f.read())
    all_sentence = []
    for i in range(len(rawNums)):
        sentence = []
        if len(rawNums[i]) > len(readNums[i]):
            print(person, " ", str(i), " error")
            continue
        for j in range(len(rawNums[i])):
            sentence.append(rawNums[i][j] - readNums[i][j])
        all_sentence.append(sentence)
    return all_sentence

if __name__ == "__main__":
    print(checkBreak("qlp"))