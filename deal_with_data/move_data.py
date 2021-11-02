import shutil
BASE_DIR = "../alphabet_data/"
FROM_DIR = "../alphabet_data_from_jjx/alphabet_data_lly/"
candidates = [chr(y) for y in range(97, 123)]

def moveData():
    for i in range(6):
        for j in range(30):
            shutil.copy(FROM_DIR + candidates[i] + "_" + str(j) + ".npy", BASE_DIR + candidates[i] + "_" + str(j + 91) + ".npy")

if __name__ == "__main__":
    moveData()