import json

def cleanWords():
    with open("../alphabet_data_from_jjx/lyh.txt", "r") as f:
        data = f.read().split("\n")
        print(data)

if __name__ == "__main__":
    cleanWords()