import os
import numpy as np
from tqdm import tqdm
from random import randint

BASE_DIR = "data"
DATA_DIRS = []
for dirpath, dirname, filename in os.walk(BASE_DIR):
    DATA_DIRS = dirname
    break
bar1 = tqdm(DATA_DIRS)
for dir in bar1:
    bar1.set_description("Augmenting data in " + os.path.join(BASE_DIR, dir))
    for i in range(4):
        p = os.path.join(BASE_DIR, dir + "_aug_%d" % i)
        if not os.path.exists(p):
            os.mkdir(p)
    for dirpath, dirname, filenames in os.walk(os.path.join(BASE_DIR, dir)):
        bar2 = tqdm(filenames)
        for filename in bar2:
            din = np.load(os.path.join(dirpath, filename))
            c, id = filename.split('.')[0].split('_')
            dout_up = []
            dout_down = []
            dout_left = []
            dout_right = []
            for mat in din:
                # up
                offset = randint(1, 5)
                dout1 = np.zeros(mat.shape)
                dout1[:-offset] = mat[offset:]
                dout_up.append(dout1)
                # down
                offset = randint(1, 5)
                dout1 = np.zeros(mat.shape)
                dout1[offset:] = mat[:-offset]
                dout_down.append(dout1)
                # left
                offset = randint(1, 5)
                dout1 = np.zeros(mat.shape)
                dout1[:, :-offset] = mat[:, offset:]
                dout_left.append(dout1)
                # right
                offset = randint(1, 5)
                dout1 = np.zeros(mat.shape)
                dout1[:, offset:] = mat[:, :-offset]
                dout_right.append(dout1)
            np.save(os.path.join(BASE_DIR, dir + "_aug_0", filename), dout_up)
            np.save(os.path.join(BASE_DIR, dir + "_aug_1", filename),
                    dout_down)
            np.save(os.path.join(BASE_DIR, dir + "_aug_2", filename),
                    dout_left)
            np.save(os.path.join(BASE_DIR, dir + "_aug_3", filename),
                    dout_right)
