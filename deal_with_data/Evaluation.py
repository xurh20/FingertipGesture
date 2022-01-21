import socket
import threading
import sys

sys.path.append('../sensel-lib-wrappers/sensel-lib-python')
import sensel
import numpy as np
import matplotlib.pyplot as plt

import argparse

from Pattern import get_best_n
from queue import PriorityQueue

HEIGHT = 105
WIDTH = 185
LEFT_BOUND = 79
UP_BOUND = 34
RIGHT_BOUND = 106
DOWN_BOUND = 68
MAX_LEN = 200

recording = False
interrupted = False
plotting = False

left_bound = 184
up_bound = 104
right_bound = 0
down_bound = 0

frame_series = []


def wait_for_enter():
    global recording, interrupted, plotting
    while True:
        try:
            if (recording):
                code = input('Press Enter to stop...')
                recording = False
                plotting = True
                if (code == 'q'):
                    interrupted = True
                    break
            else:
                code = input('Press Enter to start ...')
                if (code == 'q'):
                    interrupted = True
                    break
                recording = True
        except:
            interrupted = True
            break
    return


def open_sensel():
    handle = None
    error, device_list = sensel.getDeviceList()
    if device_list.num_devices != 0:
        error, handle = sensel.openDeviceByID(device_list.devices[0].idx)
    return handle


def init_frame():
    error = sensel.setFrameContent(handle, sensel.FRAME_CONTENT_PRESSURE_MASK)
    error, frame = sensel.allocateFrameData(handle)
    error = sensel.startScanning(handle)
    return frame


def scan_frames(frame, info: sensel.SenselSensorInfo):
    while not interrupted:
        error = sensel.readSensor(handle)
        error, num_frames = sensel.getNumAvailableFrames(handle)
        for i in range(num_frames):
            error = sensel.getFrame(handle, frame)
            if recording:
                save_frame(frame, info)


def save_frame(frame, info: sensel.SenselSensorInfo):
    global frame_series
    rows = info.num_rows
    cols = info.num_cols
    fs = np.zeros((DOWN_BOUND + 1 - UP_BOUND, RIGHT_BOUND + 1 - LEFT_BOUND))
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            fs[i - UP_BOUND][j - LEFT_BOUND] += frame.force_array[i * cols + j]
    frame_series.append(fs)


def close_sensel(frame):
    error = sensel.freeFrameData(handle, frame)
    error = sensel.stopScanning(handle)
    error = sensel.close(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    handle = open_sensel()
    if handle:
        error, info = sensel.getSensorInfo(handle)
        frame = init_frame()

        t = threading.Thread(target=wait_for_enter)
        t.setDaemon(True)
        t.start()
        u = threading.Thread(target=scan_frames, args=(frame, info))
        u.setDaemon(True)
        u.start()

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('localhost', 34827))

        total = 0
        top_k = [0] * 3
        prev = None
        while not interrupted:
            if plotting:
                q: PriorityQueue = get_best_n(np.array(frame_series), prev)
                top = []
                for i in range(3):
                    try:
                        top.append(q.get_nowait()[1])
                    except:
                        top.append("")
                print(top)
                s.sendto(repr(top).encode('gbk'), ('localhost', 34826))

                data, adddr = s.recvfrom(2048)
                total += 1
                idx = eval(data.decode())
                print("choose: ", idx)
                if idx < 0:
                    pass
                else:
                    top_k[idx] += 1
                    prev = top[idx]

                frame_series = []
                plotting = False
        print("total: %d" % total)
        for i in range(3):
            print("top%d acc: %f" % (i + 1, top_k[i] / total))
        close_sensel(frame)
