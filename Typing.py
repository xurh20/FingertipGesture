import socket
import threading
import sys
import traceback

sys.path.append('sensel-lib-wrappers/sensel-lib-python')
import sensel
import numpy as np
import matplotlib.pyplot as plt
import argparse
from queue import Queue
from Plot import getConfidenceQueue, plotOneLettersCorner, getConfidenceQueue8

candidates = [chr(y) for y in range(97, 123)]

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

frame_series = Queue()


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
            save_frame(frame, info)


def save_frame(frame, info: sensel.SenselSensorInfo):
    global frame_series
    rows = info.num_rows
    cols = info.num_cols
    fs = np.zeros((DOWN_BOUND + 1 - UP_BOUND, RIGHT_BOUND + 1 - LEFT_BOUND))
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            fs[i - UP_BOUND][j - LEFT_BOUND] += frame.force_array[i * cols + j]
    frame_series.put(fs)


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

        u = threading.Thread(target=scan_frames, args=(frame, info))
        u.setDaemon(True)
        u.start()

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('localhost', 34827))

        total = 0
        top_k = [0] * 3

        PRESSURE_THRESHOLD = 500
        FRAME_WINDOW = 10
        while True:
            while np.sum(frame_series.get()) <= PRESSURE_THRESHOLD:
                pass
            frames = []
            while np.sum(fs := frame_series.get()) > PRESSURE_THRESHOLD:
                frames.append(fs)
            if len(frames) < FRAME_WINDOW:
                continue
            try:
                q = getConfidenceQueue8(frames)

                top = []
                for i in range(3):
                    try:
                        top.append(q.get_nowait()[1])
                    except:
                        top.append("")
                print(top)
                s.sendto(repr(top).encode('gbk'), ('localhost', 34826))
            except:
                # print("Deprecated data. Please try again.")
                # traceback.print_exc()
                continue
        print("total: %d" % total)
        for i in range(1, 3):
            top_k[i] += top_k[i - 1]
        for i in range(3):
            print("top%d acc: %f" % (i + 1, top_k[i] / total))
        close_sensel(frame)