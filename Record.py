import threading
import sys

sys.path.append('sensel-lib-wrappers/sensel-lib-python')
import sensel
import numpy as np
import matplotlib.pyplot as plt

import argparse

candidates = [chr(y) for y in range(97, 123)]

HEIGHT = 105
WIDTH = 185
LEFT_BOUND = 79
UP_BOUND = 34
RIGHT_BOUND = 106
DOWN_BOUND = 68

recording = False
interrupted = False
plotting = False
candidate_index = 0
current_record_num = 0

left_bound = 184
up_bound = 104
right_bound = 0
down_bound = 0

sum_frame = np.zeros((HEIGHT, WIDTH))
frame_series = []

frame_operator = lambda x: None


def wait_for_enter():
    global recording, interrupted, plotting, candidate_index, current_record_num
    while True:
        try:
            if (recording):
                code = input('Press Enter to stop...')
                recording = False
                plotting = True
                # candidate_index += 1
                if (code == 'q'):
                    interrupted = True
                    break
            else:
                code = input('Press Enter to start record of ' +
                             candidates[candidate_index])
                recording = True
                if (code == 'q'):
                    interrupted = True
                    break
                elif (code == 'c'):
                    if (candidate_index < 26):
                        candidate_index += 1
                        current_record_num = 0
                    continue
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
                frame_operator(frame, info)


def print_bound_frame(frame, info: sensel.SenselSensorInfo):
    global left_bound, up_bound, right_bound, down_bound
    rows = info.num_rows
    cols = info.num_cols
    for i in range(0, info.num_rows):
        for j in range(0, info.num_cols):
            if (frame.force_array[i * cols + j] > 0.1):
                left_bound = min(left_bound, j)
                up_bound = min(up_bound, i)
                right_bound = max(right_bound, j)
                down_bound = max(down_bound, i)


def print_max_frame(frame, info: sensel.SenselSensorInfo):
    global sum_frame
    rows = info.num_rows
    cols = info.num_cols
    max_point = (0, 0)
    max_pressure = -1.0
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            # sum_frame[i][j] += frame.force_array[i * cols + j]
            if (frame.force_array[i * cols + j] > max_pressure):
                max_pressure = frame.force_array[i * cols + j]
                max_point = (i, j)
            # if (frame.force_array[i * cols + j] > 0.0):
            #     left_bound = min(left_bound, j)
            #     up_bound = min(up_bound, i)
    sum_frame[max_point[0]][max_point[1]] += 1.0


def print_sum_frame(frame, info: sensel.SenselSensorInfo):
    global sum_frame
    rows = info.num_rows
    cols = info.num_cols
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            sum_frame[i][j] += frame.force_array[i * cols + j]


def save_frame(frame, info: sensel.SenselSensorInfo):
    global frame_series
    rows = info.num_rows
    cols = info.num_cols
    fs = np.zeros((DOWN_BOUND + 1 - UP_BOUND, RIGHT_BOUND + 1 - LEFT_BOUND))
    for i in range(UP_BOUND, DOWN_BOUND + 1):
        for j in range(LEFT_BOUND, RIGHT_BOUND + 1):
            fs[i - UP_BOUND][j - LEFT_BOUND] += frame.force_array[i * cols + j]
    frame_series.append(fs)


def plot_frame():
    global sum_frame
    plt.imshow(sum_frame, cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()
    sum_frame = np.zeros((HEIGHT, WIDTH))


def close_sensel(frame):
    error = sensel.freeFrameData(handle, frame)
    error = sensel.stopScanning(handle)
    error = sensel.close(handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",
                        "--bound",
                        action="store_true",
                        help="print record bounds")
    parser.add_argument("-m",
                        "--max",
                        action="store_true",
                        help="print sum of max points of each frame")
    parser.add_argument("-r",
                        "--record",
                        action="store_true",
                        help="record frames")
    parser.add_argument("-s",
                        "--sum",
                        action="store_true",
                        help="print sum of each frames")
    args = parser.parse_args()
    if args.bound:
        frame_operator = print_bound_frame
    elif args.max:
        frame_operator = print_max_frame
    elif args.record:
        frame_operator = save_frame
    elif args.sum:
        frame_operator = print_sum_frame

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

        while not interrupted:
            if plotting:
                if args.bound:
                    print(left_bound, right_bound, up_bound, down_bound)
                elif args.max:
                    plot_frame()
                elif args.record:
                    np.save(
                        "alphabet_data/" + candidates[candidate_index] + "_" +
                        str(current_record_num) + ".npy", frame_series)
                elif args.sum:
                    plot_frame()

                frame_series = []
                current_record_num += 1
                plotting = False
            if (candidate_index >= len(candidates)):
                break
        close_sensel(frame)