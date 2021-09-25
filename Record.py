import threading
import sys

sys.path.append('sensel-lib-wrappers/sensel-lib-python')
import sensel
import numpy as np
import matplotlib.pyplot as plt

candidates = [chr(y) for y in range(97, 123)]

HEIGHT = 105
WIDTH = 185
LEFT_BOUND = 160
UP_BOUND = 70

recording = False
interrupted = False
plotting = False

# left_bound = 184
# up_bound = 104

sum_frame = np.zeros((HEIGHT, WIDTH))
frame_series = []


def wait_for_enter():
    global recording, interrupted, plotting
    candidate_index = 0
    while True:
        try:
            if (recording):
                code = input('Press Enter to stop...')
                recording = False
                plotting = True
                candidate_index += 1
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


def print_bound_frame(frame, info: sensel.SenselSensorInfo):
    # global left_bound, up_bound
    rows = info.num_rows
    cols = info.num_cols
    for i in range(UP_BOUND, info.num_rows):
        for j in range(LEFT_BOUND, info.num_cols):
            if (frame.force_array[i * cols + j] > 0.0):
                left_bound = min(left_bound, j)
                up_bound = min(up_bound, i)


def print_max_frame(frame, info: sensel.SenselSensorInfo):
    global sum_frame
    rows = info.num_rows
    cols = info.num_cols
    max_point = (0, 0)
    max_pressure = -1.0
    for i in range(UP_BOUND, info.num_rows):
        for j in range(LEFT_BOUND, info.num_cols):
            # sum_frame[i][j] += frame.force_array[i * cols + j]
            if (frame.force_array[i * cols + j] > max_pressure):
                max_pressure = frame.force_array[i * cols + j]
                max_point = (i, j)
            # if (frame.force_array[i * cols + j] > 0.0):
            #     left_bound = min(left_bound, j)
            #     up_bound = min(up_bound, i)
    sum_frame[max_point[0]][max_point[1]] += 1.0


def save_frame(frame, info: sensel.SenselSensorInfo):
    global frame_series
    rows = info.num_rows
    cols = info.num_cols
    fs = np.zeros((rows - UP_BOUND, cols - LEFT_BOUND))
    for i in range(UP_BOUND, info.num_rows):
        for j in range(LEFT_BOUND, info.num_cols):
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
        candidate_index = 0
        while not interrupted:
            if plotting:
                # plot_frame()
                np.save(
                    "alphabet_data/" + candidates[candidate_index] + ".npy",
                    frame_series)
                candidate_index += 1
                plotting = False
            if (candidate_index >= len(candidates)):
                break
        close_sensel(frame)