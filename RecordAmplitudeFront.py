import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import socket
import numpy as np


class Example(QWidget):
    IDLE = 0
    RUNNING = 1

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.s_th = RecvThread()
        self.s_th.sinOut.connect(self.change)
        self.s_th.start()

        self.resize(500, 500)

        self.phase = 0
        self.num_amplitude = 0
        self.target_direction = -1
        self.target_amplitude = -1

        button = QPushButton('Reset', self)
        button.clicked.connect(lambda: self.reset())
        button.resize(150, 50)
        button.move(600, 200)
        self.resetButton = button

        self.time = QTimer(self)
        self.time.setInterval(1000)
        self.time.timeout.connect(self.refresh)

        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle('Evaluation')
        self.show()

    def refresh(self):
        pass

    def reset(self):
        self.phase = self.IDLE
        self.target_amplitude = -1
        self.update()

    def change(self, num):
        if self.phase == self.IDLE:
            self.num_amplitude = num
            self.phase = self.RUNNING
        else:
            self.target_direction = num // self.num_amplitude
            self.target_amplitude = num % self.num_amplitude
        self.update()

    def paintEvent(self, a0: QPaintEvent) -> None:
        qp = QPainter()
        qp.begin(self)
        self.drawTargets(qp)
        qp.end()

    def drawTargets(self, qp: QPainter):
        center = (250, 250)
        angles = [(i - 4 / 2) * np.pi / (4 / 2) for i in range(4)]
        r = 200

        for j, a in enumerate(angles):
            for i in range(self.num_amplitude):
                if i == self.target_amplitude and j == self.target_direction:
                    qp.setBrush(QColor(220, 20, 60))
                else:
                    qp.setBrush(QColor(119, 136, 153))
                qp.drawEllipse(
                    QPoint(
                        int(center[0] + r * np.cos(a) *
                            (i + 1) / self.num_amplitude),
                        int(center[1] - r * np.sin(a) *
                            (i + 1) / self.num_amplitude)), 30, 30)


class RecvThread(QThread):
    sinOut = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super(RecvThread, self).__init__(parent)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(("localhost", 34826))

    def run(self):
        while True:
            data, addr = self.s.recvfrom(2048)
            num: int = eval(data.decode())
            self.sinOut.emit(num)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())