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

        self.target_letter = ''

        target_label = QLabel(self)
        target_label.setFont(QFont("Times New Roman", 30, QFont.Weight.Bold))
        target_label.setMinimumSize(100, 100)
        target_label.move(100, 100)
        self.target_label = target_label

        self.time = QTimer(self)
        self.time.setInterval(1000)
        self.time.timeout.connect(self.refresh)

        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle('Evaluation')
        self.show()

    def refresh(self):
        pass

    def change(self, num):
        self.target_letter = num
        self.target_label.setText(self.target_letter)


class RecvThread(QThread):
    sinOut = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super(RecvThread, self).__init__(parent)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(("localhost", 34826))

    def run(self):
        while True:
            data, addr = self.s.recvfrom(2048)
            num: str = eval(data.decode())
            print(num)
            self.sinOut.emit(num)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())