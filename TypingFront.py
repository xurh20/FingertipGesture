import sys
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QApplication,
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import socket


class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s_th = RecvThread()
        self.s_th.sinOut.connect(self.change)
        self.s_th.start()

        big_label = QLabel(self)
        big_label.setFont(QFont("Times New Roman", 30, QFont.Weight.Bold))
        big_label.setText("candidate")
        big_label.move(100, 100)
        self.big_label = big_label
        self.candiate = ""

        self.labels = []
        for i in range(3):
            label = QLabel(self)
            label.resize(150, 50)
            label.move(100 + i * 200, 200)
            label.setText(str(i))
            label.setFont(QFont("Microsoft YaHei", 20, QFont.Weight.Bold))
            self.labels.append(label)

        self.time = QTimer(self)
        self.time.setInterval(1000)
        self.time.timeout.connect(self.refresh)

        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle('Evaluation')
        self.show()

    def refresh(self):
        pass

    def choose_k(self, i):
        self.s.sendto(repr(i).encode('gbk'), ('localhost', 34827))

    def change(self, top):
        for i in range(3):
            self.labels[i].setText(top[i])
        self.candiate += top[0]
        self.big_label.setText(self.candiate)
        self.big_label.adjustSize()


class RecvThread(QThread):
    sinOut = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super(RecvThread, self).__init__(parent)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(("localhost", 34826))

    def run(self):
        while True:
            data, addr = self.s.recvfrom(2048)
            top: list = eval(data.decode())
            print("recv: ", top)
            self.sinOut.emit(top)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())