import sys
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QApplication,
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import socket


class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.labels = []
        for i in range(3):
            label = QLabel(self)
            label.resize(150, 50)
            label.move(100 + i * 200, 100)
            label.setText(str(i))
            self.labels.append(label)

        self.buttons = []
        for i in range(3):
            button = QPushButton('Me!', self)
            button.resize(150, 50)
            button.move(100 + i * 200, 200)
            button.clicked.connect(self.choose_k(i))
            self.buttons.append(button)

        retry_button = QPushButton('Retry', self)
        retry_button.resize(150, 50)
        retry_button.move(100 + 3 * 200, 200)
        retry_button.clicked.connect(self.choose_k(-1))
        self.buttons.append(retry_button)

        self.recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv.bind(("localhost", 34826))

        self.send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.time = QTimer(self)
        self.time.setInterval(1000)
        self.time.timeout.connect(self.refresh)

        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle('Evaluation')
        self.show()

    def refresh(self):
        pass

    def choose_k(self, i):
        self.send.sendto(repr(i).encode('gbk'), ('localhost', 34827))

    def recv_top(self):
        while True:
            data, addr = self.recv.recvfrom(2048)
            top: list = eval(data)
            for i in range(3):
                self.labels[i].setText(top[i])


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())