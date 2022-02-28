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
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.s_th = RecvThread()
        self.s_th.sinOut.connect(self.change)
        self.s_th.start()

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
            self.buttons.append(button)

        self.buttons[0].clicked.connect(lambda: self.choose_k(0))
        self.buttons[1].clicked.connect(lambda: self.choose_k(1))
        self.buttons[2].clicked.connect(lambda: self.choose_k(2))

        retry_button = QPushButton('Retry', self)
        retry_button.resize(150, 50)
        retry_button.move(100 + 3 * 200, 200)
        retry_button.clicked.connect(lambda: self.choose_k(-1))
        self.buttons.append(retry_button)

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