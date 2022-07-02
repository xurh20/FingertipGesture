import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import socket
import numpy as np


class Example(QWidget):
    IDLE = 0
    FIRST = 1
    SECOND = 2
    POINTS = [
        QPoint(150, 150),
        QPoint(250, 150),
        QPoint(350, 150),
        QPoint(150, 250),
        QPoint(250, 250),
        QPoint(350, 250),
        QPoint(150, 350),
        QPoint(250, 350),
        QPoint(350, 350)
    ]

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
        self.target_direction = (-1, -1, -1)

        self.time = QTimer(self)
        self.time.setInterval(333)
        self.time.timeout.connect(self.refresh)
        self.time.start()

        self.setGeometry(300, 300, 1000, 500)
        self.setWindowTitle('Evaluation')
        self.show()

    def refresh(self):
        self.phase = (self.phase + 1) % 3
        self.update()

    def change(self, num):
        self.target_direction = num

    def paintEvent(self, a0: QPaintEvent) -> None:
        qp = QPainter()
        qp.begin(self)
        self.drawTargets(qp)
        qp.end()

    def drawTargets(self, qp: QPainter):
        for i, point in enumerate(self.POINTS):
            if i == self.target_direction[0]:
                qp.setBrush(QColor(220, 20, 60))
            else:
                qp.setBrush(QColor(119, 136, 153))
            qp.drawEllipse(point, 10, 10)

        if self.phase == self.FIRST and self.target_direction[0] >= 0:
            self.drawArrow(qp, self.POINTS[self.target_direction[0]],
                           self.POINTS[self.target_direction[1]])
        if self.phase == self.SECOND and self.target_direction[1] >= 0:
            qp.setPen(QPen(QColor(220, 20, 60), 5))
            qp.drawLine(self.POINTS[self.target_direction[0]],
                        self.POINTS[self.target_direction[1]])
            self.drawArrow(qp, self.POINTS[self.target_direction[1]],
                           self.POINTS[self.target_direction[2]])

    def drawArrow(self, qp: QPainter, start: QPoint, end: QPoint):
        l = QLineF(start, end)
        v = l.unitVector()
        v.setLength(20)
        v.translate(QPointF(l.dx(), l.dy()))

        n = v.normalVector()
        n.setLength(n.length() * 0.5)
        n2 = n.normalVector().normalVector()

        p1 = v.p2()
        p2 = n.p2()
        p3 = n2.p2()

        qp.setPen(QPen(QColor(220, 20, 60), 5))
        qp.drawLine(l)
        qp.setBrush(QColor(220, 20, 60))
        qp.drawPolygon(p1, p2, p3)


class RecvThread(QThread):
    sinOut = pyqtSignal(tuple)

    def __init__(self, parent=None) -> None:
        super(RecvThread, self).__init__(parent)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(("localhost", 34826))

    def run(self):
        while True:
            data, addr = self.s.recvfrom(2048)
            num: tuple = eval(data.decode())
            self.sinOut.emit(num)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())