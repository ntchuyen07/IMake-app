import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QWidget, QGraphicsScene, QLayout
from main import Ui_MainWindow
from widget import Ui_Form
from PIL import Image
from PIL.ImageQt import ImageQt


class BasicWidget(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


class Test(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # bw = BasicWidget()
        self.pushButton.clicked.connect(self.abc)
        # bw.pushButton.clicked.connect(self.abc)
        # self.stackedWidget.insertWidget(0, bw)
        # self.stackedWidget.insertWidget(1, bw)
        print(self.stackedWidget.count())
        # im = Image.open("cat.jpg")
        #
        # img = ImageQt(im)
        # tes = QImage(img)
        # pixmap = QPixmap.fromImage(tes)
        # print(pixmap)
        self.load()

    def scale_image(self, width, height):
        print(self.graphicsView.frameGeometry().height())
        k = self.graphicsView.frameGeometry().height() / height
        if width * k <= self.graphicsView.frameGeometry().width():
            w = width * k
            h = self.graphicsView.frameGeometry().height()
        else:
            k = self.graphicsView.frameGeometry().width() / width
            w = self.graphicsView.frameGeometry().width()
            h = height * k

        return w, h

    def load(self):
        im = Image.open("cat.jpg")

        img = ImageQt(im)
        tes = QImage(img)
        pixmap = QPixmap.fromImage(tes)
        w, h = self.scale_image(pixmap.width(), pixmap.height())
        print(w, h)
        pixmap = pixmap.scaled(int(w), int(h))
        scense = QGraphicsScene()
        scense.addPixmap(pixmap.copy())
        self.graphicsView.setScene(scense)

    def abc(self):
        print("asdf")
        self.statusbar.showMessage("asdfasdf")


def gui():
    app = QtWidgets.QApplication(sys.argv)

    window = Test()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    gui()
