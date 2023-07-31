from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QMainWindow


class Display(QMainWindow):
    def __init__(self, frames):
        super().__init__()

        self.frames = frames
        self.num_frames = len(frames)
        self.width = frames[0].shape[1]
        self.height = frames[0].shape[0]
        self.fps = 30

        self.frame_idx = 0

        self.init_ui()
        self.show()

    def init_ui(self):
        self.setWindowTitle("Video Player")
        self.label = QLabel()
        self.setCentralWidget(self.label)
        self.statusBar().hide()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(30))

    def update_frame(self):
        pixmap = QPixmap.fromImage(QImage(self.frames[self.frame_idx].data, self.width, self.height, QImage.Format_RGB888))
        self.label.setPixmap(pixmap)
        self.frame_idx = (self.frame_idx + 1) % self.num_frames
        # self.show()

