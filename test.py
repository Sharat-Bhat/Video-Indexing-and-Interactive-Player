import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPainter, QImage
from PyQt5.QtWidgets import QApplication, QOpenGLWidget


class VideoRenderer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = "combined_video.mp4"
        self.cap = cv2.VideoCapture(self.video_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(33)  # 30 FPS


    def paintGL(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.painter = QPainter(self)
        self.painter.drawImage(self.rect(), img, img.rect())

    def resizeGL(self, width, height):
        self.painter.setViewport(0, 0, width, height)
        self.painter.setWindow(-width / 2, -height / 2, width, height)


if __name__ == '__main__':
    app = QApplication([])
    renderer = VideoRenderer()
    renderer.show()
    app.exec_()