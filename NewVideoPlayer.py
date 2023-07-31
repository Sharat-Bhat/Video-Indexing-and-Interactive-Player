from PyQt5.QtCore import QDir, Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QMainWindow,
                             QAction, QShortcut, QCheckBox, QGridLayout, QTreeWidget, QTreeWidgetItem)
from PyQt5.QtGui import QIcon, QKeySequence
import sys
import os

MOVIE_NAME = "Jack_Sparrow_rgb" 
    
class VideoWindow(QMainWindow):

    def __init__(self, scenes,parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("TILU Media Player")
        self.setWindowIcon(QIcon("icon.png"))
        self.mediaPlayers = []
        self.videoWidgets = QVideoWidget
        self.selector = []
        for i in range(4):
            self.mediaPlayers.append(QMediaPlayer(
                None, QMediaPlayer.VideoSurface))
            self.videoWidgets = QVideoWidget()
            self.mediaPlayers[0].setVideoOutput(self.videoWidgets)


        self.videoWidgets.setFixedSize(QSize(480, 270))
        # create video layout
        upperLayout = QHBoxLayout()
        upperLayout.setContentsMargins(0, 0, 0, 0)
        upperLayout.addWidget(self.videoWidgets)
        # upperLayout.addWidget(self.videoWidgets[1])

        finalLayout = QVBoxLayout()
        finalLayout.setContentsMargins(0, 0, 0, 0)
        finalLayout.addLayout(upperLayout)
        # finalLayout.addLayout(bottomLayout)
        self.openFile(os.path.join(MOVIE_NAME, 'combined_video.mp4'))
        # Create play button and shortcuts
        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        playShortcut = QShortcut(QKeySequence("Space"), self)
        playShortcut.activated.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Maximum)

        sidebar = QVBoxLayout()
        sidebar.insertSpacing(0, 100)

        self.treeView = QTreeWidget(parent)
        # self.treeView.setColumnCount(1)
        self.treeView.setColumnWidth(0, 100)
        self.treeView.setMaximumSize(600, 400)
        self.treeView.setHeaderLabels(['Scenes', "Shots", "Subshots"])
        shot_count = 1
        scene_count = 1
        sub_shot_count = 1
        for scene in scenes.keys():
            department_item = QTreeWidgetItem(self.treeView)
            department_item.setData(0, Qt.UserRole, scene)
            department_item.setText(0, 'Scene ' + str(scene_count))
            scene_count += 1
            for shot in scenes[scene].keys():
                employee_item = QTreeWidgetItem(self.treeView)
                employee_item.setData(0, Qt.UserRole, shot)
                employee_item.setText(1, 'Shot ' + str(shot_count))
                department_item.addChild(employee_item)
                shot_count += 1

                for subshot in range(len(scenes[scene][shot])):
                    employee_item2 = QTreeWidgetItem(self.treeView)
                    employee_item2.setData(0, Qt.UserRole, scenes[scene][shot][subshot])
                    employee_item2.setText(2, 'Sub Shot ' + str(sub_shot_count))
                    employee_item.addChild(employee_item2)
                    sub_shot_count += 1
        sidebar.addWidget(self.treeView)

        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        mainScreen = QGridLayout()
        mainScreen.addLayout(finalLayout, 0, 0)
        mainScreen.addLayout(controlLayout, 3, 0)
        mainScreen.addWidget(self.errorLabel, 4, 0)
        sidebar.addStretch()
        mainScreen.addLayout(sidebar, 0, 1)

        # Set widget to contain window contents
        wid.setLayout(mainScreen)

        for i in self.mediaPlayers:
            i.stateChanged.connect(self.mediaStateChanged)
            i.positionChanged.connect(self.positionChanged)
            i.durationChanged.connect(self.durationChanged)
            i.error.connect(self.handleError)

    def openFile(self,fileName):
        print(fileName)
        if fileName != '':
            self.mediaPlayers[0].setMedia(
                QMediaContent(QUrl.fromLocalFile(fileName)))
            # self.playButton.setEnabled(True)

    def play(self):
        for i in self.mediaPlayers:
            if i.state() == QMediaPlayer.PlayingState:
                i.pause()
            else:
                i.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayers[0].state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        for i in self.mediaPlayers:
            i.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayers[0].errorString())


if __name__ == '__main__':
    filename = os.path.join(MOVIE_NAME, "InputVideo.mp4")
    app = QApplication(sys.argv)
    scenes_and_shots = {
        0: {166: [], 247: [], 416: [], 508: [], 901: [], 1076: [1136], 1179: [], 1356: [1821], 1867: [], 2336: [],
            2459: [], 2587: [], 2712: [], 3148: []}, 3244: {3311: []},
        3542: {3620: [], 3729: [3766], 3809: [3847], 3879: [], 3990: [4022], 4053: [4086], 4129: [], 4227: []},
        4351: {4487: [], 4724: []}, 4844: {5329: []}, 5598: {5754: [], 5956: []}, 6136: {}, 6302: {6477: []},
        6852: {6969: []}, 7047: {7462: []}, 7587: {},
        7668: {7843: [7878], 7908: [7931, 7956, 7971, 7990], 8008: [8031]},
        8080: {8080: [8103], 8190: [8213, 8268, 8286], 8301: [], 8369: [8487], 8515: []}}
    player = VideoWindow(scenes=scenes_and_shots)
    player.showMaximized()
    sys.exit(app.exec_())