import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QUrl, QSize, QTimer, QRect
from PyQt5.QtGui import QOpenGLContext, QOpenGLVersionProfile, QImage
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel,
                             QSizePolicy, QSlider, QStyle, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QOpenGLWidget)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton


class VideoWindow(QMainWindow):

    def __init__(self, scenes, parent=None, fps=30):
        super(VideoWindow, self).__init__(parent)
        self.fps = fps
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # self.mediaPlayer.
        videoWidget = QVideoWidget()
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.stopButton = QPushButton()
        self.stopButton.setEnabled(True)
        self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stopButton.clicked.connect(self.stop)
        # self.positionSlider = QSlider(Qt.Horizontal)
        # self.positionSlider.setRange(0, 0)
        # self.positionSlider.sliderMoved.connect(self.setPosition)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_func)
        self.timer.start(60)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Maximum)

        wid = QWidget(self)
        self.setCentralWidget(wid)

        layout = QHBoxLayout()

        controlLayout = QVBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stopButton)
        # controlLayout.addWidget(self.positionSlider)

        self.treeView = QTreeWidget(parent)
        # self.treeView.setColumnCount(1)
        self.treeView.setColumnWidth(0, 100)
        self.treeView.setMaximumSize(600, 400)
        self.treeView.setHeaderLabels(['Scenes', "Shots", "Subshots"])
        layout.addWidget(self.treeView)
        newLayout2 = QVBoxLayout()

        newWidget = QWidget()
        newWidget.setFixedSize(QSize(480, 270))
        newLayout = QVBoxLayout()
        newLayout.addWidget(videoWidget)
        newWidget.setLayout(newLayout)
        newLayout2.addWidget(newWidget)
        newLayout2.addLayout(controlLayout)
        layout.addLayout(newLayout2)

        self.current_scene = 0
        self.current_scene_item = 0

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.treeView.itemClicked.connect(self.handleTreeItemClick)
        self.tree_obj = {}

        scene_count = 1
        for scene in scenes.keys():
            department_item = QTreeWidgetItem(self.treeView)
            department_item.setData(0, Qt.UserRole, scene)
            department_item.setText(0, 'Scene ' + str(scene_count))
            shot_count = 2
            for shot in scenes[scene].keys():
                sub_shot_count = 2
                if shot != scene:
                    employee_item = QTreeWidgetItem(self.treeView)
                    employee_item.setData(0, Qt.UserRole, shot)
                    employee_item.setText(1, 'Shot ' + str(scene_count) + '.' + str(shot_count))
                    department_item.addChild(employee_item)
                    for subshot in range(len(scenes[scene][shot])):
                        employee_item2 = QTreeWidgetItem(self.treeView)
                        employee_item2.setData(0, Qt.UserRole, scenes[scene][shot][subshot])
                        employee_item2.setText(2, 'Sub Shot ' + str(scene_count) + '.' + str(shot_count) + '.' + str(
                            sub_shot_count))
                        employee_item.addChild(employee_item2)
                        sub_shot_count += 1
                    shot_count += 1
            scene_count += 1

        wid.setLayout(layout)

    def stop(self):
        selected_items = self.treeView.selectedItems()
        if len(selected_items) > 0:
            selected_item = selected_items[0]
            scene = selected_item.data(0, Qt.UserRole)
            self.setPosition(np.ceil(int(scene) * 1000 / self.fps))
            if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer.pause()
        else:
            self.setPosition(0)
            if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer.pause()

    def handleTreeItemClick(self, item):
        scene = item.data(0, Qt.UserRole)
        print(scene)
        self.setPosition(np.ceil(int(scene) * 1000 / self.fps))

    def openFile(self, fileName):
        print(fileName)
        if fileName != '':
            self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)

    def exitCall(self):
        pass

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.timer.stop()
        else:
            self.mediaPlayer.play()
            self.timer.start()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def update_func(self):
        current_scene = np.ceil((self.mediaPlayer.position() * self.fps) / 1000)
        if current_scene != self.current_scene:
            self.current_scene = current_scene
            for i in range(self.treeView.topLevelItemCount()):
                scene_item = self.treeView.topLevelItem(i)
                last_scene_item = self.treeView.topLevelItem(self.treeView.topLevelItemCount() - 1)
                last_scene_data = last_scene_item.data(0, Qt.UserRole)
                data = scene_item.data(0, Qt.UserRole)
                if data > current_scene:
                    scene_item = self.treeView.topLevelItem(i - 1)
                    if scene_item != self.current_scene_item:
                        self.treeView.setCurrentItem(scene_item)
                        self.current_scene_item = scene_item
                        break
                    break
                elif current_scene > last_scene_data:
                    scene_item = self.treeView.topLevelItem(self.treeView.topLevelItemCount() - 1)
                    if scene_item != self.current_scene_item:
                        self.treeView.setCurrentItem(scene_item)
                        self.current_scene_item = scene_item
                #
                # else:
                #     print("HERE")
                #     scene_item = self.treeView.topLevelItem(self.treeView.topLevelItemCount()-1)
                #     self.treeView.setCurrentItem(scene_item)
                #     self.current_scene_item = scene_item

                # else:
                #     scene_item = self.treeView.topLevelItem(i)
                #     self.current_scene_item = scene_item

    # def durationChanged(self, duration):
    #     self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())
