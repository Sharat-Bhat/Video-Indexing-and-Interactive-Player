import os
import time
import sys
import cv2
import numpy as np
from NewMedia import VideoWindow
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import (QApplication)
from createTS.generateTS import generate_video, get_discontinuities, convert_to_scenes_and_shots
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

# frameSize = (480, 270)

# MOVIE_NAME = "The_Great_Gatsby_rgb"
# MOVIE_NAME = "Ready_Player_One_rgb"
# MOVIE_NAME = "The_Long_Dark_rgb"
# MOVIE_NAME = "SHERLOCK_rgb"
# MOVIE_NAME = "Jack_Sparrow_rgb"
# MOVIE_NAME = "Spiderverse_rgb"
# MOVIE_NAME = "Ignite_rgb"
MOVIE_NAME = "test4"
MOVIE_NAME = "test1"
# MOVIE_NAME = "test2"


class VideoConv:
    def __init__(self,height,width, fps, MOVIE_NAME):
        self.fps = fps
        self.width = width
        self.height = height
        self.frameSize = (width, height)
        self.mode = "BGR"
        self.MOVIE_NAME = MOVIE_NAME
        self.out = cv2.VideoWriter(os.path.join(self.MOVIE_NAME, 'output_video.avi'), cv2.VideoWriter_fourcc(*'H264'), self.fps, self.frameSize)

    def conv(self, input_video, input_audio, generate_combined_video=False):
        if generate_combined_video:
            filename = os.path.join(self.MOVIE_NAME, 'combined_video.mp4')
        else:
            filename = os.path.join(self.MOVIE_NAME, 'InputVideo.mp4')
        with open(input_video, "rb") as f:
            frame_num = 0
            finalArray = []
            while True:
                frame_data = f.read(self.width * self.height * 3)
                if not frame_data:
                    break
                byte_array = np.frombuffer(frame_data, dtype=np.uint8)
                byte_array = byte_array.reshape(self.height, self.width, 3)
                frame_num += 1
                finalArray.append(byte_array)
                if generate_combined_video:
                    self.out.write(cv2.cvtColor(byte_array, cv2.COLOR_RGB2BGR))
            if generate_combined_video:
                self.out.release()
                video = VideoFileClip(os.path.join(self.MOVIE_NAME, 'output_video.avi'))
                audio = AudioFileClip(input_audio)
                audio = audio.set_duration(video.duration)
                combined = video.set_audio(audio)
                combined.write_videofile(filename, fps=self.fps, threads=8, codec="libx264",bitrate='8000k',preset="ultrafast",logger="bar")
        return np.array(finalArray),filename

def read_video(movie_directory):
    readme_file_path = os.path.join(movie_directory, "readme.txt")
    f_rme = open(readme_file_path, 'r')
    lines = f_rme.read()
    lines = lines.strip()
    lines = lines.split("\n")
    fps = int(lines[0].split(':')[1])
    resolution = lines[1].split(':')[1].split('x')
    width = int(resolution[0])
    height = int(resolution[1])
    # num_frames = int(lines[2].split(':')[1])
    return width, height, fps

if __name__ == '__main__':
    start_time = time.time()
    PROJECT_PATH = "/Users/sharatbhat/Desktop/USC/CSCI576/Project/CSCI576-master/"
    MOVIE_PATH = os.path.join(PROJECT_PATH, MOVIE_NAME)
    width, height, fps = read_video(MOVIE_PATH)
    videoConv = VideoConv(height=height,width=width, fps=fps, MOVIE_NAME=MOVIE_PATH)
    audiofile = os.path.join(MOVIE_PATH, 'InputAudio.wav')
    videofile = os.path.join(MOVIE_PATH, 'InputVideo.rgb')
    video_data, filename = videoConv.conv(videofile, audiofile, False)
    byte_array = video_data
    # print(byte_array)
    error_entropies = generate_video(MOVIE_PATH, byte_array, fps)
    frames = get_discontinuities(MOVIE_PATH, error_entropies, fps)
    scenes_and_shots = convert_to_scenes_and_shots(byte_array,error_entropies,frames, fps)
    end_time = time.time()
    print("Processing Time = ", (end_time - start_time)/60)
    app = QApplication(sys.argv)
    player = VideoWindow(scenes=scenes_and_shots, fps=fps)
    player.openFile(filename)
    player.resize(900, 700)
    player.show()
    sys.exit(app.exec_())
