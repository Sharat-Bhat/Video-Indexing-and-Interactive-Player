import os 
import cv2
import numpy as np
import subprocess

MOVIE_NAME = "The_Dark_Knight_rgb"
MOVIE_NAME = "God_of_War_rgb"
MOVIE_NAME = "FRIENDS_rgb"
# MOVIE_NAME = "Sherlock_rgb"
# MOVIE_NAME = "GOT_rgb"
MOVIE_NAME = "Jack_Sparrow_rgb"
MOVIE_NAME = "Ignite_rgb"
# MOVIE_NAME = "Spiderverse_rgb"

cap = cv2.VideoCapture(os.path.join(MOVIE_NAME, "InputVideo.mp4"))

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    
vid = []

count = 0
# Read until video is completed
while(cap.isOpened()):
    
    count += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret != True:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = frame.shape
    vid.append(frame)
 
# When everything done, release the video capture object
cap.release()
num_frames = len(vid)
print("Yo!!!!!!!!!!!!")
with open(os.path.join(MOVIE_NAME, "readme.txt"), 'w') as f:
    f.write("fps: 30\n")
    f.write("resolution: "+str(width)+"x"+str(height)+"\n")
    f.write("number of frame: "+str(num_frames)+"\n")

print("Lol!!!!!!!!!!")

# vid_bytes = vid.flatten().tobytes()
with open(os.path.join(MOVIE_NAME, "InputVideo.rgb"), 'wb') as f_rgb:
    for frame in vid:
        # print(count)
        count -= 1
        f_rgb.write(frame.flatten().tobytes())

command = "ffmpeg -i "+os.path.join(MOVIE_NAME, "InputVideo.mp4")+" -ab 160k -ac 2 -ar 44100 -vn "+os.path.join(MOVIE_NAME, "InputAudio.wav")

subprocess.call(command, shell=True)

