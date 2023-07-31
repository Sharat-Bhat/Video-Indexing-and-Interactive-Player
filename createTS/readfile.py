import os
import numpy as np 

HOME_DIRECTORY = ""



def read_video(movie_directory):
    readme_file_path = os.path.join(HOME_DIRECTORY, movie_directory, "readme.txt")
    f_rme = open(readme_file_path, 'r')
    lines = f_rme.read()
    lines = lines.strip()
    lines = lines.split("\n")
    fps = int(lines[0].split(':')[1])
    # print("fps", fps)
    resolution = lines[1].split(':')[1].split('x')
    width = int(resolution[0])
    height = int(resolution[1])
    print("resolution", resolution)
    num_frames = int(lines[2].split(':')[1])
    print("num frames", num_frames)
    rgb_file_path = os.path.join(HOME_DIRECTORY, movie_directory, "InputVideo.rgb")
    f_rgb = open(rgb_file_path, 'rb')
    rgb = f_rgb.read()
    assert 3*width*height*num_frames == len(rgb)
    rgb = np.frombuffer(rgb, dtype=np.uint8)
    rgb = np.reshape(rgb, (num_frames, height, width, 3))
    print(rgb.shape)
    print(rgb.dtype)
    return rgb, fps
    
    

if __name__=="__main__":
    read_video("The_Long_Dark_rgb")