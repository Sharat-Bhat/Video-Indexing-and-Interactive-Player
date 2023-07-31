import time
import tkinter as tk
from collections import defaultdict
from tkinter import ttk
from PIL import ImageTk
from PIL import Image
from tkVideoPlayer import TkinterVideo
import cv2


class VideoPlayer:
    def __init__(self, filename, scenes):
        self.root = tk.Tk()
        self.fileName = filename
        self.root.title("Video Player")
        self.vid_player = TkinterVideo(scaled=False, master=self.root)
        self.vid_player.pack(expand=True, fill="both")
        self.play_pause_btn = tk.Button(self.root, text="Play", command=self.play_pause)
        self.play_pause_btn.pack()
        self.treeview = ttk.Treeview(self.root)
        self.treeview.pack()
        count = 0
        self.tempTs = defaultdict()
        # for i in range(len(frames)):
        #     f = frames[i]
        #     self.treeview.insert('', 'end', str(count), text='Scene' + str(count))
        #     self.tempTs[str(count)] = round(int(f) * 1000 / 30)
        #     count += 1
        self.treeview.insert('', 'end', '1',
                             text='Scene 1')
        self.treeview.insert('', 'end', '2',
                             text='Scene 2')

        self.treeview.insert('1', 'end', 'sh2',
                             text='Shot 2')
        self.treeview.insert('2', 'end', 'sh3',
                             text='Shot 3')
        self.treeview.insert('sh2', 'end', 'ssh1',
                             text='SubShot 1')
        self.treeview.insert('sh2', 'end', 'ssh2',
                             text='SubShot 2')

        self.load_video()
        self.root.geometry("1024x600")
        self.treeview.bind('<ButtonRelease-1>', self.selectItem)
        self.tempTs = {
            '1': 30033,
            '2': 35867,
            'sh2': 40000,
            'sh3': 41000,
            'ssh1': 50000,
            'ssh2': 60000
        }
        print(self.tempTs)
        self.root.mainloop()

    def selectItem(self, a):
        curItem = self.treeview.focus()
        print(curItem)
        # key = self.treeview.item(curItem)
        # print(key)
        ts = self.tempTs[curItem]
        self.render_to_timestamp(ts)

    def load_video(self):
        if self.fileName:
            self.vid_player.load(self.fileName)
            self.play_pause_btn["text"] = "Play"

    def play_pause(self):
        if self.vid_player.is_paused():
            self.vid_player.play()
            self.play_pause_btn["text"] = "Pause"
        else:
            self.vid_player.pause()
            self.play_pause_btn["text"] = "Play"

    def render_to_timestamp(self, ts):
        print(ts)
        self.vid_player.seek(ts)
        if self.vid_player.is_paused():
            self.vid_player.play()
            time.sleep(0.5)
            self.vid_player.pause()
