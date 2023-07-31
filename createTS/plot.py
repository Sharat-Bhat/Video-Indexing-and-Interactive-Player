import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt
import datetime

MOVIE_FILE = "The_Great_Gatsby_rgb"
# MOVIE_FILE = "Ready_Player_One_rgb"
# MOVIE_FILE = "The_Long_Dark_rgb"


def calculateTS(data):
    # f = open(MOVIE_FILE+"_entropy.txt",'r')
    # input = data
    # f.close()
    # input = input.strip()
    # vals = input.split('\n')
    vals = data
    vals = np.array(vals)
    # ma_len = 7
    # vals_ma = np.convolve(vals, np.ones(ma_len)) / ma_len
    # vals_ma = vals_ma[:len(vals)]
    vals = [abs(vals[i]-vals[i-1]) for i in range(1, len(vals))]
    vals.insert(0,0)
    vals = np.ma.anom(vals)
    peaks, _ = find_peaks(vals, distance=60, prominence=0.2)
    # cwt_peaks = find_peaks_cwt(vals, widths=15)


    # with open(MOVIE_FILE+"_cwt_peaks.txt",'w') as f:
    #     for p in cwt_peaks:
    #         p = round(float(p/30),3)
    #         timestamp = '0'+str(datetime.timedelta(seconds=p))
    #         f.write(timestamp+'\n')

    with open(MOVIE_FILE+"_peaks.txt",'w') as f:
        for p in peaks:
            # p = round(float(p/30),3)
            # timestamp = '0'+str(datetime.timedelta(seconds=p))
            f.write(str(p)+'\n')

    # print(len(vals), len(vals_ma), len(peaks), len(cwt_peaks))
    # plt.plot(vals)
    # plt.plot(vals)
    # plt.plot(peaks, vals[peaks], "x")
    # plt.plot(cwt_peaks, vals[cwt_peaks], "o")
    # plt.show()
    return peaks
