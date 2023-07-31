import os.path

import numpy as np
import cv2
import time
from skimage.restoration import denoise_wavelet
from skimage.metrics import structural_similarity
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt, argrelmax
import datetime
import pywt
# from readfile import read_video

# MOVIE_NAME = "The_Great_Gatsby_rgb"
# MOVIE_NAME = "Ready_Player_One_rgb"
# MOVIE_NAME = "The_Long_Dark_rgb"
# MOVIE_NAME = "The_Dark_Knight_rgb"

def get_error_image(frame, prev_frames, method=cv2.TM_SQDIFF, macroblock_size=(16,16), k = 31):
    # print(prev_frames.shape)
    # frame = cv2.blur(frame,ksize=(5,5))
    prev_frame = np.mean(prev_frames, axis=0).astype(np.uint8)
    # print(prev_frame.shape)
    height, width, num_channels = frame.shape
    h = macroblock_size[0]
    w = macroblock_size[1]
    new_height = int(h*round(height/h))
    new_width = int(w*round(width/w))
    # new_height = 144
    # new_width = 256
    frame = cv2.resize(frame, (new_width, new_height))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    # error_entropies = []
    error_frame = np.zeros((new_height, new_width))
    prev_frame = cv2.resize(prev_frame, (new_width, new_height))
    for i in range(new_height//h):
        for j in range(new_width//w):
            # print(prev_frame.shape, frame.shape)
            tlh = max(h*i - k, 0)
            tlw = max(w*j - k, 0) 
            brh = min(h*(i+1)+k, new_height)
            brw = min(w*(j+1)+k, new_width)
            # print(tlh, brh, tlw, brw)
            prev_sub_frame_gray = prev_frame_gray[tlh:brh,tlw:brw].copy()
            # prev_sub_frame = prev_frame[tlh:brh,tlw:brw,:].copy()
            # print(prev_sub_frame_gray.shape)
            block = frame_gray[h*i:h*(i+1), w*j:w*(j+1)].copy()
            # block = frame[h*i:h*(i+1), w*j:w*(j+1)].copy()
            res = cv2.matchTemplate(prev_sub_frame_gray, block, method)
            # res = cv2.matchTemplate(prev_sub_frame, block, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            top_left = (top_left[1], top_left[0])
            bottom_right = (top_left[0] + h, top_left[1] + w)
            prev_block = prev_sub_frame_gray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]].copy()
            # prev_block = prev_sub_frame[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]].copy()
            # print("Block = ", block)
            # print("Prev Block = ", prev_block)
            # print(block.shape, prev_block.shape)
            error_frame[h*i:h*(i+1), w*j:w*(j+1)] = block.astype(np.float64) - prev_block.astype(np.float64)
    error_frame = np.abs(error_frame).astype(np.uint8)
    error_entropy = shannon_entropy(error_frame)
    return error_entropy
    # error_entropies.append(error_entropy)

    # error_frame = cv2.cvtColor(error_frame, cv2.COLOR_GRAY2RGB)
    # error_frame = cv2.resize(error_frame, (width, height))
    # cv2.imshow("New Frame", error_frame)
    # cv2.waitKey(100)
    # return sum(error_entropies)/len(error_entropies)

def histogram_comp(frame, prev_frames, method=cv2.HISTCMP_INTERSECT):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCR_CB)
    prev_frames = [cv2.cvtColor(fr, cv2.COLOR_RGB2YCR_CB) for fr in prev_frames]
    channels = [1,2]
    hist1 = cv2.calcHist([frame],channels,None,[16]*len(channels),[0,256]*len(channels))
    hist1 = cv2.normalize(hist1,hist1).flatten()
    hist2 = cv2.calcHist(prev_frames,channels,None,[16]*len(channels),[0,256]*len(channels))
    hist2 = cv2.normalize(hist2,hist2).flatten()
    d = cv2.compareHist(hist1, hist2, method)
    return d


def find_subshots(interval, fps, thresh=0.3):
    # f = open(MOVIE_NAME+"_entropy.txt",'r')
    # input = f.read()
    # f.close()
    # input = input.strip()
    # vals = input.split('\n')
    # vals = [float(v) for v in vals]
    # vals = np.array(vals)

    # for i in range(len(peaks)-1):
    #     print(i)
    #     interval = vals[peaks[i]:peaks[i+1]]
    interval = [abs(interval[i] - interval[i - 1]) for i in range(1, len(interval))]
    interval.insert(0, 0)
    interval = np.ma.anom(interval)
    new_peaks, _ = find_peaks(interval, height=thresh, distance=fps)
    new_peaks = [x for x in new_peaks if x >= fps and x <= len(interval) - fps]
    # print(new_peaks)
    # if len(new_peaks) > 0:
    #     plt.plot(interval)
    #     plt.plot(new_peaks, interval[new_peaks], "x")
    #     plt.show()
    return new_peaks

def generate_video(MOVIE_NAME, video, fps):
    filepath = os.path.join(MOVIE_NAME, MOVIE_NAME+'_entropy.txt')
    if os.path.exists(filepath):
        print("Reading from existing file")
        with open(filepath,'r') as f:
            data = f.read()
            data = data.rstrip()
            data = data.split('\n')
            data=[float(x) for x in data]
            return data
    print(video.shape, fps)
    num_frames, width, height, _ = video.shape
    buffer_size = 10
    time_count = 0
    time_step = 1000/fps
    frame_idx = -1
    # error_images = np.zeros((num_frames, height, width, 3))
    # start_time = time.time()
    # for i in range(1, num_frames):
    #     frame = video[i,:,:,:].copy()
    #     prev_frame = video[i-1,:,:,:].copy()
    #     new_frame = get_error_image(frame, prev_frame)
    #     error_images[i,:,:,:] = new_frame.copy()
    #     end_time = time.time()
    #     out = "\rFrame "+str(i)+", Time: "+str(end_time-start_time)
    #     print(out, end="\r")
    # video = [denoise_wavelet(frame.astype(np.float64), method="BayesShrink",mode='soft', wavelet_levels=3, wavelet='coif5',rescale_sigma=True, channel_axis=2, convert2ycbcr=True).astype(np.uint8) for frame in video]
    error_entropies = []
    while True:
        if frame_idx%100 ==0:
            print(frame_idx)
        frame_idx += 1
        # print(frame_idx)/
        if frame_idx >= num_frames:
            break
        time_count += time_step
        frame = video[frame_idx,:,:,:]
        # new_frame = error_images[frame_idx,:,:,:]
        # # print(frame.shape)
        # new_frame = np.zeros(frame.shape).astype(np.uint8)
        
        if frame_idx < buffer_size//2:
            prev_frames = np.zeros((buffer_size, width, height, 3)).astype(np.uint8)
            prev_frames[-frame_idx - buffer_size//2:,:,:,:] = video[:frame_idx+buffer_size//2,:,:,:].copy()
        elif frame_idx > len(video) - buffer_size//2:
            prev_frames = np.zeros((buffer_size, width, height, 3)).astype(np.uint8)
            prev_frames[:len(video)-1 - frame_idx + buffer_size//2,:,:,:] = video[-(len(video)-1 - frame_idx)-buffer_size//2:,:,:,:].copy()
        else:
            prev_frames = video[frame_idx-buffer_size//2:frame_idx+buffer_size//2,:,:,:].copy()
            
        # if frame_idx < buffer_size:
        #     prev_frames = np.zeros((buffer_size, width, height, 3)).astype(np.uint8)
        #     if frame_idx > 0:
        #         prev_frames[-frame_idx:,:,:,:] = video[:frame_idx,:,:,:].copy()
        # else:
        #     prev_frames = video[frame_idx-buffer_size:frame_idx,:,:,:].copy()
        
        # score = histogram_comp(frame, prev_frames)
        # score = orb_similarity(frame, prev_frames)
        score = get_error_image(frame, prev_frames)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCR_CB)
        # alpha = 0.33
        # score = 0
        # score += alpha*shannon_entropy(frame[:,:,0])
        # score += (0.5 - alpha/2)*shannon_entropy(frame[:,:,1])
        # score += (0.5 - alpha/2)*shannon_entropy(frame[:,:,2])
        # print(frame_idx, score)
        error_entropies.append(score)
        continue
            
        # frame = denoise_wavelet(frame.astype(np.float64), method="BayesShrink",mode='soft', wavelet_levels=3, wavelet='coif5',rescale_sigma=True, channel_axis=2, convert2ycbcr=True).astype(np.uint8)
        # prev_frames = [denoise_wavelet(img.astype(np.float64), method="BayesShrink",mode='soft', wavelet_levels=3, wavelet='coif5',rescale_sigma=True, channel_axis=2, convert2ycbcr=True).astype(np.uint8) for img in prev_frames]
        new_frame = get_error_image(frame, prev_frames)
        # new_frame = ssim_similarity(frame, prev_frame)
        entropy = shannon_entropy(new_frame)
        error_entropies.append(entropy)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
        # display_frame = np.concatenate((frame, new_frame), axis=1).astype(np.uint8)
        # cv2.imshow("Frame", display_frame)
        # # cv2.imshow("Mask Frame", new_frame)
        # # prev_frames.append(frame)
        # wait_time = int(time_count)
        # time_count -= wait_time
        # keyboard = cv2.waitKey(wait_time)
        # if keyboard == 'q' or keyboard == 27:
        #     break 
        # if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) <1 or cv2.getWindowProperty("Mask Frame", cv2.WND_PROP_VISIBLE) <1:
        #     break
    # cv2.destroyAllWindows()
    
    f = open(filepath,'w')
    for e in error_entropies:
        f.write(str(e)+'\n')
    f.close()
    return error_entropies
    
def get_discontinuities(MOVIE_NAME, error_entropies, fps):
    filepath = os.path.join(MOVIE_NAME, MOVIE_NAME+"_peaks.txt")
    vals = np.array(error_entropies)
    ma_len = 7
    vals_ma = np.convolve(vals, np.ones(ma_len)) / ma_len 
    vals_ma = vals_ma[:len(vals)]
    old_vals = vals
    old_wavelets_cA, old_wavelets_cD = pywt.dwt(old_vals, 'db8', mode='zero')
    vals = [abs(vals[i]-vals[i-1]) for i in range(1, len(vals))]
    vals.insert(0,0)
    vals = np.ma.anom(vals)
    peaks, _ = find_peaks(vals, height=0.22, distance=2*fps, prominence=0.2)
    # cwt_peaks = find_peaks_cwt(vals, widths=15)
    # # cwt_peaks1, _ = find_peaks(vals, height=(0,0.2), distance=240)
    # cwt_peaks = argrelmax(vals, order=150)[0]
    # cwt_peaks = [x for x in cwt_peaks if x not in peaks]
    # # cwt_peaks = cwt_peaks1
    # print(cwt_peaks)

    # with open(MOVIE_NAME+"_cwt_peaks.txt",'w') as f:
    #     for p in cwt_peaks:
    #         p = round(float(p/30),3)
    #         timestamp = '0'+str(datetime.timedelta(seconds=p))
    #         f.write(timestamp+'\n')
    peaks = list(peaks)
    if peaks[0] < 2*fps:
        peaks[0] = 0
    else:
        peaks.insert(0,0)
    # if peaks[-1] > num_frames-1 - 60:
    #     peaks[-1] = num_frames-1
    # else:
    #     peaks.append(num_frames-1)
            
    with open(filepath,'w') as f:
        for p in peaks:
            p = float(p/fps)
            timestamp = '0'+str(datetime.timedelta(seconds=p))
            f.write(timestamp+'\n')
            
    # print(len(vals), len(vals_ma), len(peaks), len(cwt_peaks))
    # plt.figure(figsize=(15,5))
    # plt.plot(vals)
    # # plt.plot(old_vals)
    # # plt.plot([2*i for i in range(len(old_wavelets_cD))], [abs(x) for x in old_wavelets_cD])
    # plt.plot(peaks, vals[peaks], "x")
    # plt.plot(cwt_peaks, vals[cwt_peaks], "x")
    # plt.show()
    return peaks

def get_color_entropy(frame, alpha=0.5):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCR_CB)
    score = 0
    score += alpha*shannon_entropy(frame[:,:,0])
    score += (0.5 - alpha/2)*shannon_entropy(frame[:,:,1])
    score += (0.5 - alpha/2)*shannon_entropy(frame[:,:,2])
    return score


def convert_to_scenes_and_shots(video, entropies, peaks, fps, delta=0.1):
    scene_num = 0
    # f = open(MOVIE_NAME+"_scene_shot.txt",'r')
    # f.write(str(scene_num)+" "+str(shot_num)+" "+str(datetime.timedelta(seconds=0))+"\n")
    y = []
    x = []
    d = {}
    d[0] = {}
    interval = entropies[peaks[0]:peaks[1]]
    new_peaks = find_subshots(interval, fps)
    new_peaks = [peaks[0] + x for x in new_peaks]
    if len(new_peaks) > 0:
        d[0][0] = new_peaks
    for i in range(1, len(peaks) - 1):
        l_idx = peaks[i] - int(delta * (peaks[i] - peaks[i - 1]))
        r_idx = peaks[i] + int(delta * (peaks[i + 1] - peaks[i]))
        left_frame = video[l_idx, :, :, :].copy()
        right_frame = video[r_idx, :, :, :].copy()
        left_entropy = get_color_entropy(left_frame)
        right_entropy = get_color_entropy(right_frame)
        diff = abs(right_entropy - left_entropy)
        diff = histogram_comp(right_frame, [left_frame])
        if diff >= 1:
            d[scene_num][peaks[i]] = []
            interval = entropies[peaks[i]:peaks[i + 1]]
            new_peaks = find_subshots(interval, fps)
            new_peaks = [peaks[i] + x for x in new_peaks]
            d[scene_num][peaks[i]] = new_peaks
        else:
            scene_num = peaks[i]
            d[scene_num] = {}
            interval = entropies[peaks[i]:peaks[i + 1]]
            new_peaks = find_subshots(interval, fps)
            new_peaks = [peaks[i] + x for x in new_peaks]
            if len(new_peaks) > 0:
                d[scene_num][peaks[i]] = new_peaks
        x.append(peaks[i])
        y.append(diff)
        # print(str(datetime.timedelta(seconds=peaks[i]/fps)), peaks[i], diff)
    # plt.stem(x,y)
    # plt.show()

    return d
    
    
if __name__=="__main__":
    pass
    # video, fps = read_video(MOVIE_NAME)
    # entropies = display_video(video, fps)
    # peaks = get_discontinuities(MOVIE_NAME, fps)
    # d = convert_to_scenes_and_shots(video, peaks)
    # print(d)