import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load the audio file
sample_rate, data = wav.read('InputAudio.wav')

# Extract the audio channel if needed
if len(data.shape) > 1:
    data = data[:, 0]  # Select the first channel

# Define the cutoff frequency and order of the low pass filter
cutoff_freq = 4000  # Adjust this as per your requirement
filter_order = 4    # Adjust this as per your requirement

# Create the low pass filter
b, a = signal.butter(filter_order, cutoff_freq / (sample_rate / 2), btype='low')

# Apply the filter to the audio data
filtered_data = signal.lfilter(b, a, data)

# Calculate the frequency spectrum of the filtered data
frequencies, spectrum = signal.periodogram(filtered_data, sample_rate)

# Calculate the discontinuities in the spectrum
discontinuities = np.abs(spectrum[1:] - spectrum[:-1])

# Calculate the timestamps for each frequency bin
timestamps = np.arange(len(discontinuities)) * (len(data) / len(discontinuities)) / sample_rate

# Print the timestamps and corresponding discontinuity values
# for timestamp, dis in zip(timestamps, discontinuities):
#     print(f'Timestamp: {timestamp:.2f}s, Discontinuity: {dis:.2f}')

# Plot the discontinuities
plt.plot(timestamps, discontinuities)
plt.xlabel('Timestamp (s)')
plt.ylabel('Discontinuity')
plt.title('Discontinuities over Time')
plt.show()