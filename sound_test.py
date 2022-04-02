import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fftpack import fft
from libraries.detect_peaks import detect_peaks
from libraries.peaks_nn import PeakClassifier
from libraries.post_to_server import send_post_on, send_post_off
from audio_settings import *


def wait_for_level(min_level, max_level, chunk):
  for i in range(10000):
    in_data = stream.read(chunk)
    np_data = np.frombuffer(in_data, dtype=np.int16)
    sound_level = np.max(np_data)
    if (sound_level >= min_level) & (sound_level <= max_level):
      break
  return np_data



# instantiate nn and train
pclass = PeakClassifier(SOUND_DATA_FILE)  # type: PeakClassifier
pclass.train()

fig, plot_list = plt.subplots(PLOT_COUNT, figsize=(15, 7))

# Shows wave plot at top and then PLOT_COUNT-1 frequency plots below it
# captures wave starting when sound level > min.

# setup wave plot...
if 1:
    # format waveform axes
    plot_list[0].set_xlabel('time')
    plot_list[0].set_ylabel('volume')
    plot_list[0].set_ylim(0, 2048)
    plot_list[0].set_xlim(0, CHUNK)
    plt.setp(
        plot_list[0], yticks=[0, 2048],
        xticks=[0, CHUNK],
    )
    # create a line object with random data
    x = np.arange(0, 2 * CHUNK, 2)
    line0, = plot_list[0].plot(x, np.random.rand(CHUNK), '-', lw=2)


# setup frequency plot...
if 1:
    line_ffts = []
    for j in range(1, PLOT_COUNT):
        xf = np.linspace(0, RATE, CHUNK)
        line_fft, = plot_list[j].plot( xf, np.random.rand(CHUNK), '-', lw=2)
        line_ffts.append(line_fft)

        plot_list[j].set_xlabel('frequency')
        plot_list[j].set_ylabel('volume')

        plt.setp( plot_list[j], yticks=[0, 1.25], )

        # format spectrum axes
        plot_list[j].set_xlim(left=LOW_FREQUENCY, right=HIGH_FREQUENCY)


# Setup Stream Object
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# capture up to "PLOT_COUNT" plots.
# waits for quiet and then captures first sound for SAMPLE_LENGTH
samples = []

while True:

    # Wait for sound...
    print("Waiting For Quiet")
    wait_for_level(0, SOUND_OFF_LEVEL, CHUNK)

    print("Waiting For Sound")
    data_pre = wait_for_level(SOUND_ON_LEVEL, 9999, CHUNK1)

    # Now take rest of sample
    in_data = stream.read(CHUNK2)
    stream.start_stream()

    data_post = np.fromstring(in_data, dtype=np.int16)
    # append 2 samples
    data = np.concatenate((data_pre, data_post))

    # compute FFT and update line
    yf = fft(data)
    d1 = np.abs(yf)
    d1 = d1 / np.max(d1)

    # detect peaks in frequency graph
    d2 = d1[LOW_FREQUENCY:HIGH_FREQUENCY]
    #d2 = np.where(d2 > 0.1, d2, 0)
    peak_indices = detect_peaks(d2, mpd= MIN_PEAK_DIST,  show=False)

    # fill in any peaks by subdividing the audio spectrum.  We expect
    # a peak in each of the "slices"
    mx = MIN_PEAK_DIST
    peak_index_out = []
    for peak_index in peak_indices:
        while peak_index > mx:
            p = int(mx - MIN_PEAK_DIST / 2)
            peak_index_out.append(p)
            d2[p] = 0
            mx += MIN_PEAK_DIST
        peak_index_out.append(peak_index)
        mx += MIN_PEAK_DIST

    ind_out = [float(p) / HIGH_FREQUENCY for p in peak_index_out]
    peaks = [math.floor(d2[i] * 100.) / 100. for i in peak_index_out]

    #for i in range(len(peaks)):
    #    print(ind_out[i], peaks[i])

    out = ind_out + peaks
    print(len(out), out)

    if len(samples) == len(line_ffts):
        samples.pop(0)
    samples.append(d1)

    # render captured plots
    line0.set_ydata(data)
    for i in range(min(len(samples), len(line_ffts))):
        line_ffts[i].set_ydata(samples[i])

    # now predict what kind of sound it is.
    z = np.asarray([out])
    p = pclass.predict(z)
    if p == "glass":
        send_post_on()
    plt.pause(0.25)


