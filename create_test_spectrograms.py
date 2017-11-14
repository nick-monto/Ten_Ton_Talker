import os
import math
import fnmatch
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from my_spectrogram import my_specgram
from collections import OrderedDict
from scipy.io import wavfile
from pylab import rcParams


rcParams['figure.figsize'] = 6, 3

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = 'Input_audio_wav_16k/'
OUTPUT_FOLDER = 'Input_spectrogram_16k/'
talkers = os.listdir(INPUT_FOLDER)
talkers.sort()

audio_dict = OrderedDict()

for l in talkers:
    audio_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))


# same as training but no added noise
def plot_spectrogram_test(audiopath, plotpath=None, NFFT_window=0.025,
                          noverlap_window=0.023, freq_min=None, freq_max=None,
                          axis='off'):
    fs, data = wavfile.read(audiopath)
    data = data / data.max()
    NFFT = pow(2, int(math.log(int(fs*NFFT_window), 2) + 0.5))  # 25ms window, nearest power of 2
    noverlap = int(fs*noverlap_window)
    fc = int(np.sqrt(freq_min*freq_max))
    # Pxx is the segments x freqs array of instantaneous power, freqs is
    # the frequency vector, bins are the centers of the time bins in which
    # the power is computed, and im is the matplotlib.image.AxesImage
    # instance
    Pxx, freqs, bins, im = my_specgram(data, NFFT=NFFT, Fs=fs,
                                       Fc=fc, detrend=None,
                                       window=np.hanning(NFFT),
                                       noverlap=noverlap, cmap='Greys',
                                       xextent=None,
                                       pad_to=None, sides='default',
                                       scale_by_freq=None,
                                       minfreq=freq_min, maxfreq=freq_max)
    plt.axis(axis)
    im.axes.axis('tight')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    if plotpath:
        plt.savefig(plotpath, bbox_inches='tight',
                    transparent=False, pad_inches=0, dpi=96)
    else:
        plt.show()
    plt.clf()


# create spectrograms of randomly drawn samples from each language
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result[0]


df_test = pd.read_table('test_set_16k.txt',
                        names=['stimulus'])


if not os.path.exists(OUTPUT_FOLDER + 'Test'):
    os.makedirs(OUTPUT_FOLDER + 'Test')
    print('Successfully created a Test folder!')
print('Populating Test folder with spectrograms...')
for j in range(0, len(df_test['stimulus'])):
    for k in range(0, 1):
        plot_spectrogram_test(find(df_test['stimulus'][j], INPUT_FOLDER),
                              plotpath=OUTPUT_FOLDER + 'Test/' +
                                       str(df_test['stimulus'][j][:-4]) + '_' +
                                       str(k) + '.jpeg',
                              NFFT_window=0.025, noverlap_window=0.023,
                              freq_min=0, freq_max=5500)
        print('Done with {}.'.format(df_test['stimulus'][j][:-14]))
