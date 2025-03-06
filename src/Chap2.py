import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def stft_test1():
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)

    signal1 = np.sin(2 * np.pi * 50 * t)
    signal2 = np.sin(2 * np.pi * 120 * t)
    signal3 = np.sin(2 * np.pi * 300 * t)
    x = np.concatenate((signal1, signal2, signal3))

    f, t, spectrum = signal.stft(x, fs, nperseg=256, noverlap=128)

    plt.pcolormesh(t, f, np.abs(spectrum), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.savefig('resource/stft_test1.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    stft_test1()