import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def fft_test1():
    fs = 1000
    T = 1
    N = T * fs
    t = np.linspace(0, T, N)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.5 * np.random.randn(N)
    fft_result = np.fft.fft(x)
    fft_magnitude = np.abs(fft_result[:N//2]) * 2 / N
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]

    plt.plot(freqs, fft_magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.savefig('resource/fft_test1.png', dpi=300)
    plt.show()

def fft_test2():
    fs = 1000
    T = 1
    N = T * fs
    t = np.linspace(0, T, N)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.5 * np.random.randn(N)
    fft_result = np.fft.fftshift(np.abs(np.fft.fft(x))*2/N)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    
    plt.plot(freqs, fft_result)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.savefig('resource/fft_test2.png', dpi=300)
    plt.show()

def fft_test3():
    fs = 1000
    T = 1
    N = T * fs
    t = np.linspace(0, T, N)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.5 * np.random.randn(N)
    fft_result = np.fft.fft(x)
    fft_power = 20 * np.log10(np.abs(fft_result[:N//2]))
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]

    plt.plot(freqs, fft_power)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.savefig('resource/fft_test3.png', dpi=300)
    plt.show()

def fft_test4():
    fs = 1000
    T = 1
    N = T * fs
    t = np.linspace(0, T, N)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.5 * np.random.randn(N)
    f_periodo, Pxx = signal.periodogram(x, fs)
    plt.semilogy(f_periodo, Pxx)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.savefig('resource/fft_test4.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    pass