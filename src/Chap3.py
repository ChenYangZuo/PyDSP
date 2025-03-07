import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def fir_test1():
    fs = 1000  # 采样率
    cutoff = 100  # 截止频率 (Hz)
    numtaps = 51  # 滤波器阶数 +1
    window_type = 'hamming'  # 窗函数

    # 设计滤波器
    fir_coeff = signal.firwin(numtaps, cutoff, fs=fs, window=window_type)
    # 画出频率响应
    w, h = signal.freqz(fir_coeff, worN=8000)
    
    plt.plot(w * fs / (2 * np.pi), np.abs(h))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    plt.grid()
    plt.savefig("resource/fir_test1.png", dpi=300)
    plt.show()


def fir_test2():
    fs = 1000  # 采样率
    cutoff = 100  # 截止频率 (Hz)
    numtaps = 51  # 滤波器阶数 +1
    window_type = 'hamming'  # 窗函数
    T = 1
    N = T * fs
    t = np.linspace(0, T, N)
    
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.5 * np.random.randn(N)
    fir_coeff = signal.firwin(numtaps, cutoff, fs=fs, window=window_type)
    
    fft_result = np.fft.fftshift(np.abs(np.fft.fft(x))*2/N)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    plt.plot(freqs, fft_result, label='Original Signal')
    
    y = signal.lfilter(fir_coeff, 1, x)
    fft_result = np.fft.fftshift(np.abs(np.fft.fft(y))*2/N)
    plt.plot(freqs, fft_result, label='Filtered Signal')
    
    z = np.convolve(x, fir_coeff, mode='same')
    fft_result = np.fft.fftshift(np.abs(np.fft.fft(z))*2/N)
    plt.plot(freqs, fft_result, label='Convolved Signal')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig('resource/fir_test2.png', dpi=300)
    plt.show()
    

if __name__ == '__main__':
    fir_test2()