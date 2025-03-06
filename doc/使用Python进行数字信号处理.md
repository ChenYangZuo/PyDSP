---
typora-root-url: ../resource
---

# 使用Python进行数字信号处理

可能用到的库：

- NumPy
- SciPy
- Matplotlib

## 1-傅里叶变换

NumPy中提供了快速傅里叶变换（FFT）函数，其是一种高效计算离散傅里叶变换（DFT）及其逆变换（IDFT）的算法。

标准的FFT可以使用`numpy.fft.fft()`调用，其函数原型为：

```python
fft.fft(a, n=None, axis=-1, norm=None, out=None)
```

`a`为要分析的数据，以数组形式提供。函数返回一个ndarray。

获取FFT的采样范围可调用`numpy.fft.fftfreq()`，其原型为：

```python
fft.fftfreq(n, d=1.0)
```

其中`n`为信号长度，常设为`len(a)`；`d`表示相邻采样点的采样间隔即采样周期，通常为`1/fs`。函数返回一个频率轴，其前半部分为正频率，中点为负频率起点，后半部分为负频率。

对于一个实信号，FFT的结果是对称的，因此可以只取前半部分进行绘图：

```python
signal = cos1 + cos2 + noise
fft_result = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 1/fs)
fft_magnitude = np.abs(fft_result[:N//2]) * 2 / N
freqs = freqs[:N//2]

plt.figure()
plt.plot(freqs, fft_magnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()
```

<img src="fft_test1.png" alt="fft1" style="zoom:33%;" />

对于调制信号而言，信号可能为复信号，则需要绘制完整的频谱，可以使用`numpy.fft.fftshift()`函数重新排列结果，其原型为：

```python
fft.fftshift(x, axes=None)
```

其中`x`为待重排序的信号。函数返回一个ndarray。

对全部频谱绘图，则可使用：

```python
signal = i + 1j * q
fft_result = np.fft.fft(signal)
fft_result_shifted = np.abs(np.fft.fftshift(fft_result))
freqs_shifted = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))

plt.figure()
plt.plot(freqs_shifted, fft_result_shifted)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()
```

<img src="fft_test2.png" alt="fft2" style="zoom:33%;" />

在FFT后，得到的是一个复数数组，其中每个点表示一个特定频率分量的振幅（Magnitude）和相位（Phase）。如果绘制FFT频谱，Y轴的含义可以为：

**幅度谱：**

对于`fft_result`，取`np.abs(fft_result)`可得其幅度。在实际信号处理中，幅度通常需要归一化，即`np.abs(fft_result) * 2 / N`，Y轴单位与原始数据单位相同。

<img src="fft_test1.png" alt="fft1" style="zoom:33%;" />

**功率谱：**

对于`fft_result`，取`20 * np.log10(np.abs(fft_result))`可得其功率谱，Y轴单位为分贝（dB）。

<img src="fft_test3.png" alt="fft3" style="zoom:33%;" />

**相位谱：**

对于`fft_result`，取`np.angle(fft_result)`得其相位谱，Y轴单位为弧度（radians）。

对于功率谱分析，还可使用`scipy.signal.periodogram()`函数进行分析，其原型为：

```python
f_periodo, Pxx = periodogram(x, fs)
```

其中`x`为输入信号，`fs`为采样率，`f_periodo`为返回的频率数组，`Pxx`为对应的功率谱密度，且已自动归一化。

具体而言，可使用：

```python
signal = cos1 + cos2 + noise
f_periodo, Pxx = periodogram(signal, fs)

plt.semilogy(f_periodo, Pxx, label="Periodogram PSD")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (V²/Hz)")
plt.grid()
plt.show()
```

其中`plt.semilogy`用于绘制半对数坐标图的函数，其X轴为线性刻度，Y轴为对数刻度。

<img src="fft_test4.png" alt="fft4" style="zoom:33%;" />

## 2-时频变换

短时傅里叶变换（Short-Time Fourier Transform，STFT）是一种将信号分解为时域和频域信息的时频分析方法。它通过将信号分成短时段，并在每个短时段上应用傅里叶变换来捕捉信号的瞬时频率。即采用中心位于时间$\alpha$的时间窗$g(t-\alpha)$在时域信号上滑动，在时间窗$g(t-\alpha)$限定的范围内进行傅里叶变换，这样就使短时傅里叶变换具有了时间和频率的局部化能力，兼顾了时间和频率的分析。

SciPy中提供了STFT函数`scipy.signal.stft()`，可以分析信号的时频特征。其原型为：

```python
signal.stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1, scaling='spectrum')
```

其中：

- x：输入数据，数组形式
- fs：信号采样率
- window：要使用的所需窗函数
- nperseg：每个段的长度，增大会提升频谱分辨率、降低时间分辨率
- noverlap：段之间重叠的点数，增大会平滑时频谱，减少窗口跳变带来的误差，但计算量增加，常用`noverlap=nperseg // 2`
- nfft：默认为None，FFT长度为`nperseg`

函数返回：

- f：频率数组（y轴）
- t：时间数组（x轴）
- Zxx：STFT计算的复数矩阵，`spectrum[f, t]` 代表`t`时间点、`f`频率点的傅里叶变换结果。

目前该函数已处于停止更新状态，为此SciPy中提供了新的STFT类`scipy.signal.ShortTimeFFT`，可以分析信号的时频特征。其原型为：

```python
class ShortTimeFFT(win, hop, fs, *, fft_mode='onesided', mfft=None, dual_win=None, scale_to=None, phase_shift=0)
```

下文仍使用旧版函数进行实验。

为绘制信号的时频瀑布图，可使用：

```python
x = np.concatenate((signal1, signal2, signal3))
f, t, spectrum = signal.stft(x, fs, nperseg=256, noverlap=128)

plt.pcolormesh(t, f, np.abs(spectrum), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
```

其中`plt.pcolormesh()`用于绘制时频图，展示 STFT 结果，其原型为：

```python
matplotlib.pyplot.pcolormesh(*args, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, colorizer=None, shading=None, antialiased=False, data=None, **kwargs)
```

其中：

- X：x轴坐标
- Y：y轴坐标
- C：颜色数据
- cmap：颜色映射方案
- norm：颜色归一化方式
- shading：插值方式，默认`'flat'`为方块色块图，边界清晰；`'gouraud'`为平滑插值过渡

此处使用`t`作为x轴，`f`作为y轴，`np.abs(spectrum)`STFT得到的幅值作为数据。

<img src="stft_test1.png" alt="stft1" style="zoom:33%;" />
