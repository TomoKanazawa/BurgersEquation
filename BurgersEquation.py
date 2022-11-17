import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
xNum = 1000
tNum = 10000
period = 2*pi
dt = 0.0002
v = 0.01

energy = np.empty(xNum)

k = np.ndarray(xNum)
for i in range(0, xNum):
    k[i] = i - (i//(xNum/2))*xNum

# Discrete Fourier Transformation of sin(x)
x = np.linspace(0, period, xNum, endpoint=False)
discreteF = np.sin(x)
fftF = np.fft.fft(discreteF) / xNum

# Time Development of FFT(sin)
for i in range(1, tNum):
    a = np.fft.ifft(1j*k*fftF) * xNum
    nonLinear = np.fft.fft( a * discreteF) / xNum
    fftF = (1 - v*k*k*dt) * fftF - nonLinear * dt
    discreteF = np.real(np.fft.ifft(fftF) * xNum)
energy = abs(fftF)**2 / 2

plt.xscale("log")
plt.yscale("log")
plt.plot(k[0:49], energy[0:49])
plt.plot(k[0:49], np.exp(-4)*k[0:49]**(-2))
plt.show()

plt.cla()
plt.plot(x, discreteF)
plt.show()