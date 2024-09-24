#%%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
# %%
N = 1000
fs = 1000.0  # frecuencia de muestreo (Hz)
ts = 1 / fs  # tiempo de muestreo
df = fs / N  # resolución espectral

realizations = 200

# %%
tt = np.linspace(0, (N - 1) * ts, N)
A = 2
sig_pow = (A**2) / 2

noise_pow = sig_pow / 10**(10/20)
n = np.random.normal(0,noise_pow)

omega_zero = N / 4

fr = np.random.uniform(low=-0.5, high=0.5, size=realizations)

omega_one = (omega_zero + fr) * df

# %%
x = np.array([A * np.sin(2 * np.pi * o1 * tt) + n for o1 in omega_one])
# %%
test = x[0]
# %%
len(test)
# %%
test.max()
# %%
