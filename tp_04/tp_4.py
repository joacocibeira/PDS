#######################################################################################################################
# Configuración e inicio de la simulación
#######################################################################################################################
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Datos generales de la simulación
fs = 1000.0  # frecuencia de muestreo (Hz)
N = 1000  # cantidad de muestras

# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 4
N_os = N * over_sampling

# Datos del ADC
B = 4  # bits
Vf = 2  # Volts
q = Vf / 2**B  # Volts

# datos del ruido
kn = 10
pot_ruido = q**2 / 12 * kn  # Watts (potencia de la señal 1 W)

ts = 1 / fs  # tiempo de muestreo
df = fs / N  # resolución espectral

#######################################################################################################################
# Acá arranca la simulación
# %%
tt = np.linspace(0, (N - 1) * ts, N)
tt_os = np.linspace(0, (N - 1) * ts, N_os)

ff = np.linspace(0, (N - 1) * df, N)
ff_os = np.linspace(0, (N_os - 1) * df, N_os)

analog_sig = np.sin(2 * np.pi * 10 * df * tt_os)
analog_sig = analog_sig / np.sqrt(np.var(analog_sig))

nn = np.random.normal(0, np.sqrt(pot_ruido), size=N_os)

sr = analog_sig + nn

srq = q * np.round(sr / q)

nq = srq - sr


ft_SR = 1 / N_os * np.fft.fft(sr, axis=0)
ft_Srq = 1 / N_os * np.fft.fft(srq, axis=0)
ft_As = 1 / N_os * np.fft.fft(analog_sig, axis=0)
ft_Nq = 1 / N_os * np.fft.fft(nq, axis=0)
ft_Nn = 1 / N_os * np.fft.fft(nn, axis=0)
bfrec = ff <= fs / 2

Nnq_mean = np.mean(np.abs(ft_Nq))
nNn_mean = np.mean(np.abs(ft_Nn))

#######################################################################################################################
# %% Presentación gráfica de los resultados
plt.close("all")

plt.figure(1)
plt.plot(tt_os, srq, lw=2, label="$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)")
plt.plot(
    tt_os,
    sr,
    linestyle=":",
    color="green",
    marker="o",
    markersize=3,
    markerfacecolor="none",
    markeredgecolor="green",
    fillstyle="none",
    label="$ s_R = s + n $  (ADC in)",
)
plt.plot(tt_os, analog_sig, color="orange", ls="dotted", label="$ s $ (analog)")

plt.title(
    "Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V".format(
        B, Vf, q
    )
)
plt.xlabel("tiempo [segundos]")
plt.ylabel("Amplitud [V]")
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


plt.figure(2)
bfrec = ff <= fs / 2

Nnq_mean = np.mean(np.abs(ft_Nq) ** 2)

plt.plot(
    ff[bfrec],
    10 * np.log10(2 * np.abs(ft_Srq[ff_os <= fs / 2]) ** 2),
    lw=2,
    label="$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)",
)
plt.plot(
    ff_os[ff_os <= fs / 2],
    10 * np.log10(2 * np.abs(ft_As[ff_os <= fs / 2]) ** 2),
    color="orange",
    ls="dotted",
    label="$ s $ (analog)",
)
# plt.plot(
#     ff[bfrec],
#     10 * np.log10(2 * np.abs(ft_SR[ff_os <= fs / 2]) ** 2),
#     ":g",
#     label="$ s_R = s + n $  (ADC in)",
# )
# plt.plot(
#     ff_os[ff_os <= fs / 2], 10 * np.log10(2 * np.abs(ft_Nn[ff_os <= fs / 2]) ** 2), ":r"
# )
# plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_Nq[ff_os <= fs / 2]) ** 2), ":c")
# plt.plot(
#     np.array([ff[bfrec][0], ff[bfrec][-1]]),
#     10 * np.log10(2 * np.array([nNn_mean, nNn_mean])),
#     "--r",
#     label="$ \overline{n} = $"
#     + "{:3.1f} dB (piso analog.)".format(10 * np.log10(2 * nNn_mean)),
# )
# plt.plot(
#     np.array([ff[bfrec][0], ff[bfrec][-1]]),
#     10 * np.log10(2 * np.array([Nnq_mean, Nnq_mean])),
#     "--c",
#     label="$ \overline{n_Q} = $"
#     + "{:3.1f} dB (piso digital)".format(10 * np.log10(2 * Nnq_mean)),
# )
plt.title(
    "Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V".format(
        B, Vf, q
    )
)
plt.ylabel("Densidad de Potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
axes_hdl = plt.gca()
axes_hdl.legend()
# suponiendo valores negativos de potencia ruido en dB
plt.ylim((1.5 * np.min(10 * np.log10(2 * np.array([Nnq_mean, nNn_mean]))), 10))


plt.figure(3)
bins = 10
plt.hist(nq, bins=bins)
plt.plot(
    np.array([-q / 2, -q / 2, q / 2, q / 2]),
    np.array([0, N / bins, N / bins, 0]),
    "--r",
)
plt.title(
    "Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V".format(
        B, Vf, q
    )
)

# %%
