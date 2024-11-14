# %%
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from itertools import accumulate


# %%
sio.whosmat("ECG_TP4.mat")
mat_struct = sio.loadmat("ECG_TP4.mat")

ecg_lead = mat_struct["ecg_lead"]
qrs_pattern1 = mat_struct["qrs_pattern1"]
heartbeat_pattern1 = mat_struct["heartbeat_pattern1"]
heartbeat_pattern2 = mat_struct["heartbeat_pattern2"]
qrs_detections = mat_struct["qrs_detections"]


# %%
# Parameters
hb_length = 600
half_length = int(hb_length // 2)

heartbeats = [
    ecg_lead[int(q[0] - half_length) : int(q[0] + half_length)] for q in qrs_detections
]
heartbeats_dt = [sig.detrend(h) for h in heartbeats]


# %%


# %%
plt.figure(figsize=(10, 6))

for slice_segment in heartbeats:
    plt.plot(slice_segment, alpha=0.5)
plt.title("Overlapped ECG Slices Centered on QRS Detection Points")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()

# %%
plt.figure(figsize=(10, 6))

for slice_segment in heartbeats_dt:
    plt.plot(slice_segment, alpha=0.5)
plt.title("Overlapped ECG Slices Centered on QRS Detection Points")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()

# %%


# %%
plt.plot(heartbeats_dt[10])


# %%
