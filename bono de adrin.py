# %% [markdown]
# # Bono 

# %%
!pip install playsound


# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy.io.wavfile import read 
from scipy.io.wavfile import write
from scipy.signal import upfirdn
from playsound import playsound
import IPython.display as ipd
from scipy.signal import butter, filtfilt, freqz

# %%
audio_1 = "hola.wav"
fs1, data_1 = wavfile.read(audio_1)
print(f'Duracion = {data_1.shape[0]/fs1} , Frecuencia de Muestreo = {fs1} [=] Muestras/Seg' \
      f', Wav format = {data_1.dtype}')
ipd.Audio("hola.wav")

# %%
audio_2 = "yiruma.wav"
fs2, data_2 = wavfile.read(audio_2)
print(f'Duracion = {data_2.shape[0]/fs2} , Frecuencia de Muestreo = {fs2} [=] Muestras/Seg' \
      f', Wav format = {data_2.dtype}')
ipd.Audio("yiruma.wav")

# %%
#Upsampling

factor=2
data3=upfirdn([1],data_1,factor) 
fs3=fs1*factor

new_duration = len(data3)/fs3
new_time = np.arange(0,new_duration,1/fs3) 



# %%
write("example.wav", fs3,data3.astype(np.int16))
ipd.Audio("example.wav")

# %%
#suma de señales

sum_signal=data3[0:79000]+data_2[0:79000]
write("suma.wav", fs3, sum_signal.astype(np.int16))
ipd.Audio("suma.wav")

# %%
#filtrado 
sampling_frequency = fs3
b, a = butter(10,1700, btype='lowpass',fs =fs3)
filtered = filtfilt(b,a,sum_signal)
write("filtered.wav", fs3, filtered.astype(np.int16))
ipd.Audio("filtered.wav")
