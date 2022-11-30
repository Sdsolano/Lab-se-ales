def spect(y,fs):
  fig = plt.specgram(y, NFFT=5000, Fs=fs, noverlap=512,cmap='jet_r')
  return fig
