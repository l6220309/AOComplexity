import mne
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ch_names=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

ch_names=['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

biosemi_montage = mne.channels.read_custom_montage('.\channel32.txt')


weight = scio.loadmat('weight_S4_A.mat')
data = weight['weight']
print(data.shape)
data = data / 1000000

info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')
evoked = mne.EvokedArray(data, info)
evoked.set_montage(biosemi_montage)
times = [0.0,0.004]

with PdfPages("plot.pdf") as pdf:
    graph = evoked.plot_topomap(times)
    plt.show()
    # pdf.savefig(graph)
    plt.close()


