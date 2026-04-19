import mne
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP

def create_raw(data,labels,info,freq=250):
    labels=np.ravel(labels)
    ch_names=info.ch_names
    si,sj,sk=data.shape
    da=data.transpose(1,0,2)
    da=da.reshape(sj,si*sk)
    llen=data.shape[0]
    event=np.zeros((llen,3))
    info = mne.create_info(
        ch_names=ch_names,
        ch_types="eeg",  # channel type
        sfreq=freq  # frequency
    )
    raw = mne.io.RawArray(da, info)  # create raw
    for i in range(llen):
        event[i,0]=i*sk
        event[i,2]=labels[i]
    event=event.astype(int)
    # raw.info['events']=event
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    return raw,event


def smooth2nd(x, M=80):  ## data smooth
    K = round(M / 2 - 0.1)
    lenX = len(x)
    if lenX < 2 * K + 1:
        print('Error')
    else:
        y = np.zeros(lenX)
        for NN in range(0, lenX, 1):
            startInd = max([0, NN - K])
            endInd = min(NN + K + 1, lenX)
            y[NN] = np.mean(x[startInd:endInd])
    return (y)

def get_erders(datas):
    data=datas.transpose(1,2,0)
    data=data*data
    # C3=np.sum(data[12,:,:],axis=1)
    # C4=np.sum(data[14,:,:],axis=1)
    C3=np.sum(data[12,:,:],axis=1)
    C4=np.sum(data[22,:,:],axis=1)
    C3=C3/len(datas)
    C4=C4/len(datas)
    C3=smooth2nd(C3,80)
    C4=smooth2nd(C4,80)
    C3mean=np.average(C3[0:125])
    C4mean=np.average(C4[0:125])
    C3=((C3-C3mean)/C3mean)*1
    C4=((C4-C4mean)/C4mean)*1
    return C3,C4

def check_data(data):
    resdata=[]
    for i in range(len(data)):
        dats=data[i]
        s=np.sum(dats>100)
        if s ==0:
            resdata.append(dats)
    resdata=np.array(resdata)
    return resdata


data_A=np.empty((0,32,1000))
labels_A=np.empty((0))

path='.\S3\S3_A.mat'
info = mne.create_info(
    ch_names=["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
            "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
            "O1", "O2"],
    ch_types='eeg', sfreq=250)

da=sio.loadmat(path)
data=da['x']
data = np.transpose(data, (2,0,1))
labels=np.ravel(da['y'])

# print(data.shape)
# print(labels.shape)
# asd

raw, event = create_raw(data, labels, info)
# raw.filter(8, 30, fir_design='firwin', skip_by_annotation='edge')
event_id = dict(left=1, right=2)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False, exclude='bads')
epochs = mne.Epochs(raw, event, event_id, 0, 3.996, proj=True, picks=picks, baseline=None, preload=True)
data=epochs.get_data()
data_A=np.vstack((data_A,data))
labels_A=np.hstack((labels_A,np.ravel(labels)))

data_A_left=[]
data_A_right=[]
for i in range(len(data_A)):
    if labels_A[i]==1:
        data_A_left.append(data_A[i])
    else:
        data_A_right.append(data_A[i])
data_A_left = np.array(data_A_left)
data_A_right = np.array(data_A_right)

LC3_A,LC4_A = get_erders(data_A_left)
RC3_A,RC4_A = get_erders(data_A_right)


data_B=np.empty((0,32,1000))
labels_B=np.empty((0))

path='.\S3\S3_B.mat'
info = mne.create_info(
    ch_names=["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
            "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
            "O1", "O2"],
    ch_types='eeg', sfreq=250)

da=sio.loadmat(path)
data=da['x']
data = np.transpose(data, (2,0,1))
labels=np.ravel(da['y'])

raw, event = create_raw(data, labels, info)
# raw.filter(8, 30, fir_design='firwin', skip_by_annotation='edge')
event_id = dict(left=1, right=2)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False, exclude='bads')
epochs = mne.Epochs(raw, event, event_id, 0, 3.996, proj=True, picks=picks, baseline=None, preload=True)
data=epochs.get_data()
data_B=np.vstack((data_B,data))
labels_B=np.hstack((labels_B,np.ravel(labels)))

data_B_left=[]
data_B_right=[]
for i in range(len(data_B)):
    if labels_B[i]==1:
        data_B_left.append(data_B[i])
    else:
        data_B_right.append(data_B[i])
data_B_left = np.array(data_B_left)
data_B_right = np.array(data_B_right)

LC3_B,LC4_B = get_erders(data_B_left)
RC3_B,RC4_B = get_erders(data_B_right)


data_C=np.empty((0,32,1000))
labels_C=np.empty((0))

path='.\S3\S3_C.mat'
info = mne.create_info(
    ch_names=["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
            "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
            "O1", "O2"],
    ch_types='eeg', sfreq=250)

da=sio.loadmat(path)
data=da['x']
data = np.transpose(data, (2,0,1))
labels=np.ravel(da['y'])

raw, event = create_raw(data, labels, info)
# raw.filter(8, 30, fir_design='firwin', skip_by_annotation='edge')
event_id = dict(left=1, right=2)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False, exclude='bads')
epochs = mne.Epochs(raw, event, event_id, 0, 3.996, proj=True, picks=picks, baseline=None, preload=True)
data=epochs.get_data()
data_C=np.vstack((data_C,data))
labels_C=np.hstack((labels_C,np.ravel(labels)))

data_C_left=[]
data_C_right=[]
for i in range(len(data_C)):
    if labels_C[i]==1:
        data_C_left.append(data_C[i])
    else:
        data_C_right.append(data_C[i])
data_C_left = np.array(data_C_left)
data_C_right = np.array(data_C_right)

LC3_C,LC4_C = get_erders(data_C_left)
RC3_C,RC4_C = get_erders(data_C_right)




data_D=np.empty((0,32,1000))
labels_D=np.empty((0))

path='.\S3\S3_D.mat'
info = mne.create_info(
    ch_names=["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
            "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
            "O1", "O2"],
    ch_types='eeg', sfreq=250)

da=sio.loadmat(path)
data=da['x']
data = np.transpose(data, (2,0,1))
labels=np.ravel(da['y'])

raw, event = create_raw(data, labels, info)
# raw.filter(8, 30, fir_design='firwin', skip_by_annotation='edge')
event_id = dict(left=1, right=2)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False, exclude='bads')
epochs = mne.Epochs(raw, event, event_id, 0, 3.996, proj=True, picks=picks, baseline=None, preload=True)
data=epochs.get_data()
data_D=np.vstack((data_D,data))
labels_D=np.hstack((labels_D,np.ravel(labels)))

data_D_left=[]
data_D_right=[]
for i in range(len(data_D)):
    if labels_D[i]==1:
        data_D_left.append(data_D[i])
    else:
        data_D_right.append(data_D[i])
data_D_left = np.array(data_D_left)
data_D_right = np.array(data_D_right)

LC3_D,LC4_D = get_erders(data_D_left)
RC3_D,RC4_D = get_erders(data_D_right)


plt.plot(-LC3_A,label='Left MI Complex-A')
plt.plot(-RC3_A,label='Right MI Complex-A')
plt.plot(-LC3_B,label='Left MI Complex-B')
plt.plot(-RC3_B,label='Right MI Complex-B')
plt.plot(-LC3_C,label='Left MI Complex-C')
plt.plot(-RC3_C,label='Right MI Complex-C')
plt.plot(-LC3_D,label='Left MI Complex-D')
plt.plot(-RC3_D,label='Right MI Complex-D')

plt.ylabel('Channel C3')
plt.xlabel('Sampling points')
plt.legend(loc=0)
plt.show()


plt.plot(-RC4_A,label='Left MI Complex-A')
plt.plot(-LC4_A,label='Right MI Complex-A')
plt.plot(-RC4_B,label='Left MI Complex-B')
plt.plot(-LC4_B,label='Right MI Complex-B')
plt.plot(-RC4_C,label='Left MI Complex-C')
plt.plot(-LC4_C,label='Right MI Complex-C')
plt.plot(-LC4_D,label='Left MI Complex-D')
plt.plot(-RC4_D,label='Right MI Complex-D')

plt.ylabel('Channel C4')
plt.xlabel('Sampling points')
plt.legend(loc=0)
plt.show()






