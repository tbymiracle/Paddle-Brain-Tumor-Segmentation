import os
import keras
import SimpleITK as sitk
import numpy as np
from sklearn import metrics
from model import input_cascade
from data_load import data_gen

# m1 = input_cascade((65,65,4),(33,33,4))
# m1.summary()

m1 = keras.models.load_model('trial_input_cascasde_image_3_acc.h5')

path = 'HGG/brats_2013_pat0027_1'
q = os.listdir(path)
q.sort(key=str.lower)
arr = []
for j in range(len(q)):
    if (j != 4):
        img = sitk.ReadImage(path + '/' + q[j])
        img = sitk.GetArrayFromImage(img)
        arr.append(img)
    else:
        img = sitk.ReadImage(path + '/' + q[j])
        Y_labels = sitk.GetArrayFromImage(img)

data = np.zeros((Y_labels.shape[1], Y_labels.shape[0], Y_labels.shape[2], 4))
for i in range(Y_labels.shape[1]):
    data[i, :, :, 0] = arr[0][:, i, :]
    data[i, :, :, 1] = arr[1][:, i, :]
    data[i, :, :, 2] = arr[2][:, i, :]
    data[i, :, :, 3] = arr[3][:, i, :]
print(data.shape)
info = []
# Creating patches for each slice and training(slice-wise)
print(data.shape[0])

#test
fa=[]
fb=[]
fc=[]
for i in range(data.shape[0]):
    d_test = data_gen(data,Y_labels,i,1)
    if(len(d_test) != 0):
        y_test = np.zeros((d_test[2].shape[0],1,1,5))
        for j in range(y_test.shape[0]):
            y_test[j,:,:,d_test[2][j]] = 1
        X1_test = d_test[0]
        X2_test = d_test[1]
        pred = m1.predict([X1_test,X2_test],batch_size=64)
        # print(pred.shape)
        pred = np.around(pred)
        pred1 = np.argmax(pred.reshape(y_test.shape[0],5)[:,1:5],axis = 1)
        y2 = np.argmax(y_test.reshape(y_test.shape[0],5)[:,1:5],axis = 1)
        f1 = metrics.f1_score(y2, pred1, average='micro')
        print(f1)
        fa.append(f1)

        pred1 = np.argmax(pred.reshape(y_test.shape[0], 5)[:, np.r_[1:2, 3:5]], axis=1)
        y2 = np.argmax(y_test.reshape(y_test.shape[0], 5)[:, np.r_[1:2, 3:5]], axis=1)
        f1 = metrics.f1_score(y2, pred1, average='micro')
        print(f1)
        fb.append(f1)

        pred1 = np.argmax(pred.reshape(y_test.shape[0], 5)[:,4:5], axis=1)
        y2 = np.argmax(y_test.reshape(y_test.shape[0], 5)[:,4:5], axis=1)
        f1 = metrics.f1_score(y2, pred1, average='micro')
        print(f1)
        fc.append(f1)

print(sum(fa)/len(fa))
print(sum(fb)/len(fb))
print(sum(fc)/len(fc))