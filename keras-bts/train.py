import os
from sklearn.utils import class_weight
import SimpleITK as sitk
import numpy as np
from model import input_cascade
from data_load import data_gen

m1 = input_cascade((65,65,4),(33,33,4))
# m1.summary()

m1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

path = '../HGG'
p = os.listdir(path)
p.sort(key=str.lower)
print(p)
arr = []
for k in range(len(p)):
    q = os.listdir(path + '/' + p[k])
    q.sort(key=str.lower)
    for j in range(len(q)):
        if (j != 4):
            img = sitk.ReadImage(path + '/' + p[k] + '/' + q[j])
            img = sitk.GetArrayFromImage(img)
            arr.append(img)
        else:
            img = sitk.ReadImage(path + '/' + p[k] + '/' + q[j])
            Y_labels = sitk.GetArrayFromImage(img)
            print(Y_labels.shape)
    data = np.zeros((Y_labels.shape[1], Y_labels.shape[0], Y_labels.shape[2], 4))
    for i in range(Y_labels.shape[1]):
        data[i, :, :, 0] = arr[0][:, i, :]
        data[i, :, :, 1] = arr[1][:, i, :]
        data[i, :, :, 2] = arr[2][:, i, :]
        data[i, :, :, 3] = arr[3][:, i, :]
    info = []

    # Creating patches for each slice and training(slice-wise)
    for i in range(data.shape[0]):
        d = data_gen(data, Y_labels, i, 1)
        if (len(d) != 0):
            y = np.zeros((d[2].shape[0], 1, 1, 5))
            for j in range(y.shape[0]):
                y[j, :, :, d[2][j]] = 1
            X1 = d[0]
            X2 = d[1]
            class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                              classes=np.unique(d[2]),
                                                              y=d[2])

            print('slice no:' + str(i))
            info.append(m1.fit([X1, X2], y, epochs=1, batch_size=64, class_weight= class_weights))
            m1.save('trial_input_cascasde_image_' + str(k) + '_slice_' + str(i) + '_acc.h5')

    m1.save('trial_input_cascasde_image_' + str(k) + '_acc.h5')


