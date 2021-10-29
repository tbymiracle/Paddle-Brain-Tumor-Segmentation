import os
from sklearn.utils import class_weight
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from model import two_pathcnn, MFCcascade,input_cascade,two_path
from data_load import data_gen
import keras


m1 = input_cascade((65,65,4),(33,33,4))
#m1.summary()
m1.load_weights('trial_input_cascasde_acc.h5')

fake_data1 = np.load("../keras-bts/fake_data.npy")
fake_data2 = np.load("../keras-bts/fake_data2.npy")
fake_label = np.load("../keras-bts/fake_label.npy")
pred = m1.predict([fake_data1,fake_data2])
print(pred[0].shape,pred[0])
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()
reprod_logger.add("logits", pred[0])

reprod_logger.save("forward_keras.npy")


