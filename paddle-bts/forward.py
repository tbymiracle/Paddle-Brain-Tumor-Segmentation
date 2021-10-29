import os
from sklearn.utils import class_weight
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from model import two_path, input_cascade
from data_load import data_gen,BraTsDataset
import paddle
from paddle import nn
from paddle.static import InputSpec
import paddle.nn.functional as F
from sklearn import metrics
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()
m = input_cascade((65,65,4),(33,33,4))

input = [InputSpec([None, 65, 65, 4], 'float32', 'image'),InputSpec([None, 33, 33, 4], 'float32', 'image1')]
label = InputSpec([None, 1,1,5], 'float32', 'label')

model = paddle.Model(m,input,label)
model.prepare()

model.load('bts_paddle_ub.pdparams')
fake_data1 = np.load("../keras-bts/fake_data.npy")
fake_data2 = np.load("../keras-bts/fake_data2.npy")
fake_label = np.load("../keras-bts/fake_label.npy")
testset = BraTsDataset(fake_data1,fake_data2,fake_label)
pred = model.predict(testset,batch_size = 1)
reprod_logger.add("logits", pred[0][0])
reprod_logger.save("forward_paddle.npy")
print(pred[0][0])



