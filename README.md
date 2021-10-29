# Paddle-Brain-Tumor-Segmentation

## 1.Introduction
This project is based on the paddlepaddle_V2.1 framework to reproduce Brain-Tumor-Segmentation. 

We put the keras version in keras-bts/ (the [official code](https://github.com/jadevaibhav/Brain-Tumor-Segmentation-using-Deep-Neural-networks) based on keras is implemented in jupyter notebook and so we change it a little). 

We put our project in paddle-bts/. Our project can achieve almost the same results. 
## 2.Result

The model is trained on the train set of BraTS2015.

 Version | Dice Complete | Dice Core | Dice Enhancing
 ---- | ----- | -----  | -----
 pytorch version(official)  | -  | - | -
 paddle version(ours) | 0.907|  0.96 | 1.0


The model file of the keras version we trained：


The model file of the paddle version we trained：

链接: https://pan.baidu.com/s/1M5wGRSbIcmQLsvCoeS5Lsg 提取码: pi6p 复制这段内容后打开百度网盘手机App，操作更方便哦


## 3.Requirements

 * Hardware：GPU（Tesla V100-32G is recommended）
 * Framework:  PaddlePaddle >= 2.1.2


## 4.Quick Start

### Step1: Clone

``` 
git clone https://github.com/tbymiracle/Paddle-Brain-Tumor-Segmentation.git
cd paddle-bts
``` 

### Step2: Training

```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
### Step3: Evaluating

```  
CUDA_VISIBLE_DEVICES=0 python test.py
```  

## 5.Align

We use the [`repord_log`](https://github.com/WenmuZhou/reprod_log) tool to align.

The author use keras framework in the official version, so it is hard to align in all the steps.

Thus, we can only do the forward align.

```  
python keras-bts/forward.py # keras forward
python paddle-bts/forward.py # paddle forward
python keras-bts/check_diff.py # check diff of forward step.
```  

        
* Network structure transfer.
* Weight transfer:
  * model of keras version to do the aligh : [keras-bts/trial_input_cascasde_acc.h5](https://github.com/tbymiracle/Paddle-Brain-Tumor-Segmentation/blob/main/keras-bts/trial_input_cascasde_acc.h5)
  * model of paddle version transfered from keras: [paddle-bts/bts_paddle_ub.pdparams](https://github.com/tbymiracle/Paddle-Brain-Tumor-Segmentation/blob/main/paddle-bts/bts_paddle_ub.pdparams) 
* Verify the network.
* Forward align
  * keras-bts/forward_keras.npy
  * paddle-bts/forward_paddle.npy
  * keras-bts/forward_diff.log
* Train align
