# Paddle-Brain-Tumor-Segmentation

## 1.Introduction
This project is based on the paddlepaddle_V2.1 framework to reproduce Brain-Tumor-Segmentation and the [official code](https://github.com/jadevaibhav/Brain-Tumor-Segmentation-using-Deep-Neural-networks) of Keras.

## 2.Result

The model is trained on the train set of BraTS2013.

The model file of the keras version we trained(accuracy of 95.4%)


The model file of the paddle version we trained(accuracy of 95.4%)




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
python bts/forward.py # keras forward
python paddle-bts/forward.py # paddle forward
python bts/check_diff.py # check diff of forward step.
```  

        
* Network structure transfer.
* Weight transfer:
  * model of keras version to do the aligh : [paddle-bts/bts_paddle_ub.pdparams]()
  * model of paddle version transfered from keras: [paddle-bts/bts_paddle_ub.pdparams](https://github.com/tbymiracle/Paddle-Brain-Tumor-Segmentation/blob/main/paddle-bts/bts_paddle_ub.pdparams) 
* Verify the network.
* Forward align
  * bts/forward_keras.npy
  * paddle-bts/forward_paddle.npy
  * bts/forward_diff.log
* train align
