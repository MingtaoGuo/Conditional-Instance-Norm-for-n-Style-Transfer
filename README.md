# Conditional-Instance-Norm-for-n-Style-Transfer
Implementation of the paper A Learned Representation for Artistic Style

## Introduction
Simply implementing the paper [A Learned Representation for Artistic Style](https://arxiv.org/pdf/1610.07629.pdf) (Conditional instance normalization)
![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/cin.jpg)

``` python
def conditional_instance_norm(x, scope_bn, y1=None, y2=None, alpha=1):
    mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    if y1==None:
        beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True)  
        gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True) 
    else:
        beta = tf.get_variable(name=scope_bn+'beta', shape=[y1.shape[-1], x.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True) # label_nums x C
        gamma = tf.get_variable(name=scope_bn+'gamma', shape=[y1.shape[-1], x.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True) # label_nums x C
        beta1 = tf.matmul(y1, beta)
        gamma1 = tf.matmul(y1, gamma)
        beta2 = tf.matmul(y2, beta)
        gamma2 = tf.matmul(y2, gamma)
        beta = alpha * beta1 + (1. - alpha) * beta2
        gamma = alpha * gamma1 + (1. - alpha) * gamma2
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-10)
    return x
```

## How to use
1. Download the dataset [MSCOCO](http://images.cocodataset.org/zips/train2014.zip), and unzip the dataset to the folder 'MSCOCO'
```
├── imgs
├── results
├── save_imgs
├── save_para
├── style_imgs
├── vgg_para
├── MSCOCO
     ├── COCO_train2014_000000000009.jpg
     ├── COCO_train2014_000000000025.jpg
     ├── COCO_train2014_000000000030.jpg
     ├── COCO_train2014_000000000034.jpg
     ├── COCO_train2014_000000000036.jpg
     ├── COCO_train2014_000000000049.jpg
     ...
```
2. Download the vgg16.npy, and put it into the folder 'vgg_para'
3. Execute the python file 'main.py'

## Requirement
- python3.5
- tensorflow1.4.0
- scipy
- numpy
- pillow

## Results
Style = alpha * style2 + (1 - alpha) * style1

|Content|Style1|Style2|Result|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/imgs/5.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/5.png)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/10.png)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/4_9.gif)|

|Content|Style1|Style2|Result|
|-|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/imgs/11.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/2.png)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/1.png)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/0_1.gif)|

|Content|Style1|Style2|
|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/content.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/7.png)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/4.png)|

|alpha=0|alpha=0.6|alpha=1.0|
|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/1lanting_0.0.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/1lanting_0.6.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/1lanting_1.0.jpg)|

|Content|Style1|Style2|
|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/content.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/6.png)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/2.png)|

|alpha=0|alpha=0.6|alpha=1.0|
|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/lanting_0.0.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/lanting_0.6.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/lanting_1.0.jpg)|

|Content|Style1|Style2|
|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/content_dog.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/7.png)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/10.png)|

|alpha=0|alpha=0.2|alpha=0.4|alpha=0.6|alpha=0.8|alpha=1.0|
|-|-|-|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/dog_0.0.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/dog_0.2.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/dog_0.4.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/dog_0.6.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/dog_0.8.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/dog_1.0.jpg)|

|Content|style|result|
|-|-|-|
|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/door.jpg)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/6.png)|![](https://github.com/MingtaoGuo/Conditional-Instance-Norm-for-n-Style-Transfer/blob/master/IMAGES/door_.jpg)|
