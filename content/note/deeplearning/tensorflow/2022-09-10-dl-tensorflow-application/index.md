---
title: TensorFlow Application
author: 王哲峰
date: '2022-09-10'
slug: dl-tensorflow-application
categories:
  - tensorflow
tags:
  - tool
---

<style>
details {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
}
summary {
    font-weight: bold;
    margin: -.5em -.5em 0;
    padding: .5em;
}
details[open] {
    padding: .5em;
}
details[open] summary {
    border-bottom: 1px solid #aaa;
    margin-bottom: .5em;
}
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [预训练模型下载](#预训练模型下载)
- [图像分类模型](#图像分类模型)
  - [模型](#模型)
  - [API](#api)
- [示例](#示例)
  - [图像分类模型使用示例](#图像分类模型使用示例)
</p></details><p></p>

# 预训练模型下载

Keras Applications(`tensorflow.keras.applications`) 提供了预训练好的深度学习模型, 
这些模型可以用于预测、特征提取等

当初始化一个模型时就会自动下载, 默认下载的路径是: `~/.keras/models/`

![img](images/keras_models.png)

# 图像分类模型

## 模型

在 ImageNet 数据上预训练过的用于图像分类的模型

* Xception
* VGG16
* VGG19
* ResNet, ResNetV2, ResNeXt
* InceptionV3
* InceptionResNet2
* MobileNet
* MobileNetV2
* DenseNet
* NASNet

## API

```python
from tensorflow.keras.applications.xception import Xception

# channels_last only; 299x299
xception_model = Xception(
    include_top = True,
    weights = "imagenet",
    input_tensor = None, 
    input_shape = None,
    pooling = None,
    classes = 1000,
)
```


```python
from tensorflow.keras.applications.vgg16 import VGG16

# channels_first and channels_last; 224x224
vgg16_model = VGG16(
    include_top = True,
    weights = "imagenet",
    input_tensor = None, 
    input_shape = None,
    pooling = None,
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.vgg19 import VGG19

vgg19_model = VGG19(
    include_top = True, 
    weights = 'imagenet',
    input_tensor = None, 
    input_shape = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.resnet50 import ResNet50

resnet50_model = ResNet50(
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    input_shape = None, 
    pooling = None, 
    classes = 1000,
)
```


```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

inception_v3_model = InceptionV3(
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    input_shape = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

inception_resnet_v2_model = InceptionResNetV2(
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    input_shape = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.mobilenet import MobileNet

mobilenet_model = MobileNet(
    input_shape = None, 
    alpha = 1.0, 
    depth_multiplier = 1, 
    dropout = 1e-3, 
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.densenet import DenseNet121

densenet_model = DenseNet121(
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    input_shape = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.densenet import DenseNet169

densenet_model = DenseNet169(
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    input_shape = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.densenet import DenseNet201

densenet_model = DenseNet201(
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    input_shape = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.nasnet import NASNetLarge

nasnet_model = NASNetLarge(
    input_shape = None, 
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.nasnet import NASNetMobile

nasnet_model = NASNetMobile(
    input_shape = None, 
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    pooling = None, 
    classes = 1000,
)
```

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

mobilenet_v2_model = MobileNetV2(
    input_shape = None, 
    alpha = 1.0, 
    depth_multiplier = 1, 
    include_top = True, 
    weights = 'imagenet', 
    input_tensor = None, 
    pooling = None, 
    classes = 1000,
)
```


# 示例

## 图像分类模型使用示例

```python
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_prediction
import numpy as np

# Load model
model = ResNet50(weights = "imagenet")

# Image data
img_path = "elephant.jpg"
img = image.load_img(img_path, target_size = (224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

preds = model.predict(x)
print("Predicted:", decode_prediction(preds, top = 3)[0])
```
