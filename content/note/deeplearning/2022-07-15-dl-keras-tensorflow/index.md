---
title: TensorFlow & Keras
author: 王哲峰
date: '2022-07-15'
slug: dl-keras-tensorflow
categories:
  - deeplearning
tags:
  - tool
---

# 需要掌握的内容

## Tensorflow 新手

* [x] 快速入门
* [ ] Keras 机器学习基础知识
* [ ] 加载和预处理数据

## Tensorflow 专家:

* [x] 快速入门
* [ ] 自定义层、训练循环
* [ ] 分布式训练

## TensorFlow 库和扩展程序

* [ ] TensorBoard
* [ ] TensorFLow Hub
* [ ] 数据集
* [ ] 模型优化
* [ ] 概率
* [ ] XLA
* [ ] TFX
* [ ] ...

# Tensorflow & Keras 数据

## 1.TensorFlow Datasets 库

* 库安装:

```bash
$ pip install tensorflow
$ pip install tensorflow-datasets
```

* 库导入:

```python
# tf.data, tf.data.Dataset
import tensorflow as tf

# tf.keras.datasets.<dataset_name>.load_data
from tensorflow.keras import datasets

# tfds.load
import tensorflow_datasets as tfds
```

## 2.TensorFlow Dataset 介绍

### 2.1 tf.data

TensorFlow 提供了 `tf.data` 模块, 它包括了一套灵活的数据集构建 API, 
能够帮助快速、高效地构建数据输入的流水线, 尤其适用于数据量巨大的情景
`tf.data` API 在 TensorFlow 中引入了两个新的抽象类:

* `tf.data.Dataset`
    - `tf.data.Dataset`: 提供了对数据集的高层封装。`tf.data.Dataset` 由一系列可迭代访问的元素(element)组成, 
    其中每个元素包含一个或多个 `Tensor` 对象。`tf.data.Dataset` 可以通过两种方式来创建数据集:
        - **创建来源**: 通过一个或多个 `tf.Tensor` 对象构建数据集
            - `tf.data.Dataset.from_tensors()`
            - `tf.data.Dataset.from_tensor_slices()`
        - **应用转换**: 通过一个或多个 `tf.data.Dataset` 对象构建数据集
            - `tf.data.Dataset.map()`
            - `tf.data.Dataset.batch()`
* `tf.data.Iterator`
    - `tf.data.Iterator`: 提供了从数据集中提取元素的主要方法
    - `tf.data.Iterator.get_next()`
        - 返回的操作会在执行时生成Dataset的下一个元素, 并且此操作通常当输入管道和模型之间的接口
    - `tf.data.Iterator.initializer`
        - 使用不同的数据集重新初始化和参数化迭代器

### 2.2 tensorflow_datasets

TensorFlow Datasets(`tensorflow_datasets`) 是可用于 TensorFlow 
或其他 Python 机器学习框架(例如 Jax) 的一系列数据集。
所有数据集都作为 `tf.data.Dataset` 提供, 实现易用且高性能的输入流水线。


## 3.TensorFlow 数据集 API
  
* `tf.data`
    - `tf.data.Dataset`
    - `tf.data.Dataset.from_tensor_slices`
* `tensorflow_datasets`
    - `tensorflow_datasets.load(data, split, shuffle_files, as_supervised)`
* `tf.keras.datasets`
    - `tf.keras.datasets.mnist.load_data()`

## 3.TensorFlow Dataset 建立

1. 建立 `tf.data.Dataset` 的最基本的方法是使用 `tf.data.Dataset.from_tensor_slices()`

    - 适用于数据量较小(能够将数据全部装进内存)的情况
    - 如果数据集中的所有元素通过张量的第 0 维拼接成一个大的张量

```python
import tensorflow as tf
import numpy as np

X = tf.constant([2013, 2014, 2015, 2016, 2017])
Y = tf.constant([12000, 14000, 15000, 16500, 17500])

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
for x, y in dataset:
    print(x.numpy(), y.numpy())
```

2.使用 `tf.data.Dataset.from_tensor_slices()`、`tf.keras.datasets.mnist.load_data()`

```python
import tensorflow as tf
import matplotlib.pyplot as plt

(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dim(train_data.astype(np.float32) / 255, axis = -1)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))

for image, label in mnist_dataset.take(1):
    plt.title(label.numpy())
    plt.imshow(image.numpy())
    plt.show()
```

3.TensorFlow Datasets 提供了一系列可以和 Tensorflow 配合使用的数据集, 它负责下载和准备数据, 以及构建 `tf.data.Dataset`

- 每一个数据集(dataset) 都实现了抽象基类 tfds.core.DatasetBuilder 来构建

```python
import tensorflow_datasets as tfds

# 构建 tf.data.Dataset
dataset1 = tfds.load("mnist", split = "train", shuffle_files = True)
dataset2 = tfds.load("mnist", split = tfds.Split.TRAIN, as_supervised = True)

# 构建输入数据 Pipeline
dataset1 = dataset1 \
  .shuffle(1024) \
  .batch(32) \
  .prefetch(tf.data.experimential.AUTOTUNE)

for example in dataset1.take(1):
  image, label = example["image"], example["label"]
```

.. note:: 

- 对于特别巨大而无法完整载入内存的数据集, 可以先将数据集处理为 `TFRecord` 格式, 
然后使用 `tf.data.TFRecordDataset()` 进行载入

## 4.TensorFlow 内置 Dataset

- TensorFlow Datasets 提供了一系列可以和 Tensorflow 配合使用的数据集, 它负责下载和准备数据, 以及构建 `tf.data.Dataset`
- 每一个数据集(dataset) 都实现了抽象基类 tfds.core.DatasetBuilder 来构建

- 官方文档

- https://github.com/tensorflow/datasets
- https://www.tensorflow.org/datasets/overview
- https://www.tensorflow.org/datasets/catalog/overview#all_datasets
- https://www.tensorflow.org/datasets/api_docs/python/tfds
- https://blog.tensorflow.org/2019/02/introducing-tensorflow-datasets.html?hl=zh-CN

1. 查看可用的数据集

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# 所有可用的数据集
print(tfds.list_builders()) 

['abstract_reasoning', 'aflw2k3d', 'amazon_us_reviews', 
'bair_robot_pushing_small', 'bigearthnet', 'binarized_mnist', 'binary_alpha_digits', 
'caltech101', 'caltech_birds2010', 'caltech_birds2011', 'cats_vs_dogs', 'celeb_a', 'celeb_a_hq', 'chexpert', 'cifar10', 'cifar100', 'cifar10_corrupted', 'clevr', 'cnn_dailymail', 'coco', 'coco2014', 'coil100', 'colorectal_histology', 'colorectal_histology_large', 'curated_breast_imaging_ddsm', 'cycle_gan', 
'deep_weeds', 'definite_pronoun_resolution', 'diabetic_retinopathy_detection', 'downsampled_imagenet', 'dsprites', 'dtd', 'dummy_dataset_shared_generator', 'dummy_mnist', 
'emnist', 'eurosat', 
'fashion_mnist', 'flores', 'food101', 
'gap', 'glue', 'groove', 
'higgs', 'horses_or_humans', 
'image_label_folder', 'imagenet2012', 'imagenet2012_corrupted', 'imdb_reviews', 'iris', 'kitti', 
'kmnist', 
'lfw', 'lm1b', 'lsun', 
'mnist', 'mnist_corrupted', 'moving_mnist', 'multi_nli', 
'nsynth', 
'omniglot', 'open_images_v4', 'oxford_flowers102', 'oxford_iiit_pet', 
'para_crawl', 'patch_camelyon', 'pet_finder', 'quickdraw_bitmap', 
'resisc45', 'rock_paper_scissors', 'rock_you', 
'scene_parse150', 'shapes3d', 'smallnorb', 'snli', 'so2sat', 'squad', 'stanford_dogs', 'stanford_online_products', 'starcraft_video', 'sun397', 'super_glue', 'svhn_cropped', 
'ted_hrlr_translate', 'ted_multi_translate', 'tf_flowers', 'titanic', 'trivia_qa', 
'uc_merced', 'ucf101', 
'visual_domain_decathlon', 'voc2007', 
'wikipedia', 'wmt14_translate', 'wmt15_translate', 'wmt16_translate', 'wmt17_translate', 'wmt18_translate', 'wmt19_translate', 'wmt_t2t_translate', 'wmt_translate', 
'xnli']
```

2. 内置数据集分类

- Audio

   - groove
   - nsynth

- Image

   - abstract_reasoning
   - aflw2k3d
   - bigearthnet
   - binarized_mnist
   - binaryalphadigits
   - caltech101
   - caltech_birds2010
   - caltech_birds2011
   - catsvsdogs
   - celeb_a
   - celebahq
   - cifar10
   - cifar100
   - cifar10_corrupted
   - clevr
   - coco
   - coco2014
   - coil100
   - colorectal_histology
   - colorectalhistologylarge
   - curatedbreastimaging_ddsm
   - cycle_gan
   - deep_weeds
   - diabeticretinopathydetection
   - downsampled_imagenet
   - dsprites
   - dtd
   - emnist
   - eurosat
   - fashion_mnist
   - food101
   - horsesorhumans
   - imagelabelfolder
   - imagenet2012
   - imagenet2012_corrupted
   - kitti
   - kmnist
   - lfw
   - lsun
   - mnist
   - mnist_corrupted
   - omniglot
   - openimagesv4
   - oxford_flowers102
   - oxfordiiitpet
   - patch_camelyon
   - pet_finder
   - quickdraw_bitmap
   - resisc45
   - rockpaperscissors
   - scene_parse150
   - shapes3d
   - smallnorb
   - so2sat
   - stanford_dogs
   - stanfordonlineproducts
   - sun397
   - svhn_cropped
   - tf_flowers
   - uc_merced
   - visualdomaindecathlon
   - voc2007

- Structured

   - amazonusreviews
   - higgs
   - iris
   - rock_you
   - titanic

- Text

   - cnn_dailymail
   - definitepronounresolution
   - gap
   - glue
   - imdb_reviews
   - lm1b
   - multi_nli
   - snli
   - squad
   - super_glue
   - trivia_qa
   - wikipedia
   - xnli

- Translate

   - flores
   - para_crawl
   - tedhrlrtranslate
   - tedmultitranslate
   - wmt14_translate
   - wmt15_translate
   - wmt16_translate
   - wmt17_translate
   - wmt18_translate
   - wmt19_translate
   - wmtt2ttranslate

- Video

   - bairrobotpushing_small
   - moving_mnist
   - starcraft_video
   - ucf101

3. 构建并加载内置数据集

- `tfds.load` 是构建并加载 `tf.data.Dataset` 最简单的方式
- `tf.data.Dataset` 是构建输入流水线的标准 TensorFlow 接口
- 加载数据集时, 默认使用规范的版本, 但是可以指定要使用的数据集的主版本, 并在结果中表明使用了哪个版本的数据集

示例1:

```python
mnist_train = tfds.load("mnist", split = "train", download = False, data_dir = "~/.tensorflow_datasets/")
assert isinstance(mnist_train, tf.data.Dataset)
print(mnist_train)
```

示例2: 版本控制

```python
mnist = tfds.load("mnist:1.*.*")
```

4. 内置数据集特征字典

- 所有 `tensorflow_datasets, tfds` 数据集都包含将特征名称映射到 Tensor 值的特征字典。典型的数据集将具有 2 个键:
  - `"image"`
  - `"label"`

示例:

```python
mnist_train = tfds.load("mnist", split = "train", download = False, data_dir = "~/.tensorflow_datasets/")
for mnist_example in mnist_train.take(1):
   image, label = mnist_example["image"], mnist_example["label"]
   plt.imshow(
      image.numpy()[:, :, 0].astype(np.float32),
      cma = plt.get_cmap("gray")
   )
   print("Label: %d" % label.numpy())
   plt.show()
```

5. DatasetBuilder

- `tensorflow_datasets.load` 实际上是一个基于 `DatasetBuilder` 的简单方便的包装器

示例:

```python
mnist_builder = tfds.builder("mnist")
mnsit_builder.download_and_prepare()
mnist_train = mnist_builder.as_dataset(split = "train")
mnist_train
```

6. 内置数据集输入流水线

- 一旦有了 `tf.data.Dataset` 对象, 就可以使用 `tf.data` 接口定义适合模型训练的输入流水线的其余部分.

示例:

```python
mnist_train = mnist_train.repeat().shuffle(1024).batch(32)

# prefetch 将使输入流水线可以在模型训练时一步获取批处理
mnist_train = mnist_train \
               .repeat() \
               .shuffle(1024) \
               .batch(32) \
               .prefetch(tf.data.experimental.AUTOTUNE)
```

7. 内置数据集信息

示例:

```python
# method 1
info = mnist_builder.info
print(info)
print(info.features)
print(info.features["label"].num_classes)
print(info.features["label"].names)

# method 2
mnist_test, info = tfds.load("mnist", split = "test", with_info = True)
print(info)
```

8. 内置数据集可视化

示例:

```python
fig = tfds.show_examples(info, mnist_test)
```

## 5.TensorFlow Dataset 预处理

### 5.1 数据集预处理 API 介绍

- Sequence Preprocessing
   - TimeseriesGenerator
   - pad_sequences
   - skipgrams
   - makesamplingtable
- Text Preprocessing
   - Tokenizer
   - hashing_trick
      - 将文本转换为固定大小的散列空间中的索引序列
   - one_hot
      - One-hot将文本编码为大小为n的单词索引列表
   - texttoword_sequence
      - 将文本转换为单词(或标记)序列
- Image Preprocessing
    - ``class`` ImageDataGenerator
    - ``method``
        - .apply_transform()
        - .fit ()
        - .flow()
        - 采用数据和标签数组, 生成批量增强数据
        - .flowfromdataframe()
        - 获取数据帧和目录路径, 并生成批量的扩充/规范化数据
        - .flowfromdirectory()
        - 获取目录的路径并生成批量的增强数据
        - .getrandomtransform()
        - 为转换生成随机参数
        - .random_transform()
        - 随机转换
        - .standardize()
        - 标准化


- `tf.data.Dataset` 类提供了多种数据集预处理方法:
    - `tf.data.Dataset.map(f)`: 
        - 对数据集中的每个元素应用函数 `f`, 得到一个新的数据集
        - 结合 `tf.io` 对文件进行读写和解码
        - 结合 `tf.image` 进行图像处理
    - `tf.data.Dataset.shuffle(buffer_size)`: 
        - 将数据集打乱
        - 设定一个固定大小的缓冲区(buffer), 取出前 buffer_size 个元素放入, 并从缓冲区中随机采样, 采样后的数据用后续数据替换
    - `tf.data.Dataset.batch(batch_size)`: 
        - 将数据集分成批次
        - 对每 `batch_size` 个元素, 使用 `tf.stack()` 在第 0 维合并, 成为一个元素
    - `tf.data.Dataset.repeat()`: 
        - 重复数据集的元素
    - `tf.data.Dataset.reduce()`: 
        - 与 Map 相对的聚合操作
    - `tf.data.Dataset.take()`: 
        - 截取数据集中的前若干个元素
    - `tf.data.Dataset.prefetch()`:
        - 并行化策略提高训练流程效率
        - 获取与使用 `tf.data.Dataset` 数据集元素
    - `tf.data.Dataset` 是一个 Python 的可迭代对象


### 4.2 数据集处理示例

- (1)使用 `tf.data.Dataset.map()` 将所有图片旋转 90 度

```python
import tensorflow as tf

# data preprocessing function
def rot90(image, label):
   image = tf.image.rot90(image)
   return image, label

# data
mnist_dataset = tf.keras.datasets.mnist.load_data()

# data preprocessing
mnist_dataset = mnist_dataset.map(rot90)

# data visual
for image, label in mnist_dataset:
   plt.title(label.numpy())
   plt.imshow(image.numpy()[:, :, 0])
   plt.show()
```

- (2)使用 `tf.data.Dataset.batch()` 将数据集划分为批次, 每个批次的大小为 4

```python
import tensorflow as tf

# data
mnist_dataset = tf.keras.datasets.mnist.load_data()

# data preprocessing
mnist_dataset = mnist_dataset.batch(4)

# data visual
for images, labels in mnist_dataset: # image: [4, 28, 28, 1], labels: [4]
   fig, axs = plt.subplots(1, 4)
   for i in range(4):
      axs[i].set_title(label.numpy()[i])
      axs[i].imshow(images.numpy()[i, :, :, 0])
   plt.show()
```

- (3)使用 `tf.data.Dataset.shuffle()` 将数据打散后再设置批次, 缓存大小设置为 10000

```python
import tensorflow as tf

# data
mnist_dataset = tf.keras.datasets.mnist.load_data()

# data preprocessing
mnist_dataset = mnist_dataset.shuffle(buffer_size = 10000).batch(4)

# data visual
for i in range(2):
   for images, labels in mnist_dataset: # image: [4, 28, 28, 1], labels: [4]
      fig, axs = plt.subplots(1, 4)
      for i in range(4):
         axs[i].set_title(label.numpy()[i])
         axs[i].imshow(images.numpy()[i, :, :, 0])
      plt.show()
```

.. note:: 

- 一般而言, 若数据集的顺序分布较为随机, 则缓冲区的大小可较小, 否则需要设置较大的缓冲区

- (4)使用 `tf.data.Dataset.prefetch()` 并行化策略提高训练流程效率

- 常规的训练流程
- 当训练模型时, 希望充分利用计算资源, 减少 CPU/GPU 的空载时间, 然而, 有时数据集的准备处理非常耗时, 
   使得在每进行一次训练前都需要花费大量的时间准备带训练的数据, GPU 只能空载等待数据, 造成了计算资源的浪费
- 使用 `tf.data.Dataset.prefetch()` 方法进行数据预加载后的训练流程
- `tf.data.Dataset.prefetch()` 可以让数据集对象 `Dataset` 在训练时预先取出若干个元素, 
   使得在 GPU 训练的同时 CPU 可以准备数据, 从而提升训练流程的效率

```python
import tensorflow as tf

# data preprocessing function
def rot90(image, label):
   image = tf.image.rot90(image)
   return image, label

# data
mnist_dataset = tf.keras.datasets.mnist.load_data()

# data preprocessing
# 开启数据预加载功能
mnist_dataset = mnist_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
# 利用多 GPU 资源, 并行化地对数据进行变换
mnist_dataset = mnist_dataset.map(map_func = rot90, num_parallel_calls = 2)
mnist_dataset = mnist_dataset.map(map_func = rot90, num_parallel_calls = tf.data.experimental.AUTOTUNE)
```

- (5)获取与使用 `tf.data.Dataset` 数据集元素

- 构建好数据并预处理后, 需要从中迭代获取数据用于训练

```python
dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
for a, b, c ... in dataset:
   pass
```

```python
dataset = tf.data.Dataset.from_tensor_slices((A, B, C, ...))
it = iter(dataset)
a_0, b_0, c_0, ... = next(it)
a_1, b_1, c_1, ... = next(it)
```

### 4.2 图像

keras.preprocessing.imgae.ImageDataGenerator 通过实时数据增强生成批量张量图像数据

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center = False, # 将数据的特征均值设定为0
      samplewise_center = False,  # 将数据的样本均值设定为0
      featurewise_std_normalization = False, # 是否将特征除以特征的标准差进行归一化
      samplewise_std_normalization = False,  # 是否将样本除以样本的标准差进行归一化
      zca_whitening = False, # 是否进行 ZCA 白化
      zca_epsilon = 1e-06,   # 进行 ZCA 白化的epsilon参数
      rotation_range = 0,      # 随机旋转的角度范围
      width_shift_range = 0.0, # 宽度调整的范围
      height_shift_range = 0.0,# 高度调整的范围
      brightness_range = None, # 亮度范围 
      shear_range = 0.0,         # 剪切范围
      zoom_range = 0.0,          # 缩放范围
      channel_shift_range = 0.0, # 通道调整范围
      fill_mode = 'nearest',     # 填充边界之外点的方式:
      cval=0.0, 
      horizontal_flip=False,  # 水平翻转
      vertical_flip=False,    # 垂直翻转
      rescale=None,           # 
      preprocessing_function=None, 
      data_format=None, 
      validation_split=0.0,
      dtype=None)
```

**用法:**

```python
from keras.datasets import cifar10
from keras import utils
from keras.preprocessing.image import ImageDataGenerator

# model training parameters
num_classes = 10
data_augmentation = True
batch_size = 32
epochs = 20

# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes = num_classes)
y_test = utils.to_categorical(y_test, num_classes = num_classes)

# model training
if not data_augmentation:
      print("Not using data augmentation.")
      model.fit(x_train, y_train,
               batch_size = batch_size,
               epochs = epochs,
               validation_data = (x_test, y_test),
               shuffle = True)
else:
      print("Using real-time data augmentation.")
      # This will do preprocessing and realtime data augmentation:
      datagen = ImageDataGenerator(
         featurewise_center = False,
         samplewise_center = False,
         featurewise_std_normalization = False,
         samplewise_std_normalization = False,
         zca_whitening = False,
         zca_epsilon = 1e-6,
         rotation_range = 0,
         width_shift_range = 0.1,
         height_shift_range = 0.1,
         shear_range = 0.,
         zoom_range = 0.,
         channel_shift_range = 0,
         fill_mode = "nearest",
         cval = 0.,
         horizontal_flip = True,
         vertical_flip = False,
         rescale = None,
         preprocessing_function = None,
         data_format = None,
         validation_split = 0.0
      )
      datagen.fit(x_train)
      model.fit_generator(datagen.flow(x_train,
                                       y_train,
                                       batch_size = batch_size,
                                       epochs = epochs,
                                       validation_data = (x_test, y_test),
                                       workers = 4))
```

```python
from keras.datasets import cifar10
from keras import utils


# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes = num_classes)
y_test = utils.to_categorical(y_test, num_classes = num_classes)


# model training parameters
batch_size = 32
epochs = 20
num_classes = 10
data_augmentation = True

# model training
datagen = ImageDataGenerator(featurewise_center = True,
                              featurewise_std_normalization = True,
                              rotation_range = 20,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              horizontal_flip = True)

for e in range(epochs):
      print("Epoch", e)
      batches = 0
      for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size = batch_size):
         model.fit(x_batchd, y_batch)
         batches += 1
         if batches >= len(x_train) / 32:
            break
```



```python
train_datagen = ImageDataGenerator(rescale = 1. / 255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1.0 / 255)

train_generator = train_datagen \
      .flow_from_directory("data/train",
                           target_size = (150, 150),
                           batch_size = 32,
                           class_mode = "binary")
validation_generator = test_datagen \
      .flow_from_directory("data/validation",
                           target_size = (150, 150),
                           batch_size = 32,
                           class_mode = "binary")

model.fit_generator(train_generator,
                     steps_per_epoch = 2000,
                     epochs = 50,
                     validation_data = validation_generator,
                     validation_steps = 800)
```

```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
      'data/images',
      class_mode=None,
      seed=seed)

mask_generator = mask_datagen.flow_from_directory(
      'data/masks',
      class_mode=None,
      seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
      train_generator,
      steps_per_epoch=2000,
      epochs=50)
```


### 4.3 文本

`tf.data.TextLineDataset` 通常被用来以文本文件构建数据集(原文件中的一行为一个样本)。
这适用于大多数的基于行的文本数据(例如, 诗歌或错误日志)。

- 删除文档的页眉、页脚、行号、章节标题

```python
import os
import tensorflow as tf
import tensorflow_datasets as tfds

DIRECTORY_URL = "https://storage.googleapis.com/download.tensorflow.org/data/illiad/"
FILE_NAMES = ["cowper.txt", "derby.txt", "butler.txt"]
for name in FILE_NAMES:
   text_dir = tf.keras.utils.get_file(name, origin = DIRECTORY_URL + name)

def labeler(example, index):
   return example, tf.cast(index, tf.int64)

parent_dir = os.path.dirname(text_dir)
labeled_data_sets = []
for i, file_name in enumerate(FILE_NAMES):
   lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
   labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
   labeled_data_sets.append(labeled_dataset)

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
   all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration = False)

for ex in all_labeled_data.take(5):
   print(ex)
```

### 4.4 CSV

### 4.5 Numpy

### 4.6 pandas.DataFrame

### 4.7 Unicode

### 4.8 TF.Text

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
```

### 4.9 TFRecord

1. TFRecord 数据文件介绍

TFRecord 是 TensorFlow 中的数据集存储格式。当将数据集整理成 TFRecord 格式后, 
TensorFlow 就可以高效地读取和处理这些数据集了。从而帮助更高效地进行大规模模型训练。

TFRecord 可以理解为一系列序列化的 `tf.train.Example` 元素所组成的列表文件, 
而每一个 `tf.train.Example` 又由若干个 `tf.train.Feature` 的字典组成:

```python

   # dataset.tfrecords
   [
      {  # example 1 (tf.train.Example)
         'feature_1': tf.train.Feature,
         ...
         'feature_k': tf.train.Feature,
      },
      ...
      {  # example N (tf.train.Example)
         'feature_1': tf.train.Feature,
         ...
         'feature_k': tf.train.Feature,
      }, 
   ]
```

2. TFRecord 文件保存

- TFRecord 文件保存步骤

为了将形式各样的数据集整理为 TFRecord 格式, 可以对数据集中的每个元素进行以下步骤:

- (1) 读取该数据元素到内存
- (2) 将该元素转换为 `tf.train.Example` 对象
   - 每个 `tf.train.Example` 对象由若干个 `tf.train.Feature` 的字典组成, 因此需要先建立 Feature 的子典
- (3) 将 `tf.train.Example` 对象序列化为字符串, 并通过一个预先定义的 `tf.io.TFRecordWriter` 写入 `TFRecord` 文件

- TFRecord 文件保存示例

```python
import tensorflow as tf
import os

# root
root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
# project
project_path = os.path.join(root_dir, "deeplearning/src/tensorflow_src")
# model save
models_path = os.path.join(project_path, "save")
# data
cats_and_dogs_dir = os.path.join(root_dir, "datasets/cats_vs_dogs")
data_dir = os.path.join(root_dir, "datasets/cats_vs_dogs/cats_and_dogs_small")
# train data
train_dir = os.path.join(data_dir, "train")
train_cats_dir = os.path.join(train_dir, "cat")
train_dogs_dir = os.path.join(train_dir, "dog")
# tfrecord
tfrecord_file = os.path.join(cats_and_dogs_dir, "train.tfrecord")

# 训练数据
train_cat_filenames = [os.path.join(train_cats_dir, filename) for filename in os.listdir(train_cats_dir)]
train_dog_filenames = [os.path.join(train_dogs_dir, filename) for filename in os.listdir(train_dogs_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)

# 迭代读取每张图片, 建立 tf.train.Feature 字典和 tf.train.Example 对象, 序列化并写入 TFRecord
with tf.io.TFRecordWriter(tfrecord_file) as writer:
   for filename, label in zip(train_filenames, train_labels):
      # 读取数据集图片到内存, image 为一个 Byte 类型的字符串
      image = open(filename, "rb").read()
      # 建立 tf.train.Feature 字典
      feature = {
            # 图片是一个 Byte 对象
            "image": tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
            "label": tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
      }
      # 通过字典建立 Example
      example = tf.train.Example(features = tf.train.Features(feature = feature))
      # 将 Example 序列化并写入 TFRecord 文件
      writer.write(example.SerializeToString())
```


3. TFRecord 文件读取

- TFRecord 数据文件读取步骤
    - (1)通过 `tf.data.TFRecordDataset` 读入原始的 TFRecord 文件, 获得一个 `tf.data.Dataset` 数据集对象
    - 此时文件中的 `tf.train.Example` 对象尚未被反序列化
    - (2)通过 `tf.data.Dataset.map` 方法, 对该数据集对象中的每个序列化的 `tf.train.Example` 字符串
    执行 `tf.io.parse_single_example` 函数, 从而实现反序列化

- TFRecord 数据文件读取示例

```python
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# root
root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
# data
cats_and_dogs_dir = os.path.join(root_dir, "datasets/cats_vs_dogs")
# tfrecord
tfrecord_file = os.path.join(cats_and_dogs_dir, "train.tfrecord")

def _parse_example(example_string):
   """
   将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
   """
   # 定义 Feature 结构, 告诉解码器每个 Feature 的类型是什么
   feature_description = {
      "image": tf.io.FixedLenFeature([], tf.string),
      "label": tf.io.FixedLenFeature([], tf.int64)
   }
   feature_dict = tf.io.parse_single_example(example_string, feature_description)
   # 解码 JPEG 图片
   feature_dict["image"] = tf.io.decode_jpeg(feature_dict["image"])
   return feature_dict["image"], feature_dict["label"]

# 读取 TFRecord 文件
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = raw_dataset.map(_parse_example)

for image, label in dataset:
   plt.title("cat" if label == 0 else "dog")
   plt.imshow(image.numpy())
   plt.show()
```

### 4.10 tf.io 的其他格式

### 4.11 tf.TensorArray


1. tf.TensorArray 介绍

在部分网络结构中, 尤其是涉及时间序列的结构中, 可能需要将一系列张量以数组的方式依次存放起来, 以供进一步处理。

- 在即时执行模式下, 可以直接使用一个 Python 列表存放数组
- 如果需要基于计算图的特性, 例如使用 @tf.function 加速模型运行或者使用 SaveModel 导出模型, 就无法使用 Python 列表了

TensorFlow 提供了 `tf.TensorArray` (TensorFlow 动态数组) 支持计算图特性的 TensorFlow 动态数组.

- 声明方式如下:

   - `arr = tf.TensorArray(dtype, size, dynamic_size = False)`: 

      - 声明一个大小为 `size`, 类型为 `dtype` 的 `TensorArray arr`
      - 如果将 `dynamic_size` 参数设置为 True, 则该数组会自动增长空间

- 读取和写入的方法如下:

   - `write(index, value)`: 将 value 写入数组的第 index 个位置
   - `read(index)`: 读取数组的第 index 个值
   - `stack()`
   - `unstack()`

2. tf.TensorArray 介绍

```python
import tensorflow as tf

@tf.function
def array_write_and_read():
   arr = tf.TensorArray(dtype = tf.float32, size = 3)
   arr = arr.write(0, tf.constant(0.0))
   arr = arr.write(1, tf.constant(1.0))
   arr = arr.write(2, tf.constant(2.0))
   arr_0 = arr.read(0)
   arr_1 = arr.read(1)
   arr_2 = arr.read(2)
   return arr_0, arr_1, arr_2

a, b, c = array_write_and_read()
print(a, b, c)
```

.. note:: 

- 由于需要支持计算图, `tf.TensorArray` 的 `write()` 是不可以忽略左值的, 
  也就是说, 在图执行模式下, 必须按照以下的形式写入数组, 才可以正常生成一个计算图操作, 
  并将该操作返回给 `arr`:

```python
arr.write(index, value)
```

- 不可以写成

```python
arr.write(index, value)
```

## 6.数据输入流水线

### 6.1 tf.data

### 6.2 优化流水线性能

### 6.3 分析流水线性能



# 模型构建 [TODO 完善]

- Sequential API
- Functional API
- Subclassing API

## 1.模型共有的方法和属性

```python
from tf.keras.model import Model
from tf.keras.model import model_from_json, model_from_yaml
```

- model.layers
- model.inputs
- model.outputs
- model.summary()
- Config
   - model.get_config()
      - Model.from_config()
      - Sequential.from_config()
- Weights
   - model.get_weights()
      - *to Numpy arrays*
   - model.set_weights(weights)
      - *from Numpy arrays*
   - model.save_weights(filepath)
      - *to HDF5 file*
   - model.loadweights(filepath, byname = False)
      - *from HDF5 file*
- Save or Load
   - model.to_json()
      - modelfromjson()
   - modeltoyaml()
      - modelfromyaml()

## 2.Sequential API

Sequential 模型是层(layers)的线性堆叠

```python
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model
model = models.Sequential()
model.add(layers.Dense(units = 64, activation = "relu"))
model.add(layers.Dense(units = 10, activation = "softmax"))
model.compile(
   loss = "categorical_crossentropy",
   optimizer = "sgd",
   metrics = ["accuracy"]
)
model.fit(x_train, y_train, epochs = 5, batch_size = 32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size = 128)
classes = model.predict(x_test, batch_size = 128)
```

## 3.Functional API

* Keras 函数式 API 是定义复杂模型的方法
* Keras 函数式 API 可以重用经过训练的模型, 可以通过在张量上调用任何模型并将其视为一个层(layers)
  - 调用模型的结构
  - 调用模型的权重

函数式 API 特点

* 所有模型都像层(layer)一样可以调用
* 多输入和多输出模型
* 共享图层
* "层节点"概念


```python
inputs = tf.keras.Input(shape = (28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units = 100, activation = tf.nn.relu)(x)
x = tf.keras.layers.Dense(units = 10)(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs = inputs, outputs = outputs)
```

## 4.Subclassing API

- 使用 Subclassing API 建立模型, 即对 `tf.keras.Model` 类进行扩展以定义自己的新模型
- 实现 forward pass in the ``call`` method
- 模型的 layers 定义在 ``__init__(self, ...)`` 中
- 模型的前向传播定义在 ``call(self, inputs)`` 中
- 可以通过调用制定的自定义损失函数 ``self.add_loss(loss_tensor)``
- 在 subclassing 模型中, 模型的拓扑结构被定义为 Python 代码, 而不是 layers 的静态图, 
  因此无法检查或序列化模型的拓扑结构, 即以下方法不适用于 subclassing 模型:
   - model.inputs
   - model.outputs
   - model.to_yaml()
   - model.to_json()
   - model.get_config()
   - model.save()
- 模型(keras.model.Model)子类的 API 可以为实现更加复杂的模型提供了灵活性, 但是是有代价的, 除了以上的功能不能使用, 并且模型更复杂, 更容易出错


```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        
        # 此处添加初始化的代码(包含call方法中会用到的层)例如:
        self.layer1 = tf.keras.layers.BuildInLayer()
        self.layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码(处理输入并返回输出), 例如:
        x = layer1(input)
        self.output = layer2(x)
        return output

model = MyModel()

with tf.GradientTape() as tape:
    logits = model(images)
    loss_value = loss(logits, labels)
grads = tape.gradient(loss_value, model.trainable_variables)
optimizer.apply(zip(grads, model.trainable_variables))
```

## 5.回调函数-Callbacks

- 回调函数是一个函数的集合, 会在训练的阶段使用
- 可以使用回调函数查看训练模型的内在状态和统计。
  也可以传递一个列表的回调函数(作为 `callbacks` 关键字参数)到 `Sequential` 或 `Model` 类型的 `.fit()` 方法。
  在训练时, 相应的回调函数的方法会被在各自的阶段被调用

回调函数API:

- keras.callbacks.Callback()
   - 用来创建新的回调函数的抽象基类
   - `.params`
   - `.model`
- keras.callbacks.BaseLogger(stateful_metrics = None)
   - 基类训练 epoch 评估值的均值
- keras.callbacks.TerminateOnNaN()
   - 当遇到损失为 `NaN` 停止训练
- keras.callbacks.ProgbarLogger()
- keras.callbacks.History()
   - 所有事件都记录到 History 对象
- keras.callbacks.ModelCheckpoint()
   - 在每个训练期之后保存模型
- keras.callbacks.EarlyStopping()
- keras.callbacks.RemoteMonitor()
- keras.callbacks.LearningRateScheduler(schedule, verbose = 0)
- keras.callbacks.TensorBoard()
- keras.callbacks.ReduceLROnPlateau()
- keras.callbacks.CSVLogger()
- keras.callbacks.LambdaCallback()

创建回调函数:

```python
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

# 模型建立
model = Sequenital()
model.add(Dense(10, input_dim = 784, kernel_initializer = "uniform"))
model.add(Activation("softmax"))

# 模型编译
model.compile(loss = "categorical_crossentropy", optimizer = "rmsporp")

# 模型训练
# 在训练时, 保存批量损失值
class LossHistory(keras.callbacks.Callback):
      def on_train_begin(self, logs = {}):
         self.losses = []

      def on_batch_end(self, batch, logs = {}):
         self.losses.append(logs.get("loss"))
history = LossHistory()

# 如果验证集损失下降, 在每个训练 epoch 后保存模型
checkpointer = ModelCheckpoint(filepath = "/tmp/weight.hdf5",
                               verbose = 1,
                               save_best_only = True)
model.fit(x_train, 
         y_train, 
         batch_size = 128, 
         epochs = 20, 
         verbose = 0,
         validation_data = (x_test, y_test), 
         callbacks = [history, checkpointer]
)

# 模型结果输出
print(history.losses)
```






# Applications[TODO 完善]

## 1.目前可用模型

Keras Applications(`keras.applications`) 提供了预训练好的深度学习模型, 
这些模型可以用于预测、特征提取等. 当初始化一个模型时就会自动下载, 
默认下载的路径是: `~/.keras.models/`.

在 ImageNet 数据上预训练过的用于图像分类的模型

- Xception
- VGG16
- VGG19
- ResNet, ResNetV2, ResNeXt
- InceptionV3
- InceptionResNet2
- MobileNet
- MobileNetV2
- DenseNet
- NASNet

```python
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2

# channels_last only; 299x299
xception_model = Xception(include_top = True,
                           weights = "imagenet",
                           input_tensor = None, 
                           input_shape = None,
                           pooling = None,
                           classes = 1000)
# channels_first and channels_last; 224x224
vgg16_model = VGG16(include_top = True,
                     weights = "imagenet",
                     input_tensor = None, 
                     input_shape = None,
                     pooling = None,
                     classes = 1000)
vgg19_model = VGG19(include_top = True, 
                     weights = 'imagenet',
                     input_tensor = None, 
                     input_shape = None, 
                     pooling = None, 
                     classes = 1000)
resnet50_model = ResNet50(include_top = True, 
                           weights = 'imagenet', 
                           input_tensor = None, 
                           input_shape = None, 
                           pooling = None, 
                           classes = 1000)
inception_v3_model = InceptionV3(include_top = True, 
                                 weights = 'imagenet', 
                                 input_tensor = None, 
                                 input_shape = None, 
                                 pooling = None, 
                                 classes = 1000)
inception_resnet_v2_model = InceptionResNetV2(include_top = True, 
                                                weights = 'imagenet', 
                                                input_tensor = None, 
                                                input_shape = None, 
                                                pooling = None, 
                                                classes = 1000)
mobilenet_model = MobileNet(input_shape = None, 
                              alpha = 1.0, 
                              depth_multiplier = 1, 
                              dropout = 1e-3, 
                              include_top = True, 
                              weights = 'imagenet', 
                              input_tensor = None, 
                              pooling = None, 
                              classes = 1000)
densenet_model = DenseNet121(include_top = True, 
                              weights = 'imagenet', 
                              input_tensor = None, 
                              input_shape = None, 
                              pooling = None, 
                              classes = 1000)
densenet_model = DenseNet169(include_top = True, 
                              weights = 'imagenet', 
                              input_tensor = None, 
                              input_shape = None, 
                              pooling = None, 
                              classes = 1000)
densenet_model = DenseNet201(include_top = True, 
                              weights = 'imagenet', 
                              input_tensor = None, 
                              input_shape = None, 
                              pooling = None, 
                              classes = 1000)
nasnet_model = NASNetLarge(input_shape = None, 
                           include_top = True, 
                           weights = 'imagenet', 
                           input_tensor = None, 
                           pooling = None, 
                           classes = 1000)
nasnet_model = NASNetMobile(input_shape = None, 
                              include_top = True, 
                              weights = 'imagenet', 
                              input_tensor = None, 
                              pooling = None, 
                              classes = 1000)
mobilenet_v2_model = MobileNetV2(input_shape = None, 
                                 alpha = 1.0, 
                                 depth_multiplier = 1, 
                                 include_top = True, 
                                 weights = 'imagenet', 
                                 input_tensor = None, 
                                 pooling = None, 
                                 classes = 1000)
```

## 2.示例

- 图像分类模型使用示例

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





# Utils

## 1.模型可视化

### 1.1 `plot_model()`

- Converts a Keras model to dot format and save to a file.

```python

import tensorflow as tf

tf.keras.utils.plot_model(
    model,
    to_file = "model.png",
    show_shapes = False,
    show_dtype = False,
    show_layer_names = True,
    rankdir = "TB",
    expand_nested = False,
    dpi = 96,
)
```

### 1.2 `model_to_dot()`

- Convert a Keras model to dot format.

```python
import tensorflow as tf

tf.keras.utils.model_to_dot(
    model,
    show_shapes = False,
    show_dtype = False,
    show_layer_names = True,
    rankdir = "TB",             # "TB": a vertical plot; "LR": a horizontal plot
    expand_nested = False,
    dpi = 96,
    subgraph = False,
)
```

## 2.序列化工具(Serialization utilities)

- custom_object_scope()
- get_custom_objects()
- register_keras_serializable()
- serialize_keras_object()
- daserialize_keras_object()

### 2.1 `CustomObjectScope` class

- 作用

    - 将自定义类/函数 暴露给 Keras 反序列化内部组件
    - 在范围 `with custom_object_scope(object_dict)`, Keras 方法将能够反序列化已保存的配置引用的任何自定义对象

- 语法

```python
import tensorflow as tf

tf.keras.utils.custom_object_scope(*args)
```

- 示例

```python
# 一个自定义的正则化器 `my_regularizer`
my_regularizer = None

# a layer
layer = Dense(3, kernel_regularizer = my_regularizer)

# Config contains a reference to "my_regularizer"
config = layer.get_config()
...

# Later
with custom_object_scope({"my_regularizer": my_regularizer}):
    layer = Dense.from_config(config)
```

### 2.2 get_custom_objects()

- 作用

    - 额, 下次一定

- 语法

```python
import tensorflow as tf

tf.keras.utils.get_custom_objects()
```

- 示例

```python
get_custom_objects().clear()
get_custom_objects()["MyObject"] = MyObject
```

### 2.3 register_keras_serializable()

- 作用

    - 额, 下次一定

- 语法

```python
import tensorflow as tf

tf.keras.utils.register_keras.serializable(package = "Custom", name = None)
```

### 2.4 serialize_keras_object()

- 作用

    - 将 Keras 对象序列化为 Json 兼容的表示形式

- 语法

```python
import tensorflow as tf

tf.keras.utils.serialize_keras_object(instance)
```

### 2.5 daserialize_keras_object()

- 作用

    - 将 Keras 对象的序列化形式转换回实际对象

- 语法

```python
import tensorflow as tf

tf.keras.utils.deserialize_keras_object(
    identifier, 
    module_objects = None,
    custom_objects = None,
    printable_module_name = "object"
)
```

## 3.Python & Numpy utilities

### 3.1 `to_categorical()`

- 作用

    - 将一个类别型向量(整数)转换为 二元类别矩阵
    - 类似于 one-hot

- 语法

```python
import tensorflow as tf

utils.to_categorical(y,
                    num_classes = None,
                    dtypes = "float32")
```

- 示例

```python
# example 1
a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes = 4)
a = tf.constant(a, shape = [4, 4])
print(a)

# example 2
b = tf.constant([.9, .04, .03, .03,
                    .3, .45, .15, .13,
                    .04, .01, .94, .05,
                    .12, .21, .5, .17],
                    shape = [4, 4])
loss = tf.keras.backend.categorical_crossentropy(a, b)
print(np.around(loss, 5))

# example 3
loss = tf.keras.backend.categorical_crossentropy(a, a)
print(np.around(loss, 5))
```


### 3.2 `normalize()`

- 作用
    
    - 标准化一个 Numpy 数组

- 语法

```python
import tensorflow as tf

tf.keras.utils.normalize(x, axis = -1, order = 2)
```

### 3.3 `get_file()`

- 作用

    - Downloads a file from a URL if it not already in the cache.
    - By default the file at the url `origin` is downloaded to the cache_dir `~/.keras`, 
      placed in the cache_subdir datasets, and given the filename `fname`. 
      The final location of a file `example.txt` would therefore be `~/.keras/datasets/example.txt`.
    - Files in tar, tar.gz, tar.bz, and zip formats can also be extracted. 
      Passing a hash will verify the file after download. 
      The command line programs shasum and sha256sum can compute the hash.

- 语法

```python

tf.keras.utils.get_file(
    fname,
    origin,
    untar=False,
    md5_hash=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    cache_dir=None,
)
```

- 示例

```python

import tensorflow

path_to_downloaded_file = tf.keras.utils.get_file(
    "flower_photos",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    untar = True
)
```

### 3.4 `Progbar` class

- 作用

    - 显示进度条

- 语法

```python
import tensorflow as tf

tf.keras.utils.Progbar(
    target, 
    width = 30, 
    verbose = 1, 
    interval = 0.05, 
    stateful_metrics = None, 
    unit_name = "step"
)
```

### 3.5 `Sequence` class

- 作用

    - 用于拟合数据序列(如数据集)的基础对象
    - 每个人都Sequence必须实现__getitem__和__len__方法。如果您想在各个时期之间修改数据集, 则可以实现 on_epoch_end。该方法__getitem__应返回完整的批次
    - Sequence是进行多处理的更安全方法。这种结构保证了网络在每个时期的每个样本上只会训练一次, 而生成器则不会

- 语法

```python
import tensorflow as tf
tf.keras.utils.Sequence()
```

- 示例

```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
            for file_name in batch_x]), np.array(batch_y)
```








# 模型编译

```python
model.compile(loss, optimizer, metrics)
```

## 1.损失函数

   - Loss Function
   - Objective Function
   - Optimization score Function

**回归:**

```python
from keras import losses

# 回归 
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error
from keras.losses import mean_absolute_percentage_error
from keras.losses import mean_squared_logarithmic_error
from keras.losses import squared_hinge
from keras.losses import hinge
from keras.losses import categorical_hinge
from keras.losses import logcosh

model.Compile(loss = ["mse", "MSE", mean_squared_error], 
            optimizer, 
            metircs)
model.Compile(loss = ["mae", "MAE", mean_absolute_error], 
            optimizer, 
            metircs)
model.Compile(loss = ["mape", "MAPE", mean_absolute_percentage_error], 
            optimizer, 
            metircs)
model.Compile(loss = ["msle", "MLSE", mean_squared_logarithmic_error], 
            optimizer, 
            metircs)
```

**分类:**

```python
# 分类
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy
from keras.losses import binary_crossentropy
from keras.losses import kullback_leibler_divergence
from keras.losses import poisson
from keras.losses import cosine_proximity

model.Compile(loss = ["kld", "KLD", kullback_leibler_divergence], 
            optimizer, 
            metircs)
model.Compile(loss = ["cosine", cosine_proximity], 
            optimizer, 
            metircs)
```



The purpose of loss functions is to compute the quantity that a model 
should seek to minimize during training.

### 1.1 常用损失函数

- class handle
    - 可以传递配置参数
- function handle

1. 概率损失(Probabilistic losses)

- `BinaryCrossentropy` class
    - `binary_crossentropy()` function
- `CategoricalCrossentropy` class
    - `categorical_crossentropy()` function
- `SparseCategoricalCrossentropy` class
    - `sparse_categorical_crossentropy()` function
- `Possion` class
    - `possion()` function
- `KLDivergence` class
    - `kl_divergence()` function

class & function() 使用方法

- 作用
- 二分类损失函数
    - BinaryCrossentropy & binary_crossentropy
    - Computes the cross-entropy loss between true labels and predicted labels.
- 二分类、多分类
    - CategoricalCrossentropy & categorical_crossentropy
    - SparseCategoricalCrossentropy & sparse_categorical_crossentropy
- 其他
- 语法

```python
tf.keras.losses.Class(
    from_loits = False, 
    label_smoothing = 0, 
    reduction = "auto", 
    name = ""
)
```

- 示例

```python
# data
y_ture = [[0., 1.], [0., 0.]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]

# reduction="auto" or "sum_over_batch_size"
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred).numpy()

# reduction=sample_weight
bce = tf.keras.losses.BinaryCrossentropy()
bce(y_true, y_pred, sample_weight = [1, 0]).numpy()

# reduction=sum
bce = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.SUM)
bce(y_true, y_pred).numpy()

# reduction=none
bce = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
bce(y_true, y_pred).numpy()
```

2. 回归损失(Regression losses)

- `MeanSquaredError` class
    - `mean_squared_error` function 
- `MeanAbsoluteError` class
    - `mean_absolute_error` function
- `MeanAbsolutePercentageError` class
    - `mean_absolute_percentage_error` function
- `MeanSquaredLogarithmicError` class
    - `mean_squared_logarithmic_error` function
- `CosineSimilarity` class
    - `cosine_similarity` function
- `Huber` class
    - `huber` function
- `LogCosh` class
    - `log_cosh` function

3. Hinge losses for "maximum-margin" classification

- `Hinge` class
    - `hinge` function
- `SquaredHinge` class
    - `squared_hinge` function
- `CategoricalHinge` class
    - `categorical_hinge` function

### 1.2 损失函数的使用——compile() & fit()

- 通过实例化一个损失类创建损失函数, 可以传递配置参数

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer = "uniform", input_shape = (10,)))
model.add(layers.Activation("softmax"))

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
    optimizer = "adam", 
    metrics = ["acc"]
)
```

- 直接使用损失函数

```python
from tensorflow.keras.losses import sparse_categorical_crossentropy

model.compile(
    loss = "sparse_categorical_crossentropy", 
    optimizer = "adam", 
    metrics = ["acc"]
)
```

### 1.3 损失函数的使用——单独使用

```python
tf.keras.losses.mean_squared_error(tf.ones((2, 2)), tf.zeros((2, 2)))
loss_fn = tf.keras.losses.MeanSquaredError(resuction = "sum_over_batch_size")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError(reduction = "sum")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError(reduction = "none")
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.mean_squared_error
loss_fn(tf.ones((2, 2,)), tf.zeros((2, 2)))

loss_fn = tf.keras.losses.MeanSquaredError()
loss_fn(tf.ones((2, 2)), tf.zeros((2, 2)))
```

### 1.4 创建自定义损失函数

- Any callable with the signature `loss_fn(y_true, y_pred)` that returns an array of 
    losses (one of sample in the input batch) can be passed to compile() as a loss. 
- Note that sample weighting is automatically supported for any such loss.

示例:

```python
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis = -1)

model.compile(optimizer = "adam", loss = my_loss_fn)
```

1. `add_loss()` API

```python
from tensorflow.keras.layers import Layer

class MyActivityRegularizer(Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, rate = 1e-2):
        super(MyActivityRegularizer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))

        return inputs

from tensorflow.keras import layers

class SparseMLP(Layer):
    """Stack of Linear layers with a sparsity regularization loss."""

    def __init__(self, output_dim):
        super(SparseMLP, self).__init__()
        self.dense_1 = layers.Dense(32, activation=tf.nn.relu)
        self.regularization = MyActivityRegularizer(1e-2)
        self.dense_2 = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.regularization(x)
        return self.dense_2(x)

mlp = SparseMLP(1)
y = mlp(tf.ones((10, 10)))

print(mlp.losses)  # List containing one float32 scalar

mlp = SparseMLP(1)
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1  # No accumulation.
```

- 自定义损失函数需要继承 `tf.keras.losses.Loss` 类, 重写 `call` 方法即可, 
  输入真实值 `y_true` 和模型预测值 `y_pred`, 输出模型预测值和真实值之间通
  过自定义的损失函数计算出的损失值

```python  
import numpy as np
import tensorflow as tf

class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
```

## 2.评价指标

- Metric 是一个评估模型表现的函数
- Metric 函数类似于一个损失函数, 只不过模型评估返回的 metric
  不用来训练模型, 因此, 可以使用任何损失函数当做一个 metric 函数使用

### 2.1 metrics

API:

```python
from keras import metrics
from keras.metrics import binary_accuracy
from keras.metrics import categorical_accuracy
from keras.metrics import sparse_categorical_accuracy
from keras.metrics import top_k_categorical_accuracy
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.metrics import mae

from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error
from keras.losses import mean_absolute_percentage_error
from keras.losses import mean_squared_logarithmic_error
from keras.losses import squared_hinge
from keras.losses import hinge
from keras.losses import categorical_hinge
from keras.losses import logcosh
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy
from keras.losses import binary_crossentropy
from keras.losses import kullback_leibler_divergence
from keras.losses import poisson
from keras.losses import cosine_proximity
```

Metrics Name:

```python
metrics = ["acc", "accuracy"]
```

### 2.2 Accuracy metrics

- Accuracy class
- BinaryAccuracy class
- CategoricalAccuracy class
- TopKCategoricalAccuracy class
- SparseTopKCategoricalAccuracy class

### 2.3 Probabilistic metrics

- BinaryCrossentropy class
- CategoricalCrossentropy class
- SparseCategoricalCrossentropy class
- KLDivergence class
- Poisson class

### 2.4 Regression metrics

- MeanSquaredError class
- RootMeanSquaredError class
- MeanAbsoluteError class
- MeanAbsolutePercentageError class
- CosineSimilarity class
- LogCoshError class

### 2.5 Classification metrics based on True/False positives & negatives

- AUC class
- Precision class
- Recall class
- TurePositives class
- TrueNegatives class
- FalsePositives class
- FalseNegatives class
- PrecisionAtRecall class
- SensitivityAtSpecificity class
- SpecificityAtSensitivity class

### 2.6 image segmentation metrics

- MeanIoU class

### 2.7 Hinge metrics for "maximum-margin" Classification

- Hinge class
- SquaredHinge class
- CategoricalHinge class

### 2.8 评价指标的使用——compile() & fit()


### 2.9 评价指标的使用——单独使用

### 2.10 自定义评估指标

- 自定义评估指标需要继承 `tf.keras.metrics.Metric` 类, 
  并重写 `__init__`、`update_state`、`result` 三个方法

```python
import numpy as np
import tensorflow as tf

class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()
        self.total = self.add_weight(name = "total", dtype = tf.int32, initializer = tf.zeros_initializer())
        self.count = self.add_weight(name = "total", dtype = tf.int32, initializer = tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight = None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis = 1, output_type = tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total
```


```python

import keras.backend as K

def mean_pred(y_true, y_pred):
   return K.mean(y_pred)

model.compile(optimizers = "rmsprop",
            loss = "binary_accuracy",
            metrics = ["accuracy", mean_pred])
```

## 3.优化器

### 3.1 Optimizers

- SGD
- RMSprop
- Adagrad
- Adadelta
- Adam
- Adamax
- Nadam

```python
from keras import optimizers

sgd = optimizers.SGD(lr = 0.01)
model.compile(loss, optimizer = sgd)
# or
model.compile(loss, optimizer = "sgd")

rmsprop = optimizers.RMSprop(lr = 0.001)
model.compile(loss, optimizer = rmsprop)
# or
model.compile(loss, optimizer = "rmsprop")

adagrad = optimizers.Adagrad(lr = 0.01)
model.compile(loss, optimizer = adagrad)
# or
model.compile(loss, optimizer = "adagrad")

adadelta = optimizers.Adadelta(lr = 1.0)
model.compile(loss, optimizer = adadelta)
# or
model.compile(loss, optimizer = "adadelta")

adam = optimizers.Adam(lr = 0.001)
model.compile(loss, optimizer = adam)
# or
model.compile(loss, optimizer = "adam")

adamax = optimizers.Adamax(lr = 0.02)
model.compile(loss, optimizer = adamax)
# or
model.compile(loss, optimizer = "adamax")

nadam = optimizers.Nadam(lr = 0.002)
model.compile(loss, optimizer = nadam)
# or
model.compile(loss, optimizer = "nadam")
```

### 3.2 optimizder 的使用方式

(1) ``keras.optimizers`` 和 ``optimizer`` 参数

```python
from keras import optimizers

# 编译模型
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = "mean_squared_error", optimizer = sgd)
```

(2) ``optimizer`` 参数

```python
# 编译模型
model.compile(loss = "mean_squared_error", optimizer = "sgd")
```

### 3.3 optimizers 的共有参数

- control gradient clipping
   - ``clipnorm``
   - ``clipvalue``

```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr = 0.01, clipnorm = 1)

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr = 0.01, clipvalue = 0.5)
```

### 3.4 优化器的使用

1. 模型编译(compile)和拟合(fit)

```python
from tensorflow import keras
from tensorflow.keras import layers

# model
model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer = "uniform", input_shape = (10,)))
model.add(layers.Activate("softmax"))
# model compile
opt = keras.optimizers.Adam(learning_rate = 0.01)
model.compile(loss = "categorical_crossentropy", optimizer = opt)
# model.compile(loss = "categorical_crossentropy", optimizer = "adam")
```

2. 自定义迭代训练

```python
# Instantiate an optimizer
optimizer = tf.keras.optimizer.Adam()

# Iterate over the batches of a dataset.
for x, y in dataset:
# open a GradientTape
with tf.GradientTape() as tape:
    # Forward pass.
    logits = model(x)
    
    # Loss value for this batch
    loss_value = loss_fn(y, logits)

# Get gradients of loss wrt the weights
gradients = tape.gradient(loss_value, model.trainable_weights)

# Update the weights of the model
optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

3. 学习率衰减(decay)、调度(sheduling)

- 可以使用学习率时间表来调整优化器的学习率如何随时间变化
- ExponentialDecay: 指数衰减
- PiecewiseConstantDecay: 
- PolynomialDecay: 多项式衰减
- InverseTimeDecay: 逆时间衰减

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate = 1e-2,
decay_steps = 10000,
decay_rate = 0.9
)
optimizer = keras.optimizers.SGD(learning_rate = lr_schedule)
```

### 3.5 优化算法核心 API

- apply_gradients
- weights_property
- get_weights
- set_weights

1. apply_gradients

- 语法

```python
Optimizer.apply_gradients(
    grads_and_vars, name=None, experimental_aggregate_gradients=True
)
```

- 参数
  - grads_and_vars: 梯度、变量对的列表
  - name: 返回的操作的名称
  - experimental_aggregate_gradients: 

- 示例

```python
grads = tape.gradient(loss, vars)
grads = tf.distribute.get_replica_context().all_reduce("sum", grads)

# Processing aggregated gradients.
optimizer.apply_gradients(zip(grad, vars), experimental_aggregate_gradients = False)
```

2. weights_property

- 语法

```python
import tensorflow as tf

tf.keras.optimizers.Optimizer.weights
```

3. get_weights

- 语法

```python
Optimizer.get_weights()
```

- 示例

```python
# 模型优化器
opt = tf.keras.optimizers.RMSprop()

# 模型构建、编译
m = tf.keras.models.Sequential()
m.add(tf.keras.layers.Dense(10))
m.compile(opt, loss = "mse")

# 数据
data = np.arange(100).reshape(5, 20)
labels = np.zeros(5)

# 模型训练
print("Training")
results = m.fit(data, labels)
print(opt.get_weights)
```

4. set_weights

- 语法

```python
Optimizer.set_weights(weights)
```

- 示例

```python
# 模型优化器
opt = tf.keras.optimizers.RMSprop()

# 模型构建、编译
m = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
m.compile(opt, loss = "mse")

# 数据        
data = np.arange(100).reshape(5, 20)
labels = np.zeros(5)

# 模型训练
print("Training")
results = m.fit(data, labels)

# 优化器新权重
new_weights = [
    np.array(10),       # 优化器的迭代次数
    np.ones([20, 10]),  # 优化器的状态变量
    np.zeros([10])      # 优化器的状态变量
]
opt.set_weights(new_weights)
opt.iteration
```



# Keras 网络层

## 1.自定义层

- 自定义层需要继承 `tf.keras.layers.Layers` 类, 并重写 `__init__`、`build`、`call` 三个方法

```python
import numpy as np
import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # 初始化代码
    
    def build(self, input_shape): # input_shape 是一个 TensorShape 类型对象, 提供输入的形状
        # 在第一次使用该层的时候调用该部分代码, 在这里创建变量可以使得变量的形状自适应输入的形状
        # 而不需要使用者额外指定变量形状
        # 如果已经可以完全确定变量的形状, 也可以在 __init__ 部分创建变量
        self.variable_0 = self.add_weight(...)
        self.variable_1 = self.add_weight(...)
    
    def call(self, inputs):
        # 模型调用的代码(处理输入并返回输出)
        return output
```

- 线性层示例

```python
import numpy as np
import tensorflow as tf

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super.__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_variable(
            name = "w", 
            shape = [input_shape[-1], self.units],  # [n, 1]
            initializer = tf.zeros_initializer()
        )
        self.b = self.add_variable(
            name = "b",
            shape = [self.units],                   # [1]
            initializer = tf.zeros_initializer()
        )
    
    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred

class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer(untis = 1)
    
    def call(self, inputs):
        output = self.layer(inputs)
        return output
```

## 1.Keras Layers 共有的方法:

```python
from keras import layers
```

- layer.get_weights()
- layer.set_weights(weights)
- layer.get_config()
   - keras.layer.Dense.from_config(config)
   - keras.layer.deserialize({"class_name": , "config": config})
- 如果 Layer 是单个节点(不是共享 layer), 可以使用以下方式获取 layer
   的属性:
   - layer.input
   - layer.output
   - layer.input_shape
   - layer.output_shape
- 如果 Layer 具有多个节点(共享 layer), 可以使用以下方式获取 layer
   的属性:
   - layer.getinputat(note_index)
   - layer.getoutputat(note_index)
   - layer.getinputshapeat(noteindex)
   - layer.getoutputshaepat(noteindex)

## 2.Keras Layers

- **Core Layers**
   - Dense
   - Activation
   - Drop
   - Flatten
   - Input
   - Reshape
      - `keras.layers.Reshape(target_shape)`
   - Permute
   - RepeatVector
   - Lambda
   - ActivityRegularization
   - Masking
   - SpatialDropout1D
   - SpatialDropout2D
   - SpatialDropout3D`
- **Convolutional Layers**
   - 卷积层
      - Conv1D
      - Conv2D
      - Conv3D
      - SeparableConv1D
         - `keras.layers.SeparableConv1D(rate)`
      - SeparableConv2D
      - DepthwiseConv3D
   - Transpose
      - Conv2DTranspose
      - Conv3DTranspose
   - Cropping
      - Cropping1D
      - Cropping2D
      - Cropping3D
   - UnSampling
      - UnSampling1D
      - UnSampling2D
      - UnSampling3D
   - ZeroPadding
      - ZeroPadding1D
      - ZeroPadding2D
      - ZeroPadding3D
- **Pooling Layers**
   - 最大池化
      - `MaxPolling1D()`
      - `MaxPolling2D()`
      - `MaxPolling3D()`
      - `GlobalMaxPolling1D()`
      - `GlobalMaxPolling2D()`
      - `GlobalMaxPolling3D()`
   - 平均池化
      - `AveragePolling1D()`
      - `AveragePolling2D()`
      - `AveragePolling3D()`
      - `GlobalAveragePolling1D()`
      - `GlobalAveragePolling2D()`
      - `GlobalAveragePolling3D()`
- **Locally-connected Layers**
   - `LocallyConnected1D()`
   - `LocallyConnected2D()`
- **Recurrent Layers**
   - RNN
      - `RNN()`
      - `SimpleRNN()`
      - `SimpleRNNCell()`
   - GRU
      - `GRU()`
      - `GRUCell()`
   - LSTM
      - `LSTM()`
      - `LSTMCell()`
      - `ConvLSTM2D()`
      - `ConvLSTM2DCell()`
   - CuDNN
      - `CuDNNGRU()`
      - `CuDNNLSTM()`
- **Embedding Layers**
   - `Embedding()`
- **Merge Layers**

   - `Add()`
   - `Subtract()`
   - `Multiply()`
   - `Average()`
   - `Maximum()`
   - `Minimum()`
   - `Concatenate()`
   - `Dot()`
   - `add()`
   - `subtract()`
   - `multiply()`
   - `average()`
   - `maximum()`
   - `minimum()`
   - `concatenate()`
   - `dot()`
- **Advanced Activations Layers**
   - `LeakyReLU()`
   - `PReLU()`
   - `ELU()`
   - `ThresholdedReLU()`
   - `Softmax()`
   - `ReLU()`
   - Activation Functions
- **Normalization Layers**
   - `BatchNormalization()`
- **Nosise Layers**
   - `GaussianNoise()`
   - `GaussianDropout()`
   - `AlphaDropout()`
- **Others**
   - Layer wrapper
      - `TimeDistributed()`
      - `Bidirectional()`
   - Writting Customilize Keras Layers
      - `build(input_shape)`
      - `call(x)`
      - `compute_output_shape(input_shape)`


## 3.Keras Layers 配置

```python
model.add(Layer(
      # 输出、输出
      output_dim,
      input_dim,
      # 参数初始化
      kernel_initializer,
      bias_initializer,
      # 参数正则化
      kernel_regularizer,
      activity_regularizer,
      # 参数约束
      kernel_constraint,
      bias_constraint,
      # 层激活函数
      activation,
))
# 输出
input_s = Input()
# 激活函数
model.add(Activation)
```

### 3.1 Activation Function

- Keras Activations
   - `Activation` layer
   - `activation` argument supported by all forward layers

**调用方法:**

```python
from keras.layers import Activation, Dense
from keras import backend as K

# method 1
model.add(Dense(64))
model.add(Activation("tanh"))

# method 2
model.add(Dense(64, activation = "tanh"))

# method 3
model.add(Dense(64, activation = K.tanh))
```

### 3.2 可用的 activations

- softmax: Softmax activation function
   - x =>
   - `keras.activatons.softmax(x, axis = 1)`
- relu: Rectified Linear Unit
   - x => max(x, 0)
   - `keras.activations.relu(x, alpha = 0.0, max_value = None, threshold = 0.0)`
- tanh: Hyperbolic tangent activation function
   - `keras.activations.tanh(x)`
- sigmoid: Sigmoid activation function
   - x => 1/(1 + exp(-x))
   - `keras.activations.sigmoid(x)`
- linear: Linear activation function
   - x => x
   - `keras.activations.linear(x)`


### 3.3 Keras 参数初始化(Initializers)

**Initializers 的使用方法:**

   初始化定义了设置 Keras Layer 权重随机初始的方法

- `kernel_initializer` param

   - "random_uniform"

- `bias_initializer` param

**可用的 Initializers:**

- keras.initializers.Initializer()
   - 基类
- keras.initializers.Zeros()
   - `0`
- keras.initializers.Ones()
   - `1`
- keras.initializers.Constant()
   - keras.initializers.Constant(value = 0)
      - `0`
   - keras.initializers.Constant(value = 1)
      - `1`
- keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05, seed =
   None)
   - 正态分布
- keras.initializers.RandomUniform(minval = 0.05, maxval = 0.05, seed =
   None)
   - 均匀分布
- keras.initializers.TruncatedNormal(mean = 0.0, stddev = 0.05, seed =
   None)
   - 截尾正态分布:生成的随机值与 `RandomNormal`
      生成的类似, 但是在距离平均值两个标准差之外的随机值将被丢弃并重新生成。这是用来生成神经网络权重和滤波器的推荐初始化器
- keras.initializers.VarianveScaling(scale = 1.0, mode = "fan_in",
   distribution = "normal", seed = None)
   - 根据权值的尺寸调整其规模
- keras.initializers.Orthogonal(gain = 1.0, seed = None)
   - `随机正交矩阵 <http://arxiv.org/abs/1312.6120>`__
- keras.initializers.Identity(gain = 1.0)
   - 生成单位矩阵的初始化器。仅用于 2D 方阵
- keras.initializers.lecun_normal()
   - LeCun 正态分布初始化器
   - 它从以 0 为中心, 标准差为 stddev = sqrt(1 / fanin)
      的截断正态分布中抽取样本,  其中 fanin
      是权值张量中的输入单位的数量
- keras.initializers.lecun_uniform()
   - LeCun 均匀初始化器
   - 它从 [-limit, limit] 中的均匀分布中抽取样本,  其中 limit 是 sqrt(3
      / fanin),  fanin 是权值张量中的输入单位的数量
- keras.initializers.glorot_normal()
   - Glorot 正态分布初始化器, 也称为 Xavier 正态分布初始化器
   - 它从以 0 为中心, 标准差为 stddev = sqrt(2 / (fan*in + fanout))
      的截断正态分布中抽取样本,  其中 fanin
      是权值张量中的输入单位的数量,  fanout
      是权值张量中的输出单位的数量
- keras.initializers.glorot_uniform()
   - Glorot 均匀分布初始化器, 也称为 Xavier 均匀分布初始化器
   - 它从 [-limit, limit] 中的均匀分布中抽取样本,  其中 limit 是 sqrt(6
      / (fan*in + fanout)),  fanin 是权值张量中的输入单位的数量, 
      fanout 是权值张量中的输出单位的数量
- keras.initializers.he_normal()
   - He 正态分布初始化器
   - 它从以 0 为中心, 标准差为 stddev = sqrt(2 / fanin)
      的截断正态分布中抽取样本,  其中 fanin
      是权值张量中的输入单位的数量
- keras.initializers.he_uniform()
   - He 均匀分布方差缩放初始化器
   - 它从 :math:`[-limit, limit]` 中的均匀分布中抽取样本,  其中
      :math:`limit` 是 :math:`sqrt(6 / fan_in)`\ ,  其中 fan_in
      是权值张量中的输入单位的数量
- 自定义 Initializer

```python
from keras import backend as K

def my_init(shape, dtype = None):
      return K.random_normal(shape, dtype = dtype)

model.add(Dense(64, kernel_initializer = my_init))
```

### 3.4 Keras 正则化(Regularizers)

正则化器允许在优化过程中对\ `层的参数`\ 或\ `层的激活函数`\ 情况进行惩罚, 并且神经网络优化的损失函数的惩罚项也可以使用

惩罚是以层为对象进行的。具体的 API 因层而异, 但 Dense, Conv1D, Conv2D 和
Conv3D 这些层具有统一的 API

**Regularizers 的使用方法:**

- [class] keras.regularizers.Regularizer
   - [instance] `kernel_regularizer` param
   - [instance] `bias_regularizer` param
   - [instance] `activity_regularizer` param

**可用的 Regularizers:**

- keras.regularizers.l1(0.)
- keras.regularizers.l2(0.)
- keras.regularizers.l1_l2(l1 = 0.01, l2 = 0.01)
- 自定义的 Regularizer:
   - `def l1_reg: pass`

### 3.5 Keras 约束(Constraints)

`constraints` 模块的函数允许在优化期间对网络参数设置约束(例如非负性)。

约束是以层为对象进行的。具体的 API 因层而异, 但 Dense, Conv1D, Conv2D 和
Conv3D 这些层具有统一的 API

**Constraints 的使用方法:**

- kernel_constraint
- bias_constraint

**可用的 Constraints:**

- keras.constraints.MaxNorm(max_value = 2, axis = 0)
   - 最大范数权值约束
- keras.constraints.NonNeg()
   - 权重非负的约束
- keras.constraints.UnitNorm()
   - 映射到每个隐藏单元的权值的约束, 使其具有单位范数
- keras.constraints.MinMaxNorm(minvalue = 0, maxvalue = 1.0, rate
   = 1.0, axis = 0)
   - 最小/最大范数权值约束:映射到每个隐藏单元的权值的约束, 使其范数在上下界之间












# TensorFlow TensorBoard

## 1.实时查看参数变化情况

### 1.1 TensorBoard 使用介绍

1.首先, 在代码目录下建立一个文件夹, 存放 TensorBoard 的记录文件

```bash
$ mkdir tensorboard
```

2.在代码中实例化一个记录器

```python

      summary_writer =  tf.summary.create_file_writer("./tensorboard")

3.当需要记录训练过程中的参数时, 通过 `with` 语句指定希望使用的记录器, 并对需要记录的参数(一般是标量)运行:

```python
with summary_writer.as_default():
   tf.summary.scalar(name, tensor, step = batch_index)
```

4.当要对训练过程可视化时, 在代码目录打开终端

```bash
$ tensorboard --logdir=./tensorboard
```

5.使用浏览器访问命令行程序所输出的网址, 即可访问 TensorBoard 的可视化界面

- `http://计算机名称:6006`

.. note:: 

- 每运行一次 `tf.summary.scalar()`, 记录器就会向记录文件中写入一条记录
- 除了最简单的标量以外, TensorBoard 还可以对其他类型的数据, 如:图像、音频等进行可视化
- 默认情况下, TensorBoard 每 30 秒更新一次数据, 可以点击右上角的刷新按钮手动刷新
- TensorBoard 的使用有以下注意事项:
   - 如果需要重新训练, 那么删除掉记录文件夹内的信息并重启 TensorBoard, 
      或者建立一个新的记录文件夹并开启 TensorBoard, 将 `--logdir` 参数设置为新建里的文件夹
   - 记录文件夹目录许保持全英文

### 1.2 TensorBoard 代码框架

```python

# (1)实例化一个记录器
summary_writer =  tf.summary.create_file_writer("./tensorboard")

# (2)开始训练模型
for batch_index in range(num_batches):
# ...(训练代码, 将当前 batch 的损失值放入变量 loss 中)

# (3)指定记录器
with summary_writer.as_default():
   tf.summary.scalar("loss", loss, step = batch_index)
   tf.summary.scalar("MyScalar", my_scalar, step = batch_index)
```

## 2.查看 Graph 和 Profile 信息

在训练时使用 `tf.summary.trace_on` 开启 Trace, 此时 TensorFlow 会将训练时的大量信息, 
如:计算图的结构、每个操作所耗费的时间等, 记录下来。

在训练完成后, 使用 `tf.summary.trace_export` 将记录结果输出到文件。


1.使用 TensorBoard 代码框架对模型信息进行跟踪记录

```python

# (1)实例化一个记录器
summary_writer =  tf.summary.create_file_writer("./tensorboard")

# (2)开启 Trace, 可以记录图结构和 profile 信息
tf.summary.trace_on(graph = True, profiler = True)

# (3)开始训练模型
for batch_index in range(num_batches):
   # (4)...(训练代码, 将当前 batch 的损失值放入变量 loss 中)
   
   # (5)指定记录器, 将当前指标值写入记录器
   with summary_writer.as_default():
      tf.summary.scalar("loss", loss, step = batch_index)
      tf.summary.scalar("MyScalar", my_scalar, step = batch_index)

# (6)保存 Trace 信息到文件
with summary_writer.as_default():
   tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = log_dir)
```

2.在 TensorBoard 的菜单中选择 `PROFILE`, 以时间轴方式查看各操作的耗时情况, 
如果使用了 `@tf.function` 建立计算图, 也可以点击 `GRAPHS` 查看图结构


# TensorFLow Serving

## 1.TensorFLow Serving 安装


## 2.TensorFLow Serving 模型部署



## 3.在客户端调用以 TensorFLow  Serving 部署的模型

TensorFLow Serving 支持使用 gRPC 方法和 RESTful API 方法调用以 
TensorFLow Serving 部署的模型。

RESTful API 以标准的 HTTP POST 方法进行交互, 请求和回复均为 JSON 对象。为了调用服务器端的模型, 在客户端向服务器发送以下格式的请求.

- 服务器 URI: ``http://服务器地址:端口号/v1/models/模型名:predict``
- 请求内容

```json
{
    "signature_name": "需要调用的函数签名(Sequential模式不需要)",
    "instances": "输入数据"
}
```

- 回复:

```json
{
    "predictions": "返回值"
}
```




# TensorFlow SaveModel

为了将训练好的机器学习模型部署到各个目标平台(如服务器、移动端、嵌入式设备和浏览器等), 
我们的第一步往往是将训练好的整个模型完整导出(序列化)为一系列标准格式的文件。在此基础上, 
我们才可以在不同的平台上使用相对应的部署工具来部署模型文件。

TensorFlow 提供了统一模型导出格式 `SaveModel`, 这是我们在 TensorFlow 2 中主要使用的导出格式。
这样我们可以以这一格式为中介, 将训练好的模型部署到多种平台上. 

同时, 基于历史原因, Keras 的 Sequential 和 Functional 模式也有自有的模型导出格式。


## 1.tf.train.Checkpoint: 变量的保存与恢复

很多时候, 希望在模型训练完成后能将训练好的参数(变量)保存起来, 这样在需要使用模型的其他地方载入模型和参数, 
就能直接得到训练好的模型, 保存模型有很多中方式:

- Python 的序列化模块 `pickle` 存储 `model.variables`

    - 然而, TensorFlow 的变量类型 `ResourceVariable` 并不能被序列化
    - 语法:

```python
import pickle
```

### 1.1 tf.train.Checkpoint 介绍

- `tf.train.Checkpoint` 简介

TensorFlow 提供了 `tf.train.Checkpoint` 这一强大的变量保存与恢复类, 提供的方法可以保存和恢复 TensorFlow 中的大部分对象, 
比如下面类的实例都可以被保存: 

- `tf.keras.optimizer`
- `tf.Variable`
- `tf.keras.Layer`
- `tf.keras.Model`
- Checkpointable State 的对象


- `tf.train.Checkpoint` 使用方法

- 方法:

    - `save()`
    - `restore()`

- 语法:

```python
# 保存训练好的模型, 先声明一个 Checkpoint
model = TrainedModel()
checkpoint = tf.train.Checkpoint(myAwesomeModel = model, myAwesomeOptimizer = optimizer)
checkpoint.save(save_path_with_prefix)

# 载入保存的训练模型
model_to_be_restored = MyModel()  # 待恢复参数的同一模型
checkpoint = tf.train.Checkpoint(myAwesomeModel = model_to_be_restored)
checkpoint.restore(save_path_with_prefix_and_index)

# 为了载入最近的一个模型文件, 返回目录下最近一次检查点的文件名
tf.train.latest_checkpoint(save_path)
```

.. note:: 

- 参数:

    - `myAwesomeModel`: 待保存的模型 model 所取的任意键名, 在恢复变量时还将使用这一键名
    - `myAwesomeOptimizer`: 待保存的模型 optimizer 所取的任意键名, 在恢复变量时还将使用这一键名 
    - `save_path_with_prefix`: 保存文件的目录+前缀
    - `save_path_with_prefix_and_index`: 之前保存的文件目录+前缀+序号

- `checkpoint.save("./model_save/model.ckpt")`: 会在模型保存的文件夹中生成三个文件:

    - `checkpoint`
    - `model.ckpt-1.index`
    - `model.ckpt-1.data-00000-of-00001`

- `checkpoint.restore("./model/save/model.ckpt-1")`

    - 载入前缀为 `model.ckpt`、序号为 `1` 的文件来恢复模型


### 1.2 tf.train.Checkpoint 代码框架

1.train.py 模型训练阶段

```python

# 训练好的模型
model = MyModel()

# 实例化 Checkpoint, 指定保存对象为 model(如果需要保存 Optimizer 的参数也可以加入)
checkpoint = tf.train.Checkpoint(myModel = model)
manager = tf.train.CheckpointManager(checkpoint, directory = "./save", checkpoint_name = "model.ckpt", max_to_keep = 10)

# ...(模型训练代码)

# 模型训练完毕后将参数保存到文件(也可以在模型训练过程中每隔一段时间就保存一次)
if manager:
    manager.save(checkpoint_number = 100)
else:
    checkpoint.save("./save/model.ckpt")
```

2.test.py 模型使用阶段

```python

# 要使用的模型
model = MyModel()

# 实例化 Checkpoint, 指定恢复对象为 model
checkpoint = tf.train.Checkpoint(myModel = model)

# 从文件恢复模型参数
checkpoint.restore(tf.train.latest_checkpoint("./save))

# ...(模型使用代码)
```

.. note:: 

- `tf.train.Checkpoint` (检查点)只保存模型的参数, 不保存模型的计算过程, 
    因此一般用于在具有的模型源码时恢复之前训练好的模型参数。如果需要导出模型(无须源代码也能运行模型)。

## 2.使用 SaveModel 完整导出模型

作为模型导出格式的 `SaveModel` 包含了一个 TensorFlow 程序的完整信息: 不仅包含参数的权值, 还包含计算的流程(计算图)。
当模型导出为 SaveModel 文件时, 无须模型的源代码即可再次运行模型, 这使得 `SaveModel` 尤其适用于模型的分享和部署。

Keras 模型均可以方便地导出为 `SaveModel` 格式。不过需要注意的是, 因为 `SaveModel` 基于计算图, 
所以对于通过继承 `tf.keras.Model` 类建立的 Keras 模型来说, 需要导出为 `SaveModel` 格式的方法(比如 call) 都需要
使用 `@tf.function` 修饰。


语法:

```python
# 保存
tf.saved_model.save(model, "保存的目标文件夹名称")

# 载入
model = tf.saved_model.load("保存的目标文件夹名称")
```


示例:

```python

```







## 3.Keras 自有的模型导出格式

示例:

```bash
curl -LO https://raw.githubcontent.com/keras-team/keras/master/examples/mnist_cnn.py
```

```python
model.save("mnist_cnn.h5")
```


```python

import keras

keras.models.load_model("mnist_cnn.h5")
```


# TensorFlow Performance

## 1.使用 tf.function 提升性能

### 1.1 @tf.funciton: 图执行模式

虽然目前 TensorFlow 默认的即时执行模式具有灵活及易调试的特性, 但在特定的场合, 
例如追求高性能或部署模型时, 依然希望使用图执行模式, 将模型转换为高效的 TensorFlow 图模型。

TensorFlow 2 提供了 ``bashtf.function` 模块, 结合 AutoGraph 机制, 使得我们仅需加入一个简单的
`@tf.function` 修饰符, 就能轻松将模型以图执行模式运行。

### 1.2 @tf.function 基础使用方法


`@tf.function` 的基础使用非常简单, 只需要将我们希望以图执行模式运行的代码封装在一个函数内, 
并在函数前面加上 `@tf.function` 即可.


### 1.3 @tf.function 内在机制






### 1.4 AutoGraph: 将 Python 控制流转化为 TensorFlow 计算图




### 1.5 使用传统的 tf.Session


## 2.分析 TenforFlow 的性能


## 3.图优化


## 4.混合精度







# TensorFlow Estimator

- 一种可极大地简化机器学习编程的高阶TensorFlow API; 
- Estimator封装的操作:
   - 训练
   - 评估
   - 预测
   - 导出以使用
- Estimator优势:
   - 可以在本地主机上或分布式多服务器环境中运行基于 Estimator
      的模型, 而无需更改模型。此外, 可以在 CPU、GPU 或 TPU 上运行基于
      Estimator 的模型, 而无需重新编码模型
   - Estimator 简化了在模型开发者之间共享实现的过程
   - 可以使用高级直观代码开发先进的模型。简言之, 采用 Estimator
      创建模型通常比采用低阶 TensorFlow API 更简单
   - Estimator 本身在 tf.layers 之上构建而成, 可以简化自定义过程
   - Estimator 会为您构建图
   - Estimator 提供安全的分布式训练循环, 可以控制如何以及何时:
      - 构建图
      - 初始化变量
      - 开始排队
      - 处理异常
      - 创建检查点文件并从故障中恢复
      - 保存 TensorBoard 的摘要

## 1.预创建的Estimator

**预创建的 Estimator 程序的结构**

**依赖预创建的Estimator的TensorFlow程序通常包含下列四个步骤:**

1. 编写一个或多个数据集导入函数
   - 创建一个函数来导入训练集, 并创建另一个函数来导入测试集。每个数据集导入函数都必须返回两个对象:
      - 一个字典, 其中键是特征名称, 值是包含相应特征数据的张量(or Sparse Tensro); 
      - 一个包含一个或多个标签的张量; 
1. 定义特征列
   - 每个 `tf.feature_column` 都标识了特征名称、特征类型和任何输入预处理操作
2. 实例化相关的预创建的Estimator
   - LinearClassifier
4. 调用训练、评估或推理方法
   - 所有Estimator都提供训练模型的 `train` 方法

**上面步骤实现举例:**

```python
def input_fn_train(dataset):
   # manipulate dataset, extracting the feature dict and the label
   
   return feature_dict, label

def input_fn_test(dataset):
   # manipulate dataset, extracting the feature dict and the label
   
   return feature_dict, label


my_training_set = input_fn_train()
my_testing_set = input_fn_test()

population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education', 
                                                   normalizer_fn = lambda x: x - global_education_mean)

estimator = tf.estimator.LinearClassifier(
   feature_columns = [population, crime_rate, median_education],
)

estimator.train(input_fn = my_training_set, setps = 2000)
```

**预创建的 Estimator 的优势**

- 预创建的 Estimator 会编码最佳做法, 从而具有下列优势:
   - 确定计算图不同部分的运行位置以及在单台机器或多台机器上实现策略的最佳做法。
   - 事件(汇总)编写和普遍有用的汇总的最佳做法。


## 2.自定义的Estimator

- 每个
   Estimator(无论是预创建还是自定义)的核心都是其模型函数, 这是一种为训练、评估和预测构建图的方法。如果您使用预创建的
   Estimator, 则有人已经实现了模型函数。如果您使用自定义
   Estimator, 则必须自行编写模型函数。

- 推荐的工作流程:

   - 1.假设存在合适的预创建的Estimator, 使用它构建第一个模型并使用其结果确定基准; 
   - 2.使用此预创建的Estimator构建和测试整体管道, 包括数据的完整性和可靠性; 
   - 3.如果存在其他合适的预创建的Estimator, 则运行试验来确定哪个预创建的Estimator效果好; 
   - 4.可以通过构建自定义的Estimator进一步改进模型; 


## 3.从 Keras 模型创建 Estimator


- 可以将现有的Keras的模型转换为Estimator, 这样Keras模型就可以利用Estimator的优势, 比如进行分布式训练; 

```python
keras_inception_v3 = tf.keras.applications.keras_inception_v3.InceptionV3(weights = None)

keras_inception_v3.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.0001, momentum = 0.9),
                           loss = 'categorical_crossentropy',
                           metric = 'accuracy')

est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model = keras_inception_v3)

keras_inception_v3.input_names

train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x = {'input_1': train_data},
      y = train_labels,
      num_epochs = 1,
      shuffle = False
)

est_inception_v3.train(input_fn = train_input_fn, steps = 2000)
```

**API:**

从一个给定的Keras模型中构造一个Estimator实例

```python
tf.keras.estimator.model_to_estimator(
      keras_model = None,
      keras_model_path = None,
      custom_objects = None,
      model_dir = None,
      config = None
)
```














# TODO

- Keras Sequential 
- 模型假设, 网路只有一个输入和一个输出, 而且网络是层的线性堆叠; 
- 有些网络需要多个独立的输入, 有些网络则需要多个输出, 而有些网络在层与层之间具有内部分支, 这样的网络看起来像是层构成的图(graph), 而不是层的线性堆叠; 
- 多模态(multimodal)输入
- 元数据
- 文本描述
- 图片
- 预测输入数据的多个目标属性
- 类别
- 连续值
- 非线性地网络拓扑结构, 网络结构是有向无环图
- Inception 系列网络
   - 输入被多个并行的卷积分支所处理, 然后将这些分支的输出合并为单个张量; 
- ResNet 系列网络
   - 向模型中添加残差连接(residual connection), 将前面的输出张量与后面的输出张量相加, 
      从而将前面的表示重新注入下游数据流中, 这有助于防止信息处理流程中的信息损失; 

## 1.多输入模型

- Keras 函数式 API
   - 可以构建具有多个输入的模型, 通常情况下, 这种模型会在某一时刻用一个可以组合多个张量的层将不同输入分支合并, 张量组合方式可能是相加, 连接等, 比如:
- `keras.layers.add`
- `keras.layers.concatenate`
  
- 问答模型:


- 输入:
   - 自然语言描述的问题
   - 文本片段, 提供用于回答问题的信息
- 输出
   - 一个回答, 在最简单的情况下, 这个回答只包含一个词, 可以通过对某个预定义的词表做softmax得到; 

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

from tensorflow.keras.model import Model
# from keras.model import Model
from tensorflow.keras import layers, Input
# from keras import layers, Input

# =========================================================================
# 构建模型
# =========================================================================
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
# 文本片段
text_input = Input(
   shape = (None,), 
   dtype = "int32", 
   name = "text"
)
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layres.LSTM(32)(embedded_text)
# 自然语言描述的问题
question_input = Input(
   shape = (None,),
   dtype = "int32",
   name = "question"
)
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis = -1)

answer = layers.Dense(answer_vocabulary_size, activation = "softmax")(concatenated)

model = Model(inputs = [text_input, question_input], outputs = answer)

model.compile(
   optimizer = "rmsprop",
   loss = "categorical_crossentropy",
   metrics = ["acc"]
)

# =========================================================================
# 训练模型
# =========================================================================
import numpy as np
num_samples = 1000
max_length = 100
text = np.random.randint(1, text_vocabulary_size, size = (num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size = (num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size = (num_samples))

answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

model.fit([text, question], answers, epochs = 10, batch_size = 128)
model.fit(
   {
      "text": text,
      "question": question,
   },
   answers,
   epochs = 10,
   batch_size = 128
)
```


## 2.多输出模型

网络同时预测数据的不同性质

```python
from keras import layers, Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

# 输入层
posts_input = Input(shape = (None,), dtype = "int32", name = "posts")
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
# 隐藏层
x = layers.Conv1D(128, 5, activation = "relu")(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation = "relu")(x)
x = layers.Conv1D(256, 5, activation = "relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation = "relu")(x)
x = layers.Conv1D(256, 5, activation = "relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation = "relu")(x)
# 输出层
age_prediction = layers.Dense(1, name = "age")(x)
income_prediction = layers.Dense(num_income_groups, activation = "softmax", name = "income")(x)
gender_prediction = layers.Dense(1, activation = "sigmoid", name = "gender")(x)
# 构建模型
model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer = "rmsprop", loss = ["mse", "categorical_crossentropy", "binary_crossentropy"])
model.compile(
   optimizer = "rmsprop",
   loss = {
      "age": "mse",
      "income": "categorical_crossentropy",
      "gender": "binary_crossentropy"
   }
)
```

## 3.经验总结

### 3.1 机器、深度学习任务问题

- 二分类
- 多分类
- 标量回归

### 3.2 回归问题

- 回归问题使用的损失函数
   - 均方误差(MSE)
- 回归问题使用的评估指标
   - 平均绝对误差(MAE)
- 回归问题网络的最后一层只有一个单元, 没有激活, 是一个线性层, 这是回归的典型设置, 添加激活函数会限制输出范围

### 3.3 二分类问题

- 二分类问题使用的损失函数
   - 对于二分类问题的 sigmoid 标量输出, `binary_crossentropy`
- 对于二分类问题, 网络的最后一层应该是只有一个单元并使用 sigmoid 激活的 Dense 层, 网络输出应该是 0~1 范围内的标量, 表示概率值

### 3.4 数据预处理问题

- 在将原始数据输入神经网络之前, 通常需要对其进行预处理
   - 结构化数据
   - 图像数据
   - 文本数据
- 将取值范围差异很大的数据输入到神经网络中是有问题的
   - 网路可能会自动适应这种取值范围不同的数据, 但学习肯定变得更加困难
   - 对于这种数据, 普遍采用的最佳实践是对每个特征做标准化, 即对于输入数据的每个特征(输入数据矩阵中的列), 
      减去特征平均值, 再除以标准差, 这样得到的特征平均值为 0, 标准差为 1
   - 用于测试数据标准化的均值和标准差都是在训练数据上计算得到的。在工作流程中, 不能使用测试数据上计算得到的任何结果, 
      即使是像数据标准化这么简单的事情也不行
- 如果输入数据的特征具有不同的取值范围, 应该首先进行预处理, 对每个特征单独进行缩放

### 3.5 样本量问题

- 如果可用的数据很少, 使用 K 折交叉验证可以可靠地评估模型
- 如果可用的训练数据很少, 最好使用隐藏层较少(通常只有一到两个)的小型模型, 以避免严重的过拟合
   - 较小的网络可以降低过拟合

### 3.6 网络结构选择问题

- 如果可用的训练数据很少, 最好使用隐藏层较少(通常只有一到两个)的小型模型, 以避免严重的过拟合
- 如果数据被分为多个类别, 那么中间层过小可能会导致信息瓶颈

### 3.7 优化器

- 无论你的问题是什么, `rmsprop` 优化器通常都是足够好的选择


# TensorFlow Keras 后端

## 1.什么是 Keras 后端？

Keras 后端:

   Keras 是一个模型级库, 为开发深度学习模型提供了高层次的构建模块。
   它不处理诸如张量乘积和卷积等低级操作。
   
   相反, 它依赖于一个专门的、优化的张量操作库来完成这个操作, 它可以作为 Keras 的「后端引擎」。
   相比单独地选择一个张量库, 而将 Keras 的实现与该库相关联, Keras 以模块方式处理这个问题, 
   并且可以将几个不同的后端引擎无缝嵌入到 Keras 中。

目前可用的 Keras 后端:

   - TensorFlow
   - Theano
   - CNTK

## 2.从一个后端切换到另一个后端

如果您至少运行过一次 Keras, 您将在以下位置找到 Keras 配置文件. 如果没有, 可以手动创建它.

Keras 配置文件位置:

```bash
# Liunx or Mac
$ vim $HOME/.keras/keras.json

# Windows
$ vim %USERPROFILE%/.keras/keras.json
```

Keras 配置文件创建:

```bash
$ cd ~/.keras
$ sudo subl keras.json
```

也可以定义环境变量 `KERAS_BACKEND`, 不过这会覆盖配置文件 `$HOME/.keras/keras.json` 中定义的内容:

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend" 
Using TensorFlow backend.
```

当前环境的 Keras 配置文件内容:


```json
{
   "floatx": "float32",
   "epsilon": 1e-07,
   "backend": "tensorflow",
   "image_data_format": "channels_last"
}
```

自定义 Keras 配置文件:

   - 在 Keras 中, 可以加载除了 "tensorflow", "theano" 和 "cntk"
      之外更多的后端。Keras 也可以使用外部后端, 这可以通过更改 keras.json
      配置文件和 "backend" 设置来执行。 假设您有一个名为 my_module 的 Python
      模块, 您希望将其用作外部后端。keras.json 配置文件将更改如下.

      - 必须验证外部后端才能使用, 有效的后端必须具有以下函数:

         - `placeholder`
         - `variable`
         - `function`

      - 如果由于缺少必需的条目而导致外部后端无效, 则会记录错误, 通知缺少哪些条目:

         ```bash
         {
            "image_data_format": "channels_last",
            "epsilon": 1e-07,
            "floatx": "float32",
            "backend": "my_package.my_module"
         }
         ```

## 3.keras.json 详细配置

- `image_data_format`:
   - `"channels_last"`
      - (rows, cols, channels)
      - (conv*dim1, convdim2, conv_dim3, channels)
   - `"channels_first"`
      - (channels, rows, cols)
      - (channels, convdim1, convdim2, conv_dim3)
   - 在程序中返回: `keras.backend.image_data_format()`
- `epsilon`:
   - 浮点数, 用于避免在某些操作中被零除的数字模糊常量
- `floatx`:
   - 字符串: `float16`, `float32`, `float64`\ 。默认浮点精度
- `backend`:
   - 字符串: `tensorflow`, `theano`, `cntk`

## 5.Backend API

* `tf.keras.backend.clear_session()`
* `tf.keras.backend.epsilon()`
    - 返回数字表达式中使用的模糊因子的值
* `tf.keras.backend.floatx()`
    - 返回默认的 float 类型
* `tf.keras.backend.get_uid()`
* `tf.keras.backend.image_data_format()`
    - 返回设置图像数据格式约定的值
* `tf.keras.backend.is_keras_tensor()`
* `tf.keras.backend.reset_uids()`
* `tf.keras.backend.rnn()`
* `tf.keras.backend.set_epsilon()`
    - 设置数字表达式中使用的模糊因子的值
* `tf.keras.backend.set_floatx()`
    - 设置 float 类型
* `tf.keras.backend.set_image_data_format()`
    - 设置图像数据格式约定的值



# 相关资料

* 数据
    - [MNIST 数据集主页](http://yann.lecun.com/exdb/mnist/)
* 网络论文
    - []()