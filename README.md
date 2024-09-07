# Handwritten Digit Recognition

## Introduction

This project is a simple implementation of a convolutional neural network **(CNN)** for handwritten digit recognition using the **MNIST** dataset. The CNN architecture consists of three convolutional layers, followed by two fully connected layers, and a softmax activation function for classification. The project includes code for training, testing, and evaluating the model, as well as visualization tools for analyzing the model's performance.

## Requirements

```
matplotlib==3.8.0
torch==2.3.1+cu121
torchvision==0.19.0
torchvision==0.18.1+cu121
tqdm==4.65.0
```

## Model Architecture

```
Model(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu3): ReLU()
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (relu4): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)
```

## Dataset

### MNIST

- Download the MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/).
- Extract the dataset to a folder named `data`.
- The dataset should be organized as follows:

```
data/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

You can get the dataset by running `dataset.py`.

## Usage

Install the requirements by running the following command:

```bash
pip install -r requirements.txt
```

Run `dataset.py` to get the dataset in `./data`. Then run `train.py` to train the model.

Run `test.py` to test the model. **(to be accomplished)**

You can visualize the model's performance by running command below to use Tensorboard.

```bash
tensorboard --logdir=./runs
```
