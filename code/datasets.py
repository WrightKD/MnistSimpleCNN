import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import numpy as np
import pandas as pd
import csv
import numbers
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine import base_layer
from keras.utils import control_flow_util

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, training=True, transform=None):
        if training==True:
            f = open('/kaggle/working/MnistSimpleCNN/data/MNIST/raw/train-images-idx3-ubyte', 'rb')
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
            f.close()
            f = open('/kaggle/working/MnistSimpleCNN/data/MNIST/raw/train-labels-idx1-ubyte', 'rb')
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
            f.close()
        else:
            f = open('/kaggle/working/MnistSimpleCNN/data/MNIST/raw/t10k-images-idx3-ubyte', 'rb')
            xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
            f.close()
            f = open('/kaggle/working/MnistSimpleCNN/data/MNIST/raw/t10k-labels-idx1-ubyte', 'rb')
            ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
            f.close()
        xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float32)
        ys = ys.astype(np.int)
        
        (x_train, y_train) = self.mnist_extract_data("/kaggle/input/bbd-digit-recognizer/train.csv")
        self.x_data = x_train
        self.y_data = y_train
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = Image.fromarray(self.x_data[idx].reshape(28, 28))
        y = torch.tensor(np.array(self.y_data[idx]))
        if self.transform:
            x = self.transform(x)
        x = transforms.ToTensor()(np.array(x)/255)
        return x, y
    
    def mnist_extract_data(file, is_train=True, include_keras_dataset=False):
    
        x_values = []
        y_values = []

        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if is_train:
                        train_value = np.array([x for x in row[1:]]).reshape(28,28)
                        label_value = row[0]

                        x_values.append(train_value)
                        y_values.append(label_value)
                    else:
                        train_value = np.array([x for x in row], dtype=np.uint8).reshape(28,28)
                        x_values.append(train_value)

                    line_count += 1

            x, y = np.array(x_values), np.array(y_values, dtype=np.uint8)

            x_count_same = 0

            if include_keras_dataset:
                (x_keras_train, y_keras_train), (x_keras_test, y_keras_test) = keras.datasets.mnist.load_data()

                for x_value in x:
                    if x_value in x_keras_train:
                        x_count_same += 1


                if is_train:
                    x = np.concatenate((x, x_keras_train))
                    y = np.concatenate((y, y_keras_train))
                else:
                    x = np.concatenate((x, x_keras_test))
                    y = np.concatenate((x, y_keras_test))

        print(f'Data Count {len(x)}. Same {x_count_same}')           
        return x, y
