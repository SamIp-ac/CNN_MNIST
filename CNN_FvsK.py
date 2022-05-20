import os
import numpy as np
import pandas as pd
import tensorflow as tf
from cv2 import cv2
from sklearn.metrics import mean_squared_error
import random

# Remark: move this file('STAT4012_HW2') with the dataset('FKPY') into your own framework, the image is the sample.


def Creat_dataset(letter, filename='FKPY', test=False):

    reset = os.getcwd()
    cwd = os.path.join(os.getcwd(), filename, letter)
    os.chdir(cwd)
    image_name = []
    image = []

    for i in range(0, 5):
        filelist = [file for file in os.listdir(os.path.join(os.getcwd(), 'hsf_{}'.format(i))) if file.endswith('.png')]
        random.seed(4011)
        random.shuffle(filelist)
        image_name.append(filelist)

    filelist = [file for file in os.listdir(os.path.join(os.getcwd(), 'hsf_{}'.format(6))) if file.endswith('.png')]
    random.seed(4011)
    random.shuffle(filelist)
    image_name.append(filelist)

    for j in range(len(image_name) - 1):
        cwd_temp = os.path.join(os.getcwd(), 'hsf_{}'.format(j))
        os.chdir(cwd_temp)
        for k in image_name[j]:
            temp = cv2.imread(k, cv2.IMREAD_GRAYSCALE)
            temp = cv2.resize(temp, (32, 32))
            image.append(temp)
        os.chdir(cwd)

    cwd_temp = os.path.join(os.getcwd(), 'hsf_{}'.format(6))
    os.chdir(cwd_temp)

    for k in image_name[5]:
        temp = cv2.imread(k, cv2.IMREAD_GRAYSCALE)
        temp = cv2.resize(temp, (32, 32))
        image.append(temp)

    image = np.array(image)
    os.chdir(reset)
    return image


def Creat_label(image_data, label):
    size = image_data.shape[0]
    labels = []
    if label == 1:
        labels = np.ones(size)
    elif label == 0:
        labels = np.zeros(size)
    return labels


def Creat_test(letter, filename='FKPY'):
    reset = os.getcwd()
    cwd = os.path.join(os.getcwd(), filename, letter)
    os.chdir(cwd)
    image = []
    image_name = []

    filelist = [file for file in os.listdir(os.path.join(os.getcwd(), 'hsf_{}'.format(7))) if file.endswith('.png')]
    image_name.append(filelist)

    cwd_temp = os.path.join(os.getcwd(), 'hsf_{}'.format(7))
    os.chdir(cwd_temp)
    for i in image_name[0]:
        temp = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        temp = cv2.resize(temp, (32, 32))
        image.append(temp)

    os.chdir(reset)
    image = np.array(image)

    return image


def DATA_concatenate(data_1, data_2):
    Data = np.array(np.concatenate((data_1, data_2), axis=0))
    return Data


class CNN_model:
    def build_model(self):
        img_height = 32
        img_width = 32
        tf.random.set_seed(4012)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding='valid', input_shape=X_train.shape[1:]))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
        model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        return (model)

    def train(self, X_train, y_train, bs=32, ntry=2):
        model = self.build_model()
        model.fit(X_train, y_train, batch_size=bs, epochs=10, shuffle=True, verbose=2)
        self.best_model = model
        best_loss = model.evaluate(X_train, y_train)

        '''for i in range(ntry):
            model = self.build_model()
            model.fit(X_train, y_train, batch_size=bs, epochs=30, shuffle=True, verbose=2)
            if model.evaluate(X_train, y_train) < best_loss:
                self.best_model = model
                best_loss = model.evaluate(X_train, y_train)'''

    def predict(self, X_test):
        return (self.best_model.predict(X_test))

    def evaluate(self, y_pred, y_test):
        model = self.best_model
        evaluate = model.evaluate(y_pred, y_test)
        return evaluate


# a
X_train = DATA_concatenate(Creat_dataset('F'), Creat_dataset('K'))/255
X_train = X_train.reshape(-1, 32, 32, 1)
y_train = DATA_concatenate(Creat_label(Creat_dataset('F'), label=1), Creat_label(Creat_dataset('K'), label=0))


X_test = DATA_concatenate(Creat_test('F'), Creat_test('K'))/255
X_test = X_test.reshape(-1, 32, 32, 1)
y_test = DATA_concatenate(Creat_label(Creat_test('F'), label=1), Creat_label(Creat_test('K'), label=0))


CNN = CNN_model()
CNN.train(X_train, y_train)
y_pred = CNN.predict(X_test)

loss, acc = CNN.evaluate(X_train, y_train)
print('For training data (F vs K), the loss is %s , the accuracy is %s' % (loss, acc))

print('Testing mse:', mean_squared_error(y_test, y_pred))
loss, acc = CNN.evaluate(X_test, y_test)
print('For testing data (F vs K), the loss is %s , the accuracy is %s' % (loss, acc))
