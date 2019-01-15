#coding=utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Define discriminator model
def discriminator_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
        64,
        (5,5),
        padding = 'same',
        input_shape = (64,64,3)
    )
    )
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(128,(5,5)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128,(5,5)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))#1024个神经元的全连接层
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))

    return model

#Define generator model
#Generate pics from random number
def generator_model():
    model = tf.keras.models.Sequential()
#输入维度是100 输出维度（神经元个数）是1024的全连接层
    model.add(tf.keras.layers.Dense(input_dim = 100,units = 1024))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dense(128*8*8))#8192个神经元的全连接层
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Reshape((8,8,128),input_shape=(128*8*8,)))#8*8像素
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
    model.add(tf.keras.layers.Conv2D(128,(5,5),padding = 'same'))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
    model.add(tf.keras.layers.Conv2D(3,(5,5),padding = 'same'))#深度3
    model.add(tf.keras.layers.Activation('tanh'))

    return model

if __name__ == '__main__':
    discriminator = discriminator_model()
    generator = generator_model()










