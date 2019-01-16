#coding=utf-8
"""Train DCGAN"""

import glob
import numpy as np
from scipy import misc
import tensorflow as tf
from DCGAN import *
#Hyperparameter
EPOCH = 100
BATCH_SIZE = 128
LEARNING_RATE =0.0002
Beta_1 = 0.5
def train():
    #获取训练数据
    data = []
    for image in glob.glob("image/*"):
        image_data =misc.imread(image)#return image array

        data.append(image_data)
    input_data = np.array(data)

    #将数据标准化成[-1,1],tanh的激活取值范围

    input_data =(input_data.astype(np.float32)-127.5)/127.5

    #构造生成器
    g = generator_model()
    d = discriminator_model()

    #构建生成器和判别器组成的网络模型
    d_on_g = generator_containing_discriminator(g,d)

    #优化器用 adam optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr = learning_rate,beta_1=Beta_1)
    d_optimizer = tf.keras.optimizers.Adam(lr = learning_rate,beta_1=Beta_1)

    #配置生成器和判别器
    g.compile(loss = "binary_crossentropy",optimizer=g_optimizer)
    d.compile(loss = "binary_crossentropy",optimizer=d_optimizer)
    d.trainable = True #设置标志位 根据此时是否能够训练，若不能训练则把判别器固定 优化生成器
    d_on_g.compile(loss = "binary_crossentropy",optimizer=d_optimizer)# 先有一个生成器再经过一个判别器

    #开始训练
    for epoch in range(EPOCH):
        for index in range(int(input_data.shape[0]/(BATCH_SIZE))):

            #每经过一个区块的大小去训练一次

            input_batch = ipnut_data[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            #连续均匀分布的随机数据

            random_data = np.random.uniform(-1,1,size=(BATCH_SIZE,100))

            #生成器 生成的图片数据

            generated_images = g.predict(random_data,verbose=0)

            input_batch = np.concatenate((input_batch,generated_images))
            output_batch = [1]*BATCH_SIZE+[0]*BATCH_SIZE

            #训练判别器， 让他具备识别不合格生成图片的能力
            d_loss = d.train_on_batch(input_batch,output_batch)

            #当训练生成器时，让判别器不可被训练
            d.trainable = False

            #训练生成器，并通过不可被训练的判别器去判别
            g_loss = d_on_g.train_on_batch(random_data,[1]*BATCH_SIZE)
            #恢复判别器可被训练
            d.trainable = True
            #打印损失
        print("Step %d Generator Loss: %f Discriminator Loss: %f"%(index,g_loss,d_loss))



if __name__ == "__main__":
    train()