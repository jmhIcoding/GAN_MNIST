__author__ = 'dk'
#判别器
import sys
import os
import keras
from  keras import layers
from keras import optimizers
from keras import models
from keras import losses
class Discriminator:
    def __init__(self,height=28,width=28,channel=1):
        '''
        
        :param height:  输入图片的高
        :param width:   输入图片的宽
        :param channel: 输入图片的通道数
        :return:
        '''
        self.height = height
        self.width = width
        self.channel = channel
        self.discriminator = self.build_model()
        OPTIMIZER = optimizers.Adam()
        self.discriminator = self.build_model()
        self.discriminator.compile(optimizer=OPTIMIZER,loss=losses.binary_crossentropy,metrics =['accuracy'])
        self.discriminator.summary()
    def build_model(self):
        model = models.Sequential(name='discriminator')
        model.add(layers.Flatten(input_shape=(self.width,self.height,self.channel)))
        model.add(layers.Dense(self.height*self.width*self.channel,input_shape=(self.width,self.height,self.channel)))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dense(self.height*self.width*self.channel//2))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dense(1,activation='sigmoid'))
        return model

    def summary(self):
        return self.discriminator.summary()

    def save_model(self):
        self.discriminator.save("discriminator.h5")
