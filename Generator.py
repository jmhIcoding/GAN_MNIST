__author__ = 'dk'
#生成器

import sys
import numpy as np

import  keras
from  keras import layers
from keras import models
from  keras import optimizers
from keras import losses

class Generator:
    def __init__(self,height=28,width=28,channel=1,latent_space_dimension=100):
        '''
        :param height:    生成图片的高,minist为28
        :param width:     生成图片的宽,minist为28
        :param channel:   生成器所生成的图片的通道数目,对于mnist灰度图来说,channel为1
        :param latent_space_dimension:  噪声的维度
        :return:
        '''

        self.latent_space_dimension = latent_space_dimension
        self.height = height
        self.width = width
        self.channel = channel
        self.generator = self.build_model()
        #OPTIMIZER = optimizers.Adam()
        #self.generator.compile(optimizer=OPTIMIZER,loss=losses.binary_crossentropy,metrics =['accuracy'])
        self.generator.summary()
    def build_model(self,block_starting_size=128,num_blocks=4):
        model = models.Sequential(name='generator')
        for i in range(num_blocks):
            if i ==0 :
                model.add(layers.Dense(block_starting_size,input_shape=(self.latent_space_dimension,)))
            else:
                block_size = block_starting_size * (2**i)
                model.add(layers.Dense(block_size))
                model.add(layers.LeakyReLU())
                model.add(layers.BatchNormalization(momentum=0.75))

        model.add(layers.Dense(self.height*self.channel*self.width,activation='tanh'))
        model.add(layers.Reshape((self.width,self.height,self.channel)))
        return  model
    def summary(self):
        self.model.summary()

    def save_model(self):
        self.generator.save("generator.h5")