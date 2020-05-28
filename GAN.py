__author__ = 'dk'
#生成对抗网络

import keras
from keras import layers
from  keras import optimizers
from  keras import  losses
from  keras import models

import  sys
import os

from Discriminator import Discriminator
from Generator import Generator
class GAN:
    def __init__(self,latent_space_dimension,height,width,channel):
        self.generator  = Generator(height,width,channel,latent_space_dimension)
        self.discriminator = Discriminator(height,width,channel)
        self.discriminator.discriminator.trainable = False #GAN里面这部分不训练
        self.gan =  self.build_model()
        OPTIMIZER = optimizers.Adamax()
        self.gan.compile(optimizer = OPTIMIZER,loss = losses.binary_crossentropy)
        self.gan.summary()
    def build_model(self):
        model  = models.Sequential(name='gan')
        model.add(self.generator.generator)
        model.add(self.discriminator.discriminator)
        return  model
    def summary(self):
        self.gan.summary()

    def save_model(self):
        self.gan.save("gan.h5")