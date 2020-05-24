__author__ = 'dk'
#模型训练代码
from  .GAN import GAN
from .data_utils import Dator

epochs = 50000
height = 28
width = 28
latent_space_dimension = 100

dator = Dator()

gan = GAN(latent_space_dimension,height,width,channel)

for i in range(epochs):
    pass
