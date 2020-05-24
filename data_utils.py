__author__ = 'dk'
#数据集采集器,主要是对mnist进行简单的封装
from keras.datasets import mnist
import numpy as np
class Dator:
    def __init__(self,batch_size=2048,model_type=1):
        '''

        :param batch_size:
        :param model_type:  当model_type为-1的时候,表示0-9个数字都选;当model_type=2,说明只选择数字2
        :return:
        '''
        self.batch_size = batch_size
        self.model_type = model_type
        (X_train,y_train),(_,__) = mnist.load_data()
        if model_type ! = -1:
            X_train = X_train[np.where(y_train==model_type)[0]]

        self.X_train = (np.float32(X_train)-128)/128.0
        self.X_train = np.expand_dims(self.X_train,3)

        self.watch_index = 0
        self.train_size = self.X_train.shape[0]
    def next_batch(self,batch_size = None):
        if batch_size == None:
            batch_size  =self.batch_size

        X= (self.X_train[self.watch_index:(self.watch_index+batch_size)] + self.X_train[:batch_size])[:batch_size]
        self.watch_index  = (self.watch_index + batch_size) % self.train_size
        return  X

