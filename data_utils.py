__author__ = 'dk'
#数据集采集器,主要是对mnist进行简单的封装
from keras.datasets import mnist
import numpy as np
def sample_latent_space(instances_number,latent_space_dimension):
    return  np.random.normal(0,1,(instances_number,latent_space_dimension))

class Dator:
    def __init__(self,batch_size=None,model_type=1):
        '''

        :param batch_size:
        :param model_type:  当model_type为-1的时候,表示0-9个数字都选;当model_type=2,说明只选择数字2
        :return:
        '''
        self.batch_size = batch_size
        self.model_type = model_type
        with np.load("mnist.npz", allow_pickle=True) as f:
            X_train, y_train = f['x_train'], f['y_train']
            #X_test, y_test = f['x_test'], f['y_test']
        if model_type != -1:
            X_train = X_train[np.where(y_train==model_type)[0]]
        if batch_size == None:
            self.batch_size = X_train.shape[0]
        else:
            self.batch_size = batch_size

        self.X_train = (np.float32(X_train)-128)/128.0
        self.X_train = np.expand_dims(self.X_train,3)

        self.watch_index = 0
        self.train_size = self.X_train.shape[0]
    def next_batch(self,batch_size = None):
        if batch_size == None:
            batch_size  =self.batch_size

        X=np.concatenate([self.X_train[self.watch_index:(self.watch_index+batch_size)], self.X_train[:batch_size]])[:batch_size]
        self.watch_index  = (self.watch_index + batch_size) % self.train_size
        return  X

if __name__ == '__main__':
    print(sample_latent_space(5,4))