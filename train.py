__author__ = 'dk'
#模型训练代码
from  GAN import GAN
from data_utils import Dator,sample_latent_space
import  numpy as np
from matplotlib import pyplot as plt
import time

epochs = 50000
height = 28
width = 28
channel =1
latent_space_dimension = 100
batch = 128
dator = Dator(batch_size=batch,model_type=-1)
gan = GAN(latent_space_dimension,height,width,channel)
image_index = 0
for i in range(epochs):
    real_img = dator.next_batch(batch_size=batch*2)
    real_label = np.ones(shape=(real_img.shape[0],1))       #真实的样本设置为1的标签

    noise = sample_latent_space(real_img.shape[0],latent_space_dimension)
    fake_img = gan.generator.generator.predict(noise)
    fake_label = np.zeros(shape=(fake_img.shape[0],1))      #生成器生成的假图片标注为0

    ###合成给gan的鉴别器的数据
    x_batch = np.concatenate([real_img,fake_img])
    y_batch = np.concatenate([real_label,fake_label])
    #训练一次
    discriminator_loss = gan.discriminator.discriminator.train_on_batch(x_batch,y_batch)[0]
    ###合成训练生成器的数据
    noise = sample_latent_space(batch*2,latent_space_dimension)
    noise_labels = np.ones((batch*2,1))           #生成器的目标是把图片的label越来越像1

    generator_loss = gan.gan.train_on_batch(noise,noise_labels)

    print('Epoch : {0}, [Discriminator Loss:{1} ], [Generator Loss:{2}]'.format(i,discriminator_loss,generator_loss))

    if i!=0 and (i%50)==0:
        print('show time')
        noise = sample_latent_space(16,latent_space_dimension)
        images = gan.generator.generator.predict(noise)
        plt.figure(figsize=(10,10))
        plt.suptitle('epoch={0}'.format(i),fontsize=16)
        for index in range(images.shape[0]):
            plt.subplot(4,4,index+1)
            image  =images[index,:,:,:]
            image = image.reshape(height,width)
            plt.imshow(image,cmap='gray')
        #plt.tight_layout()
        plt.savefig("./show_time/{0}.png".format(time.time()))
        image_index += 1
        plt.close()

