import tensorflow as tf
import numpy as np

class resnet_fc(tf.keras.Model):
    def __init__(self, model_resnet,  model_fc): #model_cnn=model_cnn,
        super(resnet_fc, self).__init__()
        
        self.layer_resnet = model_resnet
        self.layer_fc = model_fc
#         self.concat = Concatenate()

    def call(self, img):
#         s = time.time()
        x = self.layer_resnet(img)
        x = self.layer_fc(x)
#         print('FC: ', time.time() - s)            
        return x