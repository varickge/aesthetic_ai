import random
import os

import argparse
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import *
from keras.models import Model
from keras.applications import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from PIL import Image
from glob import glob
from numpy.random import rand
from utils import *
    
def read_args():
	parser = argparse.ArgumentParser()	
	parser.add_argument('-d', '--data_path', required=True,
                                     help='path of single image')
	parser.add_argument('-w', '--weights_path', required=False,
                                     help='weights path')
	parser.add_argument('-v', '--visualize', required=False)
	parser.add_argument('-p', '--pca', required=False,  default='False', 
                                    help='true to use pca')
	parser.add_argument('-ga', '--genetic_algorithm', required=False,default='True',
                                    help='true to use genetic algorithm indices')
    

	args = parser.parse_args()

	return args.data_path, args.weights_path, args.visualize, args.pca, args.genetic_algorithm

def evaluator(path, weights_path, genetic_algorithm):
	model_gap = model_inceptionresnet_multigap()
	if genetic_algorithm:
		input_num = 5000
		indxs = np.load('models/Indices/best_solution.npy')
		pca_mg = None
		pca_cnn = None
	else:
		input_num = 9744
		pca_mg = pk.load(open('models/PCA/pca_mg.pkl','rb'))
		pca_cnn = pk.load(open('models/PCA/pca_cnn.pkl','rb')) 
		indxs = None               
	model = fc_model_softmax(input_num=input_num)
	model.load_weights(weights_path)
	model_CNN = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",trainable=False) ])
    
   
    
	predicted = predict_from_path(model_gap=model_gap, model=model, paths=[path], resize_func=resize_max, size=(996,996), for_all=False, save_results=None, save_to=None, model_cnn=model_CNN,take=indxs)
	return predicted


if __name__ == '__main__':
	data_path, weights_path, visualize, pca, genetic_algorithm= read_args()
	if pca=='True':
		pca = True
	else:
		pca = False
	if genetic_algorithm=='True':
		genetic_algorithm = True
	else:
		genetic_algorithm = False
 
                        
	if pca == genetic_algorithm:
		raise ValueError('Choose one method pca or ga, this arguments must have opposite values! Default values are pca = False, ga = True! ')

                        
	if weights_path == None and genetic_algorithm == True:
		weights_path = 'models/Softmax/Multigap_CNN/best_solution_GA_custom.hdf5'
      
	elif weights_path == None and pca == True:
		weights_path = 'models/Softmax/Multigap_CNN/model_fc_softmax_MG_8k_B7_1k_600x600.hdf5'

	predicted = evaluator(data_path, weights_path, genetic_algorithm)
	is_aesth = np.argmax(predicted, axis=-1) 

	if is_aesth:
		print('Image is aesthetic')
	else:
		print('Image is NOT aesthetic')

	if visualize:
		img = Image.open(data_path)

		plt.imshow(img)
		plt.title(f'Prediction on this image: {is_aesth}  ({round(np.sort(predicted)[-1] * 100,2)} %) ')
		plt.show()
