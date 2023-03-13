import argparse
from utils import *
    
def read_args():
	parser = argparse.ArgumentParser()	
	parser.add_argument('-d', '--data_path', required=True,
                                     help='path of single image')
	parser.add_argument('-w', '--weights_path', required=False,
                                     help='weights path')
	parser.add_argument('-v', '--visualize', required=False)
        
	args = parser.parse_args()
	return args.data_path, args.weights_path, args.visualize

def evaluator(path, weights_path, for_all=True):
	model_gap = model_inceptionresnet_multigap()
	input_num = 5000
	if for_all == True:
		indxs = np.load('models/Indices/best_solution_2.npy')            
	else:
		indxs = np.load('models/Indices/best_solution_1.npy')   
	model = fc_model_softmax(input_num=input_num)
	model.load_weights(weights_path)
	model_CNN = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1", trainable=False)])    
	predicted = predict_from_path(model_gap=model_gap, model=model, paths=[path], resize_func=resize_max, size=(996, 996), for_all=for_all, model_cnn=model_CNN,take=indxs)
	return predicted


if __name__ == '__main__':
	data_path, weights_path, visualize = read_args()

	if weights_path == None:
		weights_path = 'models/Multigap_CNN/best_solution_2.hdf5'
		for_all = True        
	elif weights_path != None:
		if '1' in weights_path:
			for_all=False
		else:
			for_all=True            
        
        
	predicted = evaluator(data_path, weights_path, for_all=for_all)
	is_aesth = np.argmax(predicted, axis=-1) 

	if is_aesth:
		print('Image is aesthetic')
	else:
		print('Image is NOT aesthetic')

	if visualize:
		img = Image.open(data_path)

		plt.imshow(img)
		plt.title(f'Prediction on this image: {is_aesth}  ({round(np.sort(predicted)[-1] * 100, 2)} %) ')
		plt.show()

