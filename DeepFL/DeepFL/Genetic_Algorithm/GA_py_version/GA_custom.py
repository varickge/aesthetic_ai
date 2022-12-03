import sys
sys.path.append('../../')
sys.path.append('../')

from final_utils import *
from utils import *
import argparse

def read_generation():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generation', required=True, help='number of generation')
    args = parser.parse_args()
    return args.generation

generation = int(read_generation())

gpu_idx = 0
customize_gpu(gpu_idx)

root_path = generate_root_path()

main_path = f'{root_path}/Data/AesthAI/alm/splitted/alm_train/'
bench_path = f'{root_path}/Data/AesthAI/benchmark_connected/'

# NAVSIAKI COMMENT VOR HANKARC INIT CHANI

# if generation == 1:
#     # Generating init pop
#     pop = generate_init_pop(feat_vector_size=19488, transformed_feat_vector_size=5000,                                      pop_size=100, init_size=2, init_main_path='../best_res')
#     np.save('GA_pop.npy', pop)
    
#     # Creating model
#     model = fc_model_softmax(input_num=5000)
#     weights_path =  '../models/custom_ga_popsize_100_vectorsize_5000.hdf5'
#     model.save_weights(weights_path)
    
findBestFeatsPy(pop_path='GA_pop.npy',
                path_to_save=f'../best_res/test/',
                generation=generation,
                feature_vector_size=19488,
                transformed_feat_vector_size=5000,
                num_of_parents=50,
                weights_path='../models/custom_ga_popsize_100_vectorsize_5000.hdf5',
                train_data_path=main_path,
                bench_path=bench_path,
                feats_MG='max_996',
                feats_CNN='border_600x600', 
                cnn='cnn_efficientnet_b7',
                log_txt='log.txt')