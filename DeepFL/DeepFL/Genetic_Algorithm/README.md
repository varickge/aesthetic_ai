# Usage
    1. run GA.sh script with default paramerts from terminal
    2. change number of generations or run from certain generation in GA.sh script in for cycle
    3. change default parametrs from GA_custom.py
         root_path=generate_root_path()
         main_path='Data/AesthAI/alm/splitted/alm_train/'
         root_path + main_path - must be data's directory         
         init - boolean , inintialize population or not in 0-th generation ,  default value is False
         initialization_path - if init = True path to initialize population , this folder can contain less or more vectors 
                               than pop_size it wouldn't raise errors,  default value is '../for_initialization'
         eval_on_bench - boolean , evaluate model on bebchmark or not ,  default value is False         
         feats_MG - folder's name to take features extracted with multigap,  default value is 'original'
         feats_CNN = folder's name to take features extracted with CNN,  default value is 'border_600x600', 
         save_for_mutation - percentage in individual vector to save during mutation ,  default value is 0.97
         pop_size - size of population,  default value is 100
         transformed_feat_vector_size -feature vector's size after transformation, default value is 5000
         num_of_parents - number of parents to use in selection and crossover , default value is 50
         weights_path=f'../models/custom_ga_popsize_100_vectorsize_5000_class.hdf5'
         perc - percentage of validation data duringtraining, default value is 0.11
         batch_size=128, 
         epochs=15
         learning_rate=0.003
         verbose=0
