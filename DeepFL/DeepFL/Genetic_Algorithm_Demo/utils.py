import tensorflow as tf
from glob import glob
import sys
import sklearn
import numpy as np
import random
import json
import cv2
import os

from random import shuffle
from keras.optimizers import SGD, Adam

from sklearn.utils import shuffle
from numpy.random import rand

import keras
import tensorflow.keras
from keras.layers import *
from keras.models import Model
import logging

 
# fc_model_softmax
# generate_root_path
# generate_init_pop
# take_from_feats
# extract_static_val_data
# load_for_current_bad
# loading_data_from_json
# loading_bench_data
# selection
# new_population
# trainer
#  predict
# calc_acc_eval
# lr_exp_decay
# mutation
# crossover
# customize_gpu
# cycle_for_population
# findBestFeats


def fc_model_softmax(input_num=16928):
    input_ = Input(shape=(input_num,))
    x = Dense(2048, kernel_initializer='he_normal', activation='relu')(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(256, kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    pred = Dense(2, activation='softmax')(x)

    model = Model(input_,pred)
    return model


def generate_root_path():
    if glob('D:Data/AesthAI/alm/splitted/alm_train/images/good/good1/*'): #or if os.path.exists('D:Data/AesthAI')
        return 'D:'
    else:
        return ''


def generate_init_pop(feat_vector_size, transformed_feat_vector_size, pop_size, init_size=2, init_main_path='best_res'):
    # generates random population
    # use init True if you have previously saved vectors in best_res and init_size < number of saved vectors
    population = []
    try:
        paths = glob(f'{init_main_path}/best_solution_custom_*')
        paths = shuffle(paths)
        init_num = 0

        if len(paths) < init_size:
            init_size = len(paths)
        # Initializes a part of random population
        while init_num < init_size:
            vector = np.load(paths[init_num])
            if len(vector) == transformed_feat_vector_size:
                population.append(vector)
                init_num += 1
    except:
        pass 
    
    len_after_init = len(population)
    print(len_after_init)
    # Fills in the rest of population with random solutions
    for i in range(pop_size - len_after_init):
        population.append(np.random.choice(feat_vector_size, size=transformed_feat_vector_size, replace=False))
    return np.stack(population)



# takes from features via selected indices 
def take_from_feats(data, idx):
    new_data = np.squeeze(data[:, idx])
    return new_data


def extract_static_val_data(data, perc = 0.1):
    np.random.seed(0)
    np.random.shuffle(data)
    lensplit = int( len(data) * perc )
    data_val = data[:lensplit]
    data = data[lensplit:]
    return data, data_val


# A function for loading only 1 bad data cluster. Can be called only from inside the loading_data_from_json function.
def load_for_current_bad(index, main_path=generate_root_path() + 'Data/AesthAI/alm/splitted/alm_train/',  
                         feats_MG = 'original',feats_CNN = 'border_600x600', cnn = 'cnn_efficientnet_b7'):       
    alm_train_bad = open(f'{main_path}data_bad{index+1}.json')
    bad_data = json.load(alm_train_bad)
    features_bad_list = []
    for data in bad_data:
        feat_path_1 = main_path + f'features/multigap/{feats_MG}/' + data['feature']
        feat_path_2 = main_path + f'features/{cnn}/{feats_CNN}/' + data['feature']
        connected = np.concatenate((np.squeeze(np.load(feat_path_1)), np.squeeze(np.load(feat_path_2))))
        features_bad_list.append(connected)
    features_bad_list = np.squeeze(np.array(features_bad_list))
    return features_bad_list

def loading_data_from_json(main_path = generate_root_path() + 'Data/AesthAI/alm/splitted/alm_train/', feats_MG = 'original',
                           feats_CNN = 'border_600x600', cnn = 'cnn_efficientnet_b7'):
    features_good1 = []
    lbl = []
    data_all = []  
    
    # Loading bad data
    features_bad_list = list(map(load_for_current_bad, range(7)))
    # Loading good data
    alm_train_good = open(f'{main_path}/data_good1.json')
    good_data = json.load(alm_train_good)
    for data in good_data:
        feat_path_1 = main_path + f'features/multigap/{feats_MG}/' + data['feature']
        feat_path_2 = main_path + f'features/{cnn}/{feats_CNN}/' + data['feature']
        connected = np.concatenate((np.squeeze(np.load(feat_path_1)), np.squeeze(np.load(feat_path_2))))
        features_good1.append(connected)
    features_good1 = np.squeeze(np.array(features_good1))
    features_bad_list[0], features_bad1_val = extract_static_val_data(features_bad_list[0], perc = 0.11)
    features_good, features_good_val = extract_static_val_data(features_good1, perc = 0.11)
    
    
    # Extracting validation data from train data
    val_data = np.concatenate( (features_bad1_val, features_good_val ) , axis=0 )
    val_lbl = np.concatenate( (np.zeros(len(features_bad1_val)), np.ones(len(features_good_val)) ), axis=0 )
    
    # Shuffling good data and creating labels
    for i in range(7):
        np.random.shuffle(features_good)
        data_i = np.concatenate((features_good, features_bad_list[i] ) , axis=0) 
        lbl_i = np.concatenate( (np.ones(len(features_good)), np.zeros(len(features_bad_list[i])) ), axis=0 )
        data_i, lbl_i = shuffle(data_i, lbl_i)
        data_all.append(data_i)
        lbl.append(lbl_i)
        
    data_all = np.array(data_all, dtype=object)
    lbl = np.array(lbl, dtype=object)
       
    return data_all, lbl, val_data, val_lbl

def loading_bench_data(bench_path=generate_root_path() + 'Data/AesthAI/benchmark_connected/', feats_MG='max_996', 
                       feats_CNN='border_600x600', cnn='cnn_efficientnet_b7'):    
    # loading benchmarks data features
    bench_bad = open(f'{bench_path}data_bad.json')
    bad_data = json.load(bench_bad)
    bench_bad = []
    for data in bad_data:
        feat_path_1 = bench_path + f'features/multigap/{feats_MG}/' + data['feature']
        feat_path_2 = bench_path + f'features/{cnn}/{feats_CNN}/' + data['feature']
        connected = np.concatenate((np.squeeze(np.load(feat_path_1)), np.squeeze(np.load(feat_path_2))))
        bench_bad.append(connected)
    
    bench_good = open(f'{bench_path}data_good.json')
    good_data = json.load(bench_good)
    bench_good = []
    for data in good_data:
        feat_path_1 = bench_path + f'features/multigap/{feats_MG}/' + data['feature']
        feat_path_2 = bench_path + f'features/{cnn}/{feats_CNN}/' + data['feature']
        connected = np.concatenate((np.squeeze(np.load(feat_path_1)), np.squeeze(np.load(feat_path_2))))
        bench_good.append(connected)
        
    bench = np.concatenate((bench_bad, bench_good))
    bench_labels = np.concatenate((np.zeros(len(bench_bad)), np.ones(len(bench_good))))
    
    return bench, bench_labels

# Selecting indicies for crossover and mutation
def selection(pop_acc, num_of_parents = 1):
    sorted_parents = np.argsort(pop_acc)[:: -1]
    return sorted_parents[: num_of_parents], sorted_parents[2 : num_of_parents]

# Writing children on population
def new_population(pop, pop_acc, children):
    indx_to_change = np.argsort(pop_acc)[:len(children)]
    pop[indx_to_change] = children
    return pop


def trainer(model, data, weights_path, data_val, batch_size=128, 
            epochs=15, learning_rate=0.003):

    X_train, y_train = data
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate)) #decay=0, 
    model.load_weights(weights_path)
    checkpoint = keras.callbacks.ModelCheckpoint(weights_path, 
                                                 monitor='val_loss', 
                                                 verbose=1, 
                                                 save_best_only=True, 
                                                 mode='min')
    
#     schedule = tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)
    
    callbacks_list = [checkpoint] # schedule, add for lr decay
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks_list, validation_data=data_val)
    return history

    
def predict(x, y, model, par):
    '''
    Does prediction on given numpy image using
    model_gap and model
    '''
    predicted = []
    for feat in x:
        feat = feat[par]
        pred_score = model.predict(feat[None], verbose=0)
        predicted.append(pred_score)
        
    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)
    acc = calc_acc_eval(y, predicted)
    
    return acc

    
def calc_acc_eval(labels, predicted):
    '''
    Calculating mean class error, e.g. predicted classes are 1vs0, 0vs0, 0vs0, 0vs0, then we have acc=0.25
    Inputs: 
        labels = target labels
        predicted = predicted binary probability distribution for the input
    Output:
        mean class error
    '''
    acc = np.sum(np.array(labels) == np.argmax(np.array(predicted), axis=1)) / len(labels)
    
    return np.round(acc * 100, 2)


# learning rate decay
def lr_exp_decay(epoch, lr):
    k = 0.048
    return lr * np.exp(-k*epoch)


def mutation(population, save=0.97):
    result = []
    save_size = int(save * population.shape[1])
    all_indx = np.arange(19488)
    
    for elem in population:
        #Selecting indices for mutation in each individual 
        #Indices in every individual must be unique and in range (0, 19488)
        res = np.random.choice(elem, size=save_size, replace=False)
        indx_mutation = np.delete(all_indx, elem)
        change = np.random.choice(indx_mutation, size=population.shape[1] - save_size, replace=False)
        result.append(np.concatenate((res,change), axis=0))
    return np.array(result)   


def crossover(parent_matrix):
    # Selecting indicies for crossover
    idx1 = np.arange(parent_matrix.shape[0])
    idx2 = np.roll(idx1, -1)
    idxs = np.vstack((idx1, idx2))
    # Concatenating [idx1], [idx2] solutions to do crossover between them
    concat_child_matrix = np.concatenate((parent_matrix[idxs[0]], parent_matrix[idxs[1]]), axis=1)
    child_matrix = np.empty(parent_matrix.shape)
    len_solution = len(child_matrix[0])
    for i in range(concat_child_matrix.shape[0]):
        concat_i = concat_child_matrix[i]
        # Selecting duplicate elements between the 2 parents
        uniques, count = np.unique(concat_i, return_counts=True)
        dup = uniques[count > 1]
        # Selecting the elements that are unique for 2 parents
        singles = np.setdiff1d(uniques, dup)
        if len(singles) == 0:
            child_matrix[i] = dup
        # Creating offspring (with all duplicate elemenets and the rest is randomly filled with unique elements)
        child_matrix[i] = np.concatenate((dup, np.random.choice(singles, len_solution - len(dup), replace=False)), axis=0)
    return child_matrix.astype(int)




def customize_gpu(gpu_index=0):
    gpus = tf.config.list_physical_devices('GPU')
    # Restrict gpu memory growt
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
    # Use only one virtual gpu if there is more than one device
    if len(gpus) > 1:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
            
def cycle_for_population(curr_pop, weights_path, model, data, val_data, lbl, val_lbl, bench,  bench_labels, 
                         transformed_feat_vector_size=5000, epochs=3,learning_rate = 0.003, batch_size = 128):  
        # Creating FC model
        model = fc_model_softmax(input_num=transformed_feat_vector_size)
        model.save_weights(weights_path) #if we want to cancel learning and start from 0
        for i in range(7):
            #Teansforming train and validation datas for current individual
            data_transformed = take_from_feats(data[i], curr_pop)
            val_data_transformed = take_from_feats(val_data, curr_pop)
            #Model training
            trainer(data=(data_transformed, lbl[i]), 
                    data_val=(val_data_transformed, val_lbl),
                    model=model,
                    weights_path=weights_path,
                    batch_size=batch_size,
                    epochs=epochs, 
                    learning_rate=learning_rate)
            
        #Model evaluation
        acc = predict(bench, bench_labels, model, curr_pop)
        
        #Saving best results: individual and model
        if acc > 91:
            np.save(f'/best_res/best_solution_custom_{acc}', curr_pop)
            model.save_weights(f'/best_res/best_custom_{acc}.hdf5')
        print(f'ACC: {acc}')
        return acc
    
# The main function which executes Genetic Algorithm. For more details check utils.py

def findBestFeats(pop_size, feature_vector_size, transformed_feat_vector_size, num_of_parents=4, max_gen=100, init_size=2,
                  weights_path =  f'models/custom_ga_popsize_100_vectorsize_5000.hdf5',
                  train_data_path=main_path,
                  bench_path=bench_path,
                  feats_MG='max_996',
                  feats_CNN='border_600x600', cnn='cnn_efficientnet_b7'):
    # Creating FC model
    model = fc_model_softmax(input_num=transformed_feat_vector_size)
    # Loading train data
    data, lbl, val_data, val_lbl = loading_data_from_json(main_path=train_data_path,
                                                                    feats_CNN=feats_CNN,
                                                                    cnn=cnn)
    
    # loading benchmark data 
    bench, bench_labels = loading_bench_data(bench_path=bench_path,
                                             feats_MG=feats_MG,
                                             feats_CNN=feats_CNN,
                                             cnn=cnn)
    
    print("Generating random population")
    # Initializing population for first generation
    pop = generate_init_pop(feature_vector_size, transformed_feat_vector_size, pop_size, init_size)

    gen = 1
    while gen <= max_gen:
        pop_acc = [] 
        for i in range(len(pop)):
            # Cycle_for_population is used to train, evaluate and return accuracy for each given solution 
            acc = cycle_for_population(curr_pop=pop[i],
                                       weights_path=weights_path,
                                       model=model, 
                                       data=data,
                                       val_data=val_data,
                                       lbl=lbl,
                                       val_lbl=val_lbl,
                                       bench=bench,
                                       bench_labels=bench_labels,
                                       transformed_feat_vector_size=transformed_feat_vector_size,
                                       epochs=1,
                                       learning_rate = 0.003,
                                       batch_size = 128)
            
            pop_acc = np.concatenate((pop_acc, acc[None]))

        print('Selection')
        # Selects solutions for crossover, mutation
        idxs_for_cross, idxs_for_mut = selection(pop_acc, num_of_parents=num_of_parents)
        
        parents = pop[idxs_for_cross]
        parents_for_mut = pop[idxs_for_mut]

        print('Crossover')
        children = crossover(parents)
        
        print('Mutation')
        pop[idxs_for_mut] = mutation(parents_for_mut)
        # Creates new population for next generation using children and mutated solutions
        pop = new_population(pop, pop_acc, children)

        print(f'Generation {gen}')
        gen += 1
        

# Same function as the above one, only for PY use with argparse 
def findBestFeatsPy(pop_path, generation, feature_vector_size, transformed_feat_vector_size, num_of_parents=4, 
                  weights_path =  f'models/custom_ga_popsize_100_vectorsize_5000.hdf5',
                  train_data_path=main_path,
                  bench_path=bench_path,
                  feats_MG='max_996',
                  feats_CNN='border_600x600', cnn='cnn_efficientnet_b7', log_txt='log.txt'):
    logging.basicConfig(filename=log_txt, level=logging.INFO)

    # Loading population from previous Generation
    pop = np.load(pop_path)
    # Creating FC model
    model = fc_model_softmax(input_num=transformed_feat_vector_size)
    # Loading train data
    data, lbl, val_data, val_lbl = loading_data_from_json(main_path=train_data_path,
                                                                    feats_CNN=feats_CNN,
                                                                    cnn=cnn)
    
    # loading benchmark data 
    bench, bench_labels = loading_bench_data(bench_path=bench_path,
                                             feats_MG=feats_MG,
                                             feats_CNN=feats_CNN,
                                             cnn=cnn)
    

    gen = generation
    pop_acc = [] 
    for i in range(len(pop)):
        # Cycle_for_population is used to train, evaluate and return accuracy for each given solution 
        acc = cycle_for_population(curr_pop=pop[i],
                                   weights_path=weights_path,
                                   model=model, 
                                   data=data,
                                   val_data=val_data,
                                   lbl=lbl,
                                   val_lbl=val_lbl,
                                   bench=bench,
                                   bench_labels=bench_labels,
                                   transformed_feat_vector_size=transformed_feat_vector_size,
                                   epochs=1,
                                   learning_rate = 0.003,
                                   batch_size = 128)
        
        logging.info(f'ACC: {acc}')
        pop_acc = np.concatenate((pop_acc, acc[None]))

    print('Selection')
    logging.info('Selection')
    # Selects solutions for crossover, mutation
    idxs_for_cross, idxs_for_mut = selection(pop_acc, num_of_parents=num_of_parents)

    parents = pop[idxs_for_cross]
    parents_for_mut = pop[idxs_for_mut]

    print('Crossover')
    logging.info('Crossover')
    children = crossover(parents)

    print('Mutation')
    logging.info('Mutation')

    pop[idxs_for_mut] = mutation(parents_for_mut)
    # Creates new population for next generation using children and mutated solutions
    pop = new_population(pop, pop_acc, children)
    np.save(pop_path, pop)
    
    logging.info(f'Generation {gen} ended, population saved')

    print(f'Generation {gen}')
