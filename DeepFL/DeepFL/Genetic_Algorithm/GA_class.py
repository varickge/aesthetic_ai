import sys
sys.path.append('../')
from final_utils import *

class GA_custom:
    def __init__(self, root_path=generate_root_path(), init=False, initialization_path='../for_initialization',eval_on_bench=False,
                 main_path='Data/AesthAI/alm/splitted/alm_train/', 
                 feats_MG = 'original', feats_CNN = 'border_600x600', 
                 save_for_mutation=0.97, pop_size=100, transformed_feat_vector_size=5000, num_of_parents=50,
                 weights_path=f'../models/custom_ga_popsize_100_vectorsize_5000_class.hdf5', batch_size=128, 
                 epochs=15, learning_rate=0.003, verbose=0, perc=0.11, 
                 color = {'white':      "\033[1;37m",
                          'yellow':     "\033[1;33m",
                          'green':      "\033[1;32m",
                          'blue':       "\033[1;34m",
                          'cyan':       "\033[1;36m",
                          'red':        "\033[1;31m",
                          'magenta':    "\033[1;35m",
                          'black':      "\033[1;30m",
                          'darkwhite':  "\033[0;37m",
                          'darkyellow': "\033[0;33m",
                          'darkgreen':  "\033[0;32m",
                          'darkblue':   "\033[0;34m",
                          'darkcyan':   "\033[0;36m",
                          'darkred':    "\033[0;31m",
                          'darkmagenta':"\033[0;35m",
                          'darkblack':  "\033[0;30m",
                          'off':        "\033[0;0m"}):
        
        self.eval_on_bench = eval_on_bench
        self.root_path = root_path
        self.init = init
        self.initialization_path = initialization_path
        self.main_path = self.root_path + main_path
        self.feats_MG = feats_MG
        self.feats_CNN = feats_CNN
        self.save_for_mutation = save_for_mutation
        self.pop_size = pop_size
        self.transformed_feat_vector_size = transformed_feat_vector_size
        self.num_of_parents = num_of_parents
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate =learning_rate
        self.verbose = verbose
        self.perc = perc
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.color = color

    def read_generation(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-g', '--generation', required=True,
                            help='number of generation')
        args = parser.parse_args()
        return args.generation
    
    def generate_init_pop(self, feat_vector_size=19488, transformed_feat_vector_size=None, pop_size=None, init=False):
        # generates random population
        population = []
        len_after_init = len(population)
        for i in range(pop_size - len_after_init):
            population.append(np.random.choice(feat_vector_size, size=transformed_feat_vector_size, replace=False))
        return population
    
    def initialize_pop(self, pop):
        vectors = glob(f'{self.initialization_path}/*.npy')
        if len(vectors) > self.pop_size:
            vectors = vectors[:self.pop_size]
        for i, path in enumerate(vectors):
            pop[i] = np.load(path)
        return pop   
    
    def take_from_feats(self, data, idx):
        new_data = np.squeeze(data[:, idx])
        return new_data

    def load_for_current_bad(self, i=0):
        alm_train_bad = open(f'{self.main_path}train_data_bad{i+1}_new.json')
        bad_data = json.load(alm_train_bad)
        features_bad_list = []
        for data in bad_data:
            feat_path_1 = self.main_path + f'features/multigap/{self.feats_MG}/' + data['feature']
            feat_path_2 = self.main_path + f'features/cnn_efficientnet_b7/{self.feats_CNN}/' + data['feature']
            connected = np.concatenate((np.squeeze(np.load(feat_path_1)), np.squeeze(np.load(feat_path_2))))
            features_bad_list.append(connected)
        features_bad_list = np.squeeze(np.array(features_bad_list))
        
        return features_bad_list

    def loading_data_from_json(self):
        features_good1 = []
        lbl = []
        data_all = []  
        
        features_bad_list = list(map(self.load_for_current_bad, range(7)))
        alm_train_good = open(f'{self.main_path}/train_data_good1_new.json')
        good_data = json.load(alm_train_good)
        for data in good_data:
            feat_path_1 = self.main_path + f'features/multigap/{self.feats_MG}/' + data['feature']
            feat_path_2 = self.main_path + f'features/cnn_efficientnet_b7/{self.feats_CNN}/' + data['feature']
            connected = np.concatenate((np.squeeze(np.load(feat_path_1)), np.squeeze(np.load(feat_path_2))))
            features_good1.append(connected)
            
        features_good1 = np.squeeze(np.array(features_good1))
        features_bad_list[0], features_bad1_val = extract_static_val_data(features_bad_list[0], self.perc)
        features_good, features_good_val = extract_static_val_data(features_good1, self.perc)
        val_data = np.concatenate( (features_bad1_val, features_good_val ) , axis=0 )
        val_lbl = np.concatenate( (np.zeros(len(features_bad1_val)), np.ones(len(features_good_val)) ), axis=0 )
        for i in range(7):
            np.random.shuffle(features_good1)
            data_i = np.concatenate((features_good1, features_bad_list[i] ) , axis=0) # ---- Changed ----
            lbl_i = np.concatenate( (np.ones(len(features_good1)), np.zeros(len(features_bad_list[i])) ), axis=0 )
            data_i, lbl_i = shuffle(data_i,lbl_i)
            data_all.append(data_i)
            lbl.append(lbl_i)
  
        data_all = np.array(data_all, dtype=object)
        lbl = np.array(lbl, dtype=object)

        return data_all, lbl, val_data, val_lbl
    
    def loading_bench_datas(self):
        bench_path = self.root_path + 'Data/AesthAI/benchmark_connected/' 

        # loading benchmark1 data features
        bench_bad = open(f'{bench_path}data_bad.json')
        bad_data = json.load(bench_bad)
        bench_bad = []
        for data in bad_data:
            feat_mg_path = bench_path + f'features/multigap/all_res_996/' + data['feature']
            feat_cnn_path = bench_path + f'features/cnn_efficientnet_b7/border_600x600/' + data['feature']
            connected = np.concatenate((np.squeeze(np.load(feat_mg_path)), np.squeeze(np.load(feat_cnn_path))))
            bench_bad.append(connected)

        bench_good = open(f'{bench_path}data_good.json')
        good_data = json.load(bench_good)
        bench_good = []
        for data in good_data:
            feat_mg_path = bench_path + f'features/multigap/all_res_996/' + data['feature']
            feat_cnn_path = bench_path + f'features/cnn_efficientnet_b7/border_600x600/' + data['feature']
            connected = np.concatenate((np.squeeze(np.load(feat_mg_path)), np.squeeze(np.load(feat_cnn_path))))
            bench_good.append(connected)
        bench = np.concatenate((bench_bad, bench_good))
        bench_labels = np.concatenate((np.zeros(len(bench_bad)), np.ones(len(bench_good))))

        return bench, bench_labels

    def loading_test_data_from_json(self):
        test_bad = open(f'{self.main_path}test_data_bad.json')#hanel new
        test_bad_data = json.load(test_bad)
        test_bad_feats = []
        for data in test_bad_data:
            feat_mg_path = self.main_path + f'features/multigap/all_res_996/' + data['feature']
            feat_cnn_path = self.main_path + f'features/cnn_efficientnet_b7/border_600x600/' + data['feature']
            connected = np.concatenate((np.squeeze(np.load(feat_mg_path)), np.squeeze(np.load(feat_cnn_path))))
            test_bad_feats.append(connected)

        test_good = open(f'{self.main_path}test_data_good.json')#hanel new
        test_good_data = json.load(test_good)
        test_good_feats = []
        for data in test_good_data:
            feat_mg_path = self.main_path + f'features/multigap/all_res_996/' + data['feature']
            feat_cnn_path = self.main_path + f'features/cnn_efficientnet_b7/border_600x600/' + data['feature']
            connected = np.concatenate((np.squeeze(np.load(feat_mg_path)), np.squeeze(np.load(feat_cnn_path))))
            test_good_feats.append(connected)
        test_data = np.concatenate((test_bad_feats, test_good_feats))
        test_labels = np.concatenate((np.zeros(len(test_bad_feats)), np.ones(len(test_good_feats))))
        test_data, test_labels = shuffle(test_data, test_labels)

        return test_data, test_labels

    def selection(self, pop_acc, num_of_parents = 1):
        sorted_parents = np.argsort(pop_acc)[:: -1]
        return sorted_parents[: num_of_parents], sorted_parents[2 : num_of_parents]

    def new_population(self, pop, pop_acc, children):
        indx_to_change = np.argsort(pop_acc)[:len(children)]
        pop[indx_to_change] = children
        return pop

    def trainer(self, model, data, weights_path, data_val, batch_size=128, epochs=15, learning_rate=0.003, verbose=0):

        X_train, y_train = data
        model.compile(loss=self.loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        checkpoint = keras.callbacks.ModelCheckpoint(self.weights_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        schedule = tf.keras.callbacks.LearningRateScheduler(self.lr_exp_decay, verbose=0)
        callbacks_list = [checkpoint, schedule]
        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, callbacks=callbacks_list,
                  validation_data = data_val)

    def predict(self, x, y, model, par):
        '''
        Does prediction on given x[par] using model, and calc acc 
        '''
        predicted = []
        for feat in x:
            feat = feat[par]
            pred_score = model.predict(feat[None], verbose=0)
            predicted.append(pred_score)

        predicted = np.array(predicted)
        predicted = np.squeeze(predicted)
        acc = self.calc_acc_eval(y, predicted)

        return acc

    def calc_acc_eval(self, labels, predicted):
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

    def lr_exp_decay(self, epoch, lr):
        k = 0.048 # k was choosen for 15 epoch
        return lr * np.exp(-k*epoch)

    def mutation(self, population, save=0.97):
        result = []
        save_size = int(save  * population.shape[1])
        all_indx = np.arange(19488)

        for elem in population:
            res = np.random.choice(elem, size=save_size, replace=False)
            indx_mutation = np.delete(all_indx, elem)
            change = np.random.choice(indx_mutation, size=population.shape[1] - save_size, replace=False)
            result.append(np.concatenate((res,change), axis=0))
        return np.array(result)   

    def crossover(self, parent_matrix, num_parents):
        idxs = (np.reshape(np.arange(parent_matrix.shape[0]), (parent_matrix.shape[0], 1)) 
                + np.arange(parent_matrix.shape[0])[None]) % parent_matrix.shape[0]

        concat_child_matrix = np.concatenate((parent_matrix[idxs[0:num_parents]]), axis=1)
        child_matrix = np.empty(parent_matrix.shape)
        k = len(child_matrix[0])
        for i in range(concat_child_matrix.shape[0]):
            a = concat_child_matrix[i]
            unique, count = np.unique(a, return_counts=True)
            new_idx = np.argsort(count)[::-1]
            unique[new_idx]
            child_matrix[i] = unique[new_idx][:child_matrix.shape[1]]

        return child_matrix.astype(int)
        

    def __call__(self, generation=None):
        # --- creating log.txt ---
        logging.basicConfig(filename="log_class.txt", level=logging.ERROR)

        # --- creating model and loading data ---
        model = fc_model_softmax(input_num=self.transformed_feat_vector_size)
        data_full, lbl_full, val_data_full, val_lbl_full = self.loading_data_from_json()
        test_data, test_labels = self.loading_test_data_from_json()
        bench, bench_labels = self.loading_bench_datas()
        
        try:
            # --- reading generation, if GA run from script ---
            generation = int(self.read_generation())
        except:
            generation=generation
        

        def cycle_for_population(pop, model=model, data=data_full,
                                 val_data=val_data_full, lbl=lbl_full, val_lbl=val_lbl_full, bench=bench, 
                                 bench_labels=bench_labels, epochs=5):
            # choosing hyperparams
            learning_rate = 0.003
            batch_size = 128

            model = fc_model_softmax(input_num=self.transformed_feat_vector_size)
            model.save_weights(self.weights_path)

            def trainer_for_pair_full(bad_i, curr_pop=pop, data=data_full, val_data=val_data, 
                                      lbl=lbl_full, val_lbl=val_lbl, 
                                      test_data=test_data, test_labels=test_labels,
                                      model=model, epochs=epochs):
                # --- take from feats needed indices ---
                data_transformed = self.take_from_feats(data[bad_i], curr_pop)
                val_data_transformed = self.take_from_feats(val_data, curr_pop)
                
                # --- starting training process ---
                self.trainer(model=model, data=(data_transformed, lbl[bad_i]), weights_path=self.weights_path, 
                        data_val=(val_data_transformed, val_lbl), batch_size=batch_size, epochs=epochs, 
                        learning_rate=learning_rate, verbose=0)
                
            # --- training FC on 7 (good_1, bad_i) data ---
            for i in range(7):
                trainer_for_pair_full(i)
            
            # --- evaluate FC on test data ---
            acc1 = self.predict(test_data, test_labels, model, pop)
            if self.eval_on_bench:
                acc2 = self.predict(bench, bench_labels, model, pop)
                if acc1 >= 94 and acc2 >= 94: # Hard code: this for security
                    np.save(f'../best_res/best_solution_custom_test_{acc1}_bench_{acc2}', pop)
                    model.save_weights(f'../best_res/best_custom_test_{acc1}_bench_{acc2}.hdf5')
                    print(self.color['green'] + f'ACC: test - {acc1}, bench - {acc2}' + self.color['off'])
                    logging.error(self.color['green'] + f'ACC: test - {acc1}, bench - {acc2}' + self.color['off'])
                else:
                    print(self.color['white'] + f'ACC: test - {acc1}, bench - {acc2}' + self.color['off'])
                    logging.error(self.color['white'] + f'ACC: test - {acc1}, bench - {acc2}' + self.color['off'])
            
            elif acc1 >= 94:
                np.save(f'../best_res/best_solution_custom_test_{acc1}', pop)
                model.save_weights(f'../best_res/best_custom_test_{acc1}.hdf5')
                print(self.color['green'] + f'ACC: test - {acc1}' + self.color['off'])
                logging.error(self.color['green'] + f'ACC: test - {acc1}' + self.color['off']) 
            else:
                print(self.color['white'] + f'ACC: test - {acc1}' + self.color['off'])
                logging.error(self.color['black'] + f'ACC: test - {acc1}' + self.color['off'])
            
            return acc1
        
        # --- Attention: if we run from script we need load population ---
        if generation == 0:
            pop = self.generate_init_pop(transformed_feat_vector_size=self.transformed_feat_vector_size, pop_size=self.pop_size)
            np.save('GA_pop.npy', pop)
            
            if self.init:
                pop = self.initialize_pop(pop)
            
        print('Generating random population')
        pop = np.load('GA_pop.npy')
        pop_acc = []
        for i in range(len(pop)):
            acc = cycle_for_population(pop=pop[i], epochs=15)
            pop_acc = np.concatenate((pop_acc, [acc]))

        print('Selection')
        logging.error('Selection')
        idxs_for_cross, idxs_for_mut = self.selection(pop_acc, num_of_parents=num_of_parents)
        pop = np.stack(pop)
        parents = pop[idxs_for_cross]
        parents_for_mut = pop[idxs_for_mut]

        print('Crossover')
        logging.error('Crossover')
        children = self.crossover(parents, 2)

        print('Mutation')
        logging.error('Mutation')
        pop[idxs_for_mut] = self.mutation(parents_for_mut)
        pop = new_population(pop, pop_acc, children)
        np.save('GA_pop.npy', pop)

        print(f'End of  {generation} generation')
        logging.error(f'End of  {generation} generation')