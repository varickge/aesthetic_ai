import sys

sys.path.append('../')
sys.path.append('../models/Resnet')
sys.path.append('../Genetic_Algorithm/best_res')

from final_utils import *
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)

from ResNet import *
import time
import argparse
import logging

root_path = generate_root_path()
def read_epoch():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', required=True, help='number of epoch')
    args = parser.parse_args()
    return args.epoch

epoch = int(read_epoch())
### DataLoader and other nesseccary functions


indxs = np.load('../Genetic_Algorithm/best_res/best_solution_custom_90.09_91.78_mg.npy')
# print(indxs.shape)
# def take_from_vector(feature_vector, indxs ):    
# #     return feature_vector[indxs]
#     return np.take(feature_vector, indxs)

def take_from_vector(data, indxs):
    idx = np.nonzero(indxs)
    new_data = np.squeeze(data[idx])
    return new_data

# def lr_exp_decay(epoch, lr):
#     k = 0.04
#     return lr * np.exp(-k*epoch)


# def pca_transform(vector,path = "/home/server3090ti/Data/AesthAI/alm/splitted/alm_train/features/multigap/original_PCA_8464_auto/model/pca.pkl"):
#     pca = pk.load(open(path,'rb'))
#     return pca.transform(vector)

def load_batch(paths, main_path=f'{root_path}Data/AesthAI/alm/splitted/alm_train/'):
    images = []
    feats = []
    
    feats_MG = 'original'  
    feats_CNN = 'border_600x600'
    # feats_CNN_MG_PCA = 'cnn_mg_concat/pca_9744_auto'
#     cnn = 'cnn_efficientnet_b7'
    
    for i, path in enumerate(paths):
#         if i % 100 == 0:
#             print(f'batch {i}')
        try:
            img = cv2.imread(path, cv2.COLOR_BGR2RGB)
            img = resize_add_border(img, size=(600, 600)) # 600
            img = img.astype('float32')
            img /= 255 
#             img = tf.convert_to_tensor(img)
        except:
            continue
            
        images.append(img)
        path_to_feat = path.split('.')[0].split('/')[-1] + '.npy'
#         path_to_feat = main_path + 'features/multigap/all_res_996_PCA_4232_auto/' + path_to_feat 
#         feats.append(np.load(path_to_feat))
        feat_path_1 = main_path + f'features/multigap/{feats_MG}/' + path_to_feat
#         feat_path_2 = main_path + f'features/{cnn}/{feats_CNN}/' + path_to_feat

#         feat_1 = pca_mg.transform(np.load(feat_path_1))
#         feat_2 = pca_cnn.transform(np.load(feat_path_2)) 
        feat_1 = np.load(feat_path_1)
#         feat_2 = np.load(feat_path_2)
#         connected = np.concatenate((np.squeeze(feat_1), np.squeeze(feat_2)))
#         print(feat_1.shape)
        connected = take_from_vector(np.squeeze(feat_1), indxs) # must be commented later
        feats.append(connected)
    
                     
    images = np.stack(images) #? maybe ToDo:tf stack & add axis=0
    feats = np.squeeze(feats)
    
    return images, feats

### Trainer

def trainer(model, data, data_val,batch_size=128, epochs=30, learning_rate=0.03, verbose=0):
    X_train, y_train = load_batch(data)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)#.prefetch(tf.data.AUTOTUNE)  # ToDo: here use .cache()

    history = model.fit(train_dataset,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=verbose,
                        validation_data = data_val)

    return history

### Loading data

main_path = main_path=f'{root_path}Data/AesthAI/alm/splitted/alm_train/'
paths_bad = []
paths_good = []
    
for i in range(7):
    alm_train_bad = open(f'{main_path}data_bad{i+1}.json')
    bad_data = json.load(alm_train_bad)
    
    for data in bad_data:
        path_to_img = main_path + f'images/{data["label"]}/{data["splitted"]}/' + data['name']
        paths_bad.append(path_to_img)
        
alm_train_good = open(f'{main_path}/data_good1.json')
good_data = json.load(alm_train_good)
for data in good_data:
    path_to_img = main_path + f'images/{data["label"]}/{data["splitted"]}/' + data['name']
    paths_good.append(path_to_img)
    
for i in range(7):
    paths_bad[i] = np.squeeze(np.array(paths_bad[i]))
paths_good = np.squeeze(np.array(paths_good))
   
# Generating static validation data
paths_bad, paths_bad_val = extract_static_val_data(paths_bad, perc = 0.014) #original - 0.017
paths_good, paths_good_val = extract_static_val_data(paths_good, perc = 0.06) #original - 0.11

paths_bad = np.array(paths_bad)
paths_bad_val = np.array(paths_bad_val)
paths_good = np.array(paths_good)
paths_good_val = np.array(paths_good_val)

paths_bad

full_data = np.concatenate((np.repeat(paths_good, 7), paths_bad))
    
#shuffling
idx = np.random.permutation(len(full_data))
full_data = full_data[idx]
# full_data = full_data[:5000]  # debug

#Splitting data 
split_factor = 1024
splitted_data = []

global_batches = int(full_data.shape[0] / split_factor)
for i in range(global_batches):
    batch_data = full_data[i*split_factor: (i+1)*split_factor]
    splitted_data.append(batch_data)
    
splitted_data[-1] = np.concatenate((splitted_data[-1], full_data[len(splitted_data)*split_factor:]))

data = splitted_data


#Loading validation data
paths_val = np.concatenate( (paths_bad_val, paths_good_val ) , axis=0 )
X_val, y_val = load_batch(paths_val)

data_val = (X_val, y_val)
data_val = tf.data.Dataset.from_tensor_slices(data_val).batch(32)#.prefetch(tf.data.AUTOTUNE)   # ToDo: also use .prefetch(tf.data.AUTOTUNE)

### Creating model and training

batch_size = 32 #64
epochs = 20
learning_rate = 0.003

model = ResNet18(num_classes=3409)
model.build((None, 600, 600, 3))
weights_path = '../models/ResNet/ResNet_original_border_600x600.hdf5'

learning_rate = lr_exp_decay(epoch, learning_rate)
print(f'Epoch {epoch}/{epochs}: ')
print('Learnin rate:', learning_rate)
model.compile(loss=tf.keras.losses.MeanSquaredError(),
          optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, 
                                          epsilon=1e-07, decay=0, amsgrad=False), run_eagerly=True)
#     if epoch != 0:
model.load_weights(weights_path)

for i in range(len(data)):
    verbose = 0
    if i % 10 == 0:
        print(i)
        model.save_weights(weights_path)
        verbose = 1

    random.shuffle(data)
    history = trainer(model, 
                      data[i], 
                      data_val,
                      batch_size=batch_size,
                      epochs=epochs,
                      learning_rate=learning_rate,
                      verbose=verbose)    

#     model.save_weights(f'models/Shufflenet/Shufflenet_border_996x996_labels_MG_all_res_996_20.09.h5', save_format='h5')
print('Done, epoch training!')

