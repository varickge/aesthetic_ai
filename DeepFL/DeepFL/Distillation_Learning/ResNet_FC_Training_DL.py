import sys
sys.path.append('../')

from final_utils import *

from ResNet_FC import *
from ResNet import *
import time
import argparse
import logging
import os
from tensorflow.keras.models import load_model
import pickle

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[1], True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
        
root_path = generate_root_path()
SC_CE_KLD = tf.keras.losses.Huber()

logging.basicConfig(filename="log_resnet_fc.txt", level=logging.INFO)

def read_epoch():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', required=True, help='number of epoch')
    args = parser.parse_args()
    return args.epoch

epoch = int(read_epoch())
### DataLoader and other nesseccary functions
logging.info(f'EPOCH: {epoch}/50')

def lr_exp_decay(epoch, lr):
    k = 0.04
    return lr * np.exp(-k*epoch)

def load_batch(paths):
    images = []
    labels = []
    for i,path in enumerate(paths):
        
        try:
            img = cv2.imread(path, cv2.COLOR_BGR2RGB)
            img = img.astype('float32')
            img /= 255 
            img_basename = os.path.basename(path).split('.')[0] + '.npy'
            label = np.load(root_path + f'Data/AesthAI/alm/splitted/alm_train/predictions/mg_cnn_fc/{img_basename}')
        except:
            continue
            
        images.append(img)
        labels.append(label)
    images = np.stack(images)
    
    return images, labels

def trainer(model, data, data_val,batch_size=128, epochs=50, learning_rate=0.03, verbose=0):
    X_train, y_train = load_batch(data)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)#.prefetch(tf.data.AUTOTUNE)  # ToDo: here use .cache()
    history = model.fit(train_dataset,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=verbose,
                        validation_data = data_val)

    return history

main_path = f'{root_path}Data/AesthAI/alm/splitted/alm_train/'
paths_bad = []
paths_good = []
    
for i in range(7):
    alm_train_bad = open(f'{main_path}data_bad{i+1}.json')
    bad_data = json.load(alm_train_bad)
    
    for data in bad_data:
        path_to_img = main_path + f'images/{data["label"]}/{data["splitted"]}_resized_600x600/' + data['name']
        paths_bad.append(path_to_img)
        
alm_train_good = open(f'{main_path}/data_good1.json')
good_data = json.load(alm_train_good)
for data in good_data:
    path_to_img = main_path + f'images/{data["label"]}/{data["splitted"]}_resized_600x600/' + data['name']
    paths_good.append(path_to_img)
    
for i in range(7):
    paths_bad[i] = np.squeeze(np.array(paths_bad[i]))
paths_good = np.squeeze(np.array(paths_good))
   
# Generating static validation data
paths_bad, paths_bad_val = extract_static_val_data(paths_bad, perc = 0.014) #original - 0.017
paths_good, paths_good_val = extract_static_val_data(paths_good, perc = 0.04) #original - 0.11

paths_bad = np.array(paths_bad)
paths_bad_val = np.array(paths_bad_val)
paths_good = np.array(paths_good)
paths_good_val = np.array(paths_good_val)

full_data = np.concatenate((np.repeat(paths_good, 7), paths_bad))
    
#shuffling
idx = np.random.permutation(len(full_data))
full_data = full_data[idx]
# full_data = full_data[:5000]  # debug


split_factor = 1024
splitted_data = []

global_batches = int(full_data.shape[0] / split_factor)
for i in range(global_batches):
    batch_data = full_data[i*split_factor: (i+1)*split_factor]
    splitted_data.append(batch_data)
    
splitted_data.append(full_data[len(splitted_data)*split_factor:])

data = splitted_data


paths_val = np.concatenate((paths_bad_val, paths_good_val), axis=0)
print(paths_val.shape)
X_val, y_val = load_batch(paths_val)
print('ended loading val')
data_val = (X_val, y_val)
data_val = tf.data.Dataset.from_tensor_slices(data_val).batch(32)#.prefetch(tf.data.AUTOTUNE)   # ToDo: also use .prefetch(tf.data.AUTOTUNE)
logging.info(f'End of data loading')
print('end of data loading')

batch_size = 32
epochs = 50
learning_rate = 0.003

def fc_model_softmax(input_num=16928):
    input_ = Input(shape=(input_num,))
    x = Dense(1024, kernel_initializer='he_normal', activation='relu')(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(256, kernel_initializer='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    pred = Dense(2, activation='softmax')(x)

    model = Model(input_,pred)
    return model

print(f'Epoch {epoch+1}/{epochs}: ')
logging.info(f'Epoch {epoch+1}/{epochs}: ')
learning_rate = lr_exp_decay(epoch, learning_rate)
print('Learnin rate:', learning_rate)
weights_path = '../models/Softmax/ResNet_FC/'
if epoch == 0:
    model_resnet = ResNet18(num_classes=5000)
    model_fc = fc_model_softmax(input_num=5000)
    model = resnet_fc(model_resnet, model_fc)
    model.compile(loss=SC_CE_KLD, optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
else:
    model = load_model(weights_path)
    
random.shuffle(data)
logging.info(f'Start of training')
print('start of training')
for i in range(len(data)):
    verbose = 0
    if i % 10 == 0 and i != 0:
        logging.info(f'Batch data[{i}]')
        print('Batch data --', i)
        verbose = 1
        model.save(weights_path, save_format="tf")
    batch_data = np.array(data[i])
    random.shuffle(batch_data)
    history = trainer(model, batch_data, data_val,  batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, verbose=verbose)    

model.save(weights_path, save_format="tf")  
print('Done, epoch training!')
logging.info(f'Done, epoch training!')