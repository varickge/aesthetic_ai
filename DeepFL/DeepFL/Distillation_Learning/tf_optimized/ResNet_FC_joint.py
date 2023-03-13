import tensorflow as tf
import numpy as np
import sys
from ResNet import *
sys.path.append('../../')
from final_utils import *

def fc_model_softmax_t(input_num=16928):
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
    
def fc_model_softmax_s(input_num=16928):
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

model_fc_t = fc_model_softmax_t(input_num=5000)
model_fc_t.load_weights('../../Genetic_Algorithm/best_res_all_res_996/best_custom_95.72.hdf5')
model_fc_s = fc_model_softmax_s(input_num=5000)

class resnet_fc(tf.keras.Model):
    def __init__(self, model_resnet=ResNet18(num_classes=5000), model_fc_t=model_fc_t, model_fc_s=model_fc_s): 
        super(resnet_fc, self).__init__()
        self.layer_resnet = model_resnet
        self.layer_fc_t = model_fc_t
        self.layer_fc_t.trainable = False
        self.layer_fc_s = model_fc_s

    def call(self, img):
        feat_rn = self.layer_resnet(img)
        x_pred_t = self.layer_fc_t(feat_rn)
        x_pred_s = self.layer_fc_s(feat_rn)
        return feat_rn, x_pred_t, x_pred_s
    
loss_tracker = tf.keras.metrics.Mean(name="loss")
mse_metric = tf.keras.metrics.MeanSquaredError(name='mean_squared_error')
ce_metric = tf.keras.metrics.CategoricalCrossentropy(name='categorical_crossentropy')
sce_metric = tf.keras.metrics.SparseCategoricalCrossentropy(name='sparse_categorical_crossentropy')

class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        print(type(data[0]))
        print(x)
        
        feat_t = tf.slice(y, [0, 0], [32, 5000])
        probs_t = tf.slice(y, [0, -2], [32, 2])
        label = tf.slice(y, [0, -1], [32, 1])

        with tf.GradientTape() as tape:
            feat_s, probs_s, label_s = self(x, training=True)  # Forward pass
            # Compute our own loss
            mse = tf.keras.losses.MeanSquaredError()
            ce = tf.keras.losses.CategoricalCrossentropy()
            sce = tf.keras.losses.SparseCategoricalCrossentropy()
            loss = mse(feat_t, feat_s) + ce(probs_t, probs_s) + sce(label, label_s)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mse_metric.update_state(feat_t, feat_s)
        ce_metric.update_state(probs_t, probs_s)
        sce_metric.update_state(label, label_s)
        
        return {"loss": loss_tracker.result(),
                "mean_squared_error": mse_metric.result(),
                "categorical_crossentropy": ce_metric.result(),
                "sparse_categorical_crossentropy": sce_metric.result(),}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mse_metric, ce_metric, sce_metric]

