import os
from tqdm import tqdm
import uuid
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import RMSprop
from multiprocessing import Pool
from learning.conf.conf_cifar100 import DROPOUT_SEED, ANALYSIS_TYPES
from learning.callback.batch import BatchEvolutionCallback
from learning.data.data import save_batch

# MLP training
batch_size = 256
num_classes = 100
epochs = 20
TIMES = 5
EXPERIMENT_NAME = "CIFAR100CNN"

if __name__ == '__main__':
    # Computing
    distance_metrics = ['silhouette', 'heat']
    output_path = os.path.join(f'./output/learning_{BatchEvolutionCallback.__name__}/', EXPERIMENT_NAME)
    os.makedirs(output_path, exist_ok=True)
    model_tl = tf.keras.Sequential()
    model_tl.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                        input_shape=(32, 32, 3)))
    model_tl.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model_tl.add(MaxPooling2D((2, 2)))
    model_tl.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_tl.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_tl.add(MaxPooling2D((2, 2)))
    model_tl.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_tl.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_tl.add(MaxPooling2D((2, 2)))
    model_tl.add(Flatten())
    model_tl.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model_tl.add(Dropout(0.2, seed=DROPOUT_SEED))
    model_tl.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform'))
    model_tl.add(Dropout(0.2, seed=DROPOUT_SEED))
    model_tl.add(Dense(num_classes, activation='softmax'))
    model_tl.summary()
    model_tl.compile(loss='categorical_crossentropy',
                     optimizer=RMSprop(),
                     metrics=['accuracy'])
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    history = model_tl.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_split=0.1)

    datas = list()
    for execution_number in range(TIMES):
        for analysis in ANALYSIS_TYPES:
            analysis_type, analysis_values = analysis['name'], analysis['values']
            for analysis_value in analysis_values:
                tmp_model = os.path.join(output_path, 'tmp')
                os.makedirs(tmp_model, exist_ok=True)
                tmp_model = os.path.join(output_path, str(uuid.uuid4()))
                model_tl.save(tmp_model)
                datas.append(
                    {'analysis_type': analysis_type, 'analysis_value': analysis_value, 'output_path': output_path,
                     'execution_number': execution_number, 'distance_metrics': distance_metrics, 'model_tl': tmp_model})

    with open(os.path.join(output_path, 'experiments.json'), 'w') as f:
        json.dump(datas, f)
