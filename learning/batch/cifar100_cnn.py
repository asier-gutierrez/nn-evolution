import os
from tqdm import tqdm
import uuid
import numpy as np
from multiprocessing import Pool
import json


# MLP training
batch_size = 256
num_classes = 100
epochs = 20
TIMES = 5
EXPERIMENT_NAME = "CIFAR100CNN"


def perform_experiment(params):
    import tensorflow as tf
    from tensorflow.keras.datasets import cifar100
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
    from tensorflow.keras.optimizers import RMSprop
    from learning.conf.conf_cifar100 import DROPOUT_SEED, ANALYSIS_TYPES
    from learning.callback.batch import BatchEvolutionCallback
    from learning.data.data import save_batch
    analysis_type = params['analysis_type']
    analysis_value = params['analysis_value']
    distance_metrics = params['distance_metrics']
    execution_number = params['execution_number']
    output_path = params['output_path']
    model_tl = params['model_tl']
    model_tl = tf.keras.models.load_model(model_tl)

    for layer in model_tl.layers[:10]:
        layer.trainable = False

    if analysis_type == 'LAYER_SIZE':
        x = Dense(analysis_value, activation='relu', kernel_initializer='glorot_uniform')(model_tl.layers[9].output)
        x = Dropout(0.2, seed=DROPOUT_SEED)(x)
        x = Dense(analysis_value, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.2, seed=DROPOUT_SEED)(x)
    elif analysis_type == 'NUMBER_LAYERS':
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(model_tl.layers[9].output)
        x = Dropout(0.2, seed=DROPOUT_SEED)(x)
        for _ in range(analysis_value - 1):
            x = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(x)
            x = Dropout(0.2, seed=DROPOUT_SEED)(x)
    elif analysis_type == 'DROPOUT':
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(model_tl.layers[9].output)
        x = Dropout(analysis_value, seed=DROPOUT_SEED)(x)
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(analysis_value, seed=DROPOUT_SEED)(x)
    else:
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(model_tl.layers[9].output)
        x = Dropout(0.2, seed=DROPOUT_SEED)(x)
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dropout(0.2, seed=DROPOUT_SEED)(x)

    if analysis_type == 'NUMBER_LABELS':
        x = Dense(analysis_value, activation='softmax')(x)
    else:
        x = Dense(num_classes, activation='softmax')(x)

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = np.squeeze(y_train, axis=1)
    y_test = np.squeeze(y_test, axis=1)
    if analysis_type == 'NUMBER_LABELS':
        # Select labels
        labels = sorted(list(set(y_train)))
        labels = labels[:analysis_value]

        # Train filter
        train_idxs = np.where(np.isin(y_train, labels))
        x_train = x_train[train_idxs]
        y_train = y_train[train_idxs]

        # Test filter
        test_idxs = np.where(np.isin(y_test, labels))
        x_test = x_test[test_idxs]
        y_test = y_test[test_idxs]

        # Categories
        y_train = tf.keras.utils.to_categorical(y_train, len(labels))
        y_test = tf.keras.utils.to_categorical(y_test, len(labels))
    elif analysis_type == 'INPUT_ORDER':
        # Get indexes
        # np.random.seed(analysis_value)
        train_idxs = np.random.permutation(len(x_train))
        test_idxs = np.random.permutation(len(x_test))

        # Train randomization
        x_train = x_train[train_idxs]
        y_train = y_train[train_idxs]

        # Test randomization
        x_test = x_test[test_idxs]
        y_test = y_test[test_idxs]

        # Categories
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        # Reset seed for model initialization. BE CAREFUL.
        # np.random.seed(SEED)
        # np.random.set_state(RANDOM_STATE)
    else:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = tf.keras.Model(inputs=model_tl.inputs, outputs=x)
    model.summary()
    if analysis_type == 'LEARNING_RATE':
        lr = analysis_value
    else:
        lr = 0.001
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=lr),
                  metrics=['accuracy'])
    evolution_callback = BatchEvolutionCallback(method='reverse', distance_metrics=distance_metrics, n_jobs=1)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[evolution_callback])

    save_batch(path=output_path, name=f'{analysis_type}_{analysis_value}',
               distances=evolution_callback.distances, index=execution_number,
               history=history.history)


if __name__ == '__main__':
    # Computing
    output_path = os.path.join(f'./output/learning_BatchEvolutionCallback/', EXPERIMENT_NAME)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'experiments.json'), 'r') as f:
        datas = json.load(f)

    with Pool(30) as pool:
        pool.map(perform_experiment, datas)
