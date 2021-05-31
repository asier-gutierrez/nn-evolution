import os
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from itertools import combinations
from graph import model2graphig
from homology import graphigs2vrs
from gtda.diagrams import PairwiseDistance, Filtering
from learning.conf.conf_cifar10 import DROPOUT_SEED, ANALYSIS_TYPES
import functools
from learning.callback.epoch import EpochEvolutionCallback
from learning.data.data import save_epoch

# MLP training
batch_size = 256
num_classes = 10
epochs = 20
TIMES = 5
EXPERIMENT_NAME = "CIFAR10MLP"


if __name__ == '__main__':
    # Computing
    epoch_executed = False
    callback = EpochEvolutionCallback
    output_path = os.path.join(f'./output/learning_{callback.__name__}/', EXPERIMENT_NAME)
    os.makedirs(output_path, exist_ok=True)
    for metric in ['silhouette', 'heat']:
        distances_all = list()
        for execution_number in range(TIMES):
            graphs = list()
            for analysis in ANALYSIS_TYPES:
                analysis_type, analysis_values = analysis['name'], analysis['values']
                for analysis_value in analysis_values:
                    model = tf.keras.Sequential()

                    if analysis_type == 'LAYER_SIZE':
                        model.add(Dense(analysis_value, activation='relu', input_shape=(3072,),
                                        kernel_initializer='glorot_uniform'))
                        model.add(Dropout(0.2, seed=DROPOUT_SEED))
                        model.add(Dense(analysis_value, activation='relu', kernel_initializer='glorot_uniform'))
                        model.add(Dropout(0.2, seed=DROPOUT_SEED))
                    elif analysis_type == 'NUMBER_LAYERS':
                        model.add(
                            Dense(512, activation='relu', input_shape=(3072,), kernel_initializer='glorot_uniform'))
                        model.add(Dropout(0.2, seed=DROPOUT_SEED))
                        for _ in range(analysis_value - 1):
                            model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
                            model.add(Dropout(0.2, seed=DROPOUT_SEED))
                    elif analysis_type == 'DROPOUT':
                        model.add(
                            Dense(512, activation='relu', input_shape=(3072,), kernel_initializer='glorot_uniform'))
                        model.add(Dropout(analysis_value, seed=DROPOUT_SEED))
                        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
                        model.add(Dropout(analysis_value, seed=DROPOUT_SEED))
                    else:
                        model.add(
                            Dense(512, activation='relu', input_shape=(3072,), kernel_initializer='glorot_uniform'))
                        model.add(Dropout(0.2, seed=DROPOUT_SEED))
                        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
                        model.add(Dropout(0.2, seed=DROPOUT_SEED))

                    if analysis_type == 'NUMBER_LABELS':
                        model.add(Dense(analysis_value, activation='softmax'))
                    else:
                        model.add(Dense(num_classes, activation='softmax'))

                    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                    x_train = x_train.reshape(50000, 3072)
                    x_test = x_test.reshape(10000, 3072)
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

                    model.summary()
                    if analysis_type == 'LEARNING_RATE':
                        lr = analysis_value
                    else:
                        lr = 0.001
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=RMSprop(learning_rate=lr),
                                  metrics=['accuracy'])
                    if type(callback) == EpochEvolutionCallback and epoch_executed:
                        evolution_callback = []
                    else:
                        evolution_callback = [callback(method='reverse')]
                    history = model.fit(x_train, y_train,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_split=0.1,
                                        callbacks=evolution_callback)
                    if len(evolution_callback):
                        evolution_callback = evolution_callback[0]
                    if type(evolution_callback) == EpochEvolutionCallback:
                        save_epoch(path=output_path, name=f'{analysis_type}_{analysis_value}',
                                   graphs=evolution_callback.graphs, index=execution_number,
                                   history=history.history)
                    graphs.append(model2graphig(model, method='reverse'))
            diagrams = graphigs2vrs(graphs)

            # Filter
            print("Before filtering", diagrams.shape)
            diagrams = Filtering(epsilon=0.01).fit_transform(diagrams)
            print("After filtering", diagrams.shape)

            # Replace
            diagrams[diagrams == np.Inf] = 1.0

            # Compute
            start = time.time()
            distances = list()
            for idx_0, idx_1 in tqdm(combinations(range(diagrams.shape[0]), r=2)):
                dist = PairwiseDistance(metric=metric, n_jobs=1, metric_params={'n_bins': 200}).fit_transform(
                    np.take(diagrams, [idx_0, idx_1], axis=0))
                distances.append((idx_0, idx_1, dist[0][1]))
            end = time.time()
            print("Elapsed time:", end - start)
            max_n = max([max(distances, key=lambda x: x[0])[0], max(distances, key=lambda x: x[1])[1]]) + 1
            final_data = np.zeros((max_n, max_n))
            for d in distances:
                final_data[d[0]][d[1]] = d[2]
                final_data[d[1]][d[0]] = d[2]
            distances_all.append(final_data)
        distances_all = functools.reduce(lambda x1, x2: x1 + x2, distances_all) / TIMES
        with open(os.path.join(output_path, f'{metric}_comparison_matrix.npy'), 'wb') as f:
            np.save(f, distances_all)
        if callback == EpochEvolutionCallback:
            epoch_executed = True
