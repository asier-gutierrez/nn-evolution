import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from learning.conf.conf_reuters import DROPOUT_SEED, ANALYSIS_TYPES
from learning.callback.batch import BatchEvolutionCallback
from learning.data.data import save_batch
from multiprocessing import Pool

# MLP training
batch_size = 256
num_classes = 46
epochs = 20
TIMES = 5
MAXLEN = 100
EXPERIMENT_NAME = "REUTERS"


def vectorize_sequences(sequences, dimension=5000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


def perform_experiment(params):
    analysis_type = params['analysis_type']
    analysis_value = params['analysis_value']
    distance_metrics = params['distance_metrics']
    execution_number = params['execution_number']
    output_path = params['output_path']
    model = tf.keras.Sequential()

    if analysis_type == 'LAYER_SIZE':
        model.add(Dense(analysis_value, activation='relu', input_shape=(5000,),
                        kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.2, seed=DROPOUT_SEED))
        model.add(Dense(analysis_value, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.2, seed=DROPOUT_SEED))
    elif analysis_type == 'NUMBER_LAYERS':
        model.add(
            Dense(512, activation='relu', input_shape=(5000,), kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.2, seed=DROPOUT_SEED))
        for _ in range(analysis_value - 1):
            model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
            model.add(Dropout(0.2, seed=DROPOUT_SEED))
    elif analysis_type == 'DROPOUT':
        model.add(
            Dense(512, activation='relu', input_shape=(5000,), kernel_initializer='glorot_uniform'))
        model.add(Dropout(analysis_value, seed=DROPOUT_SEED))
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dropout(analysis_value, seed=DROPOUT_SEED))
    else:
        model.add(
            Dense(512, activation='relu', input_shape=(5000,), kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.2, seed=DROPOUT_SEED))
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.2, seed=DROPOUT_SEED))

    if analysis_type == 'NUMBER_LABELS':
        model.add(Dense(analysis_value, activation='softmax'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    # the data, split between train and test sets
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=5000)
    # Vectorized training data
    x_train = vectorize_sequences(train_data)
    # Vectorized test data
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    # convert class vectors to binary class matrices
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
    distance_metrics = ['silhouette', 'heat']
    output_path = os.path.join(f'./output/learning_{BatchEvolutionCallback.__name__}/', EXPERIMENT_NAME)
    os.makedirs(output_path, exist_ok=True)

    datas = list()
    for execution_number in range(TIMES):
        for analysis in ANALYSIS_TYPES:
            analysis_type, analysis_values = analysis['name'], analysis['values']
            for analysis_value in analysis_values:
                datas.append(
                    {'analysis_type': analysis_type, 'analysis_value': analysis_value, 'output_path': output_path,
                     'execution_number': execution_number, 'distance_metrics': distance_metrics})
    with Pool(30) as pool:
        pool.map(perform_experiment, datas)
