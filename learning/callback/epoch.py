from tensorflow import keras
from graph import model2graphig


class EpochEvolutionCallback(keras.callbacks.Callback):
    def __init__(self, method):
        super(EpochEvolutionCallback, self).__init__()
        self.method = method
        self.graphs = list()

    def on_train_begin(self, logs=None):
        self.graphs.append(model2graphig(self.model, method=self.method))

    def on_epoch_end(self, epoch, logs=None):
        self.graphs.append(model2graphig(self.model, method=self.method))

    '''
    def on_train_batch_end(self, epoch, logs=None):
        self.graphs.append(model2graphig(self.model, method=self.method))
    '''