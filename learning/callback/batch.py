from tensorflow import keras
from graph import model2graphig
from homology import graphigs2vrs_clean
from collections import defaultdict
from gtda.diagrams import PairwiseDistance


class BatchEvolutionCallback(keras.callbacks.Callback):
    def __init__(self, method, distance_metrics, n_jobs=2):
        super(BatchEvolutionCallback, self).__init__()
        self.method = method
        self.distance_metrics = distance_metrics
        self.n_jobs = n_jobs
        self.previous_graph = None
        self.distances = defaultdict(list)

    def on_train_begin(self, logs=None):
        self.previous_graph = model2graphig(self.model, method=self.method)

    def on_train_batch_end(self, epoch, logs=None):
        network_graph = model2graphig(self.model, method=self.method)
        diagrams = graphigs2vrs_clean([self.previous_graph, network_graph], n_jobs=self.n_jobs)
        for distance_metric in self.distance_metrics:
            distances = PairwiseDistance(metric=distance_metric, n_jobs=self.n_jobs,
                                         metric_params={'n_bins': 200}).fit_transform(diagrams)
            self.distances[distance_metric].append(distances[0][1])
        self.previous_graph = network_graph
