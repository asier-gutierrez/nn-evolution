import os
import re
import json
import numpy as np
import igraph


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types taken from:
    https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_epoch_data(path):
    dirs = os.listdir(path)
    experiment_results = list()
    for dir in dirs:
        if os.path.isfile(os.path.join(path, dir)):
            continue
        match = re.search(r'(.*)\_(.*)', dir)
        experiment, value = match.group(1), match.group(2)

        experiment_folder = os.path.join(path, dir)
        experiment_execution = list()
        for dir_execution in os.listdir(experiment_folder):
            graphs = list()
            execution_folder = os.path.join(experiment_folder, dir_execution)
            for idx in range(len(os.listdir(execution_folder)) - 1):
                graph_file = os.path.join(execution_folder, f'{idx}.pickle')
                graphs.append(igraph.read(graph_file))

            with open(os.path.join(path, dir, dir_execution, 'history.json'), 'rb') as f:
                history = json.loads(f.read())
            experiment_execution.append({'analysis_type': experiment, 'analysis_value': value,
                                         'graphs': graphs, 'history': history})
        experiment_results.append(experiment_execution)
    return experiment_results


def load_batch_data(path):
    dirs = os.listdir(path)
    experiment_results = list()
    for dir in dirs:
        if os.path.isfile(os.path.join(path, dir)):
            continue
        match = re.search(r'(.*)\_(.*)', dir)
        experiment, value = match.group(1), match.group(2)

        experiment_folder = os.path.join(path, dir)
        heats, silhouettes, val_accuracies = list(), list(), list()
        for idx, dir_execution in enumerate(os.listdir(experiment_folder)):
            execution_folder = os.path.join(experiment_folder, dir_execution)
            with open(os.path.join(execution_folder, 'data.json'), 'r') as f:
                execution = json.loads(json.loads(f.read()))
                heats.append(execution['distances']['heat'])
                silhouettes.append(execution['distances']['silhouette'])
                val_accuracies.append(execution['history']['val_acc'])
        heats = np.array(heats)
        silhouettes = np.array(silhouettes)
        val_accuracies = np.array(val_accuracies)
        experiment_results.append({'analysis_type': experiment, 'analysis_value': value,
                                   'heat_distances': heats, 'silhouette_distances': silhouettes,
                                   'val_accuracy': val_accuracies})
    return experiment_results


def save_epoch(path, name, graphs, index, history):
    path = os.path.join(path, name, str(index))
    os.makedirs(path, exist_ok=True)
    for idx, graph in enumerate(graphs):
        graph.write_pickle(fname=os.path.join(path, f'{idx}.pickle'))

    dumped = json.dumps(history, cls=NumpyEncoder)
    with open(os.path.join(path, 'history.json'), 'w') as f:
        json.dump(dumped, f)


def save_batch(path, name, distances, index, history):
    path = os.path.join(path, name, str(index))
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'data.json'), 'w') as f:
        dumped = json.dumps({'distances': distances, 'history': history}, cls=NumpyEncoder)
        json.dump(dumped, f)
