import os
import json
from learning.data.data import NumpyEncoder
from learning.data.data import load_batch_data
from scipy.stats import pearsonr
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
CMAP = cm.tab10

plt.rcParams.update({
    "font.family": "CMU Serif"
})


def plot_points(data, output_path):
    for distance_metric in ['heat', 'silhouette']:
        experiment_names = list(set([experiment['analysis_type'] for experiment in data]))
        _shp = len(experiment_names) / 2
        shp = (math.ceil(_shp), 2)
        fig, axs = plt.subplots(shp[0], shp[1], sharey=False, figsize=(9, 10))
        axs = np.array(axs).flatten()
        if not _shp.is_integer():
            fig.delaxes(axs[-1])
        for experiment_idx, experiment_name in enumerate(experiment_names):
            datas = [datas for datas in data if datas['analysis_type'] == experiment_name]
            ax = axs[experiment_idx]
            for idx, experiment in enumerate(sorted(datas, key=lambda x: float(x['analysis_value']))):
                distances = experiment[f'{distance_metric}_distances']
                val_scores = np.mean(experiment['val_accuracy'], axis=0)
                val_scores = val_scores/max(val_scores)

                label = f'{experiment["analysis_value"]}'  # {experiment["analysis_type"]}_

                color = CMAP(idx)
                means = np.mean(distances, axis=0)
                means = np.cumsum(means)
                means = means/max(means)
                means = np.take(means, np.arange(0, len(means), len(means) / 21, dtype=int))[1:]

                ax.plot(val_scores, means, '.-', label=label, color=color, linewidth=0.05)

            ax.set_xlabel("Topological cumulative")
            ax.set_ylabel("Validation score")
            experiment_name = experiment_name.replace('_', ' ').capitalize()
            ax.set_title(experiment_name)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.1), ncol=5, prop={'size': 7})
            ax.xaxis.labelpad = 20
        fig.tight_layout()
        fig.suptitle(f'{os.path.basename(output_path)} epoch points using {distance_metric.capitalize()} distance',
                     y=1.05)
        fig.savefig(os.path.join(output_path, f'batch_points_{distance_metric}.pdf'))


if __name__ == '__main__':
    path_batch = './output/learning_BatchEvolutionCallback/'

    for dir in os.listdir(path_batch):
        dir = os.path.join(path_batch, dir)
        print(dir)
        if os.path.isdir(dir):
            # Batch
            batch_data = copy.deepcopy(load_batch_data(dir))
            plot_points(batch_data, dir)
