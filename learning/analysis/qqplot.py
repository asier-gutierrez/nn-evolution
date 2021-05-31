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


def evolution_plot_by_dataset_by_batch(data, output_path):
    for distance_metric in ['heat', 'silhouette']:
        experiment_names = list(set([experiment['analysis_type'] for experiment in data]))
        _shp = len(experiment_names) / 2
        shp = (math.ceil(_shp), 2)
        fig, axs = plt.subplots(shp[0], shp[1], sharey=True, figsize=(9, 10))
        axs = np.array(axs).flatten()
        if not _shp.is_integer():
            fig.delaxes(axs[-1])
        for experiment_idx, experiment_name in enumerate(experiment_names):
            datas = [datas for datas in data if datas['analysis_type'] == experiment_name]
            ax = axs[experiment_idx]
            for idx, experiment in enumerate(sorted(datas, key=lambda x: float(x['analysis_value']))):
                distances = experiment[f'{distance_metric}_distances']
                distances = list(map(np.cumsum, distances))
                distances = list(map(lambda x: np.take(x, np.arange(0, len(x), len(x) / 20, dtype=int)), distances))
                distances = np.array(distances).reshape(-1)
                val_scores = np.array(experiment['val_accuracy']).reshape(-1)

                label = f'{experiment["analysis_value"]}'  # {experiment["analysis_type"]}_

                color = CMAP(idx)
                distances = distances / np.sum(distances)
                ax.scatter(np.sort(distances), np.sort(val_scores), label=label, color=color)
            experiment_name = experiment_name.replace('_', ' ').capitalize()
            ax.set_title(experiment_name)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), ncol=5, prop={'size': 7})
            ax.xaxis.labelpad = 20
        fig.tight_layout()
        fig.suptitle(f'{os.path.basename(output_path)} qqplot using {distance_metric.capitalize()} distance',
                     y=1.05)
        fig.savefig(os.path.join(output_path, f'qqplot_{distance_metric}.pdf'))


if __name__ == '__main__':
    path_batch = './output/learning_BatchEvolutionCallback/'

    for dir in os.listdir(path_batch):
        dir = os.path.join(path_batch, dir)
        print(dir)
        if os.path.isdir(dir):
            # Batch
            batch_data = copy.deepcopy(load_batch_data(dir))
            evolution_plot_by_dataset_by_batch(batch_data, dir)
