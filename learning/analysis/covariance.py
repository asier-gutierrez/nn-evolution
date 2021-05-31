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
import pathlib
CMAP = cm.tab10

plt.rcParams.update({
    "font.family": "CMU Serif"
})


def evolution_plot_by_dataset_by_batch(data, output_path):
    output_path_covariances = os.path.join(output_path, 'covariances')
    os.makedirs(output_path_covariances, exist_ok=True)
    output_path_qq = os.path.join(output_path, 'qq')
    os.makedirs(output_path_qq, exist_ok=True)
    for distance_metric in ['heat', 'silhouette']:
        experiment_names = list(set([experiment['analysis_type'] for experiment in data]))
        for experiment_idx, experiment_name in enumerate(experiment_names):
            datas = [datas for datas in data if datas['analysis_type'] == experiment_name]
            for idx, experiment in enumerate(sorted(datas, key=lambda x: float(x['analysis_value']))):
                distances = experiment[f'{distance_metric}_distances']
                distances = list(map(np.cumsum, distances))
                distances = list(map(lambda x: np.take(x, np.arange(0, len(distances[0]), len(distances[0]) / 21, dtype=int)[1:]), distances))
                distances = np.array(distances).reshape(-1)
                val_scores = np.array(experiment['val_accuracy']).reshape(-1)
                distances = distances / np.sum(distances)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cov = np.cov(list(zip(distances, val_scores)))
                cov = cov / np.max(cov)
                cax = ax.matshow(cov, interpolation='nearest')
                fig.colorbar(cax)
                ax.set_xticks(np.arange(0, len(distances), len(distances) // 5))
                ax.set_xticklabels(list(map(str, range(1, len(np.arange(0, len(distances), len(distances) // 5)) + 1))))
                ax.set_yticks(np.arange(0, len(distances), len(distances) // 5))
                ax.set_yticklabels(list(map(str, range(1, len(np.arange(0, len(distances), len(distances) // 5)) + 1))))

                fig.suptitle(f'{experiment_name.capitalize().replace("_", " ")} {experiment["analysis_value"]}')
                fig.savefig(os.path.join(output_path_covariances, f'{distance_metric}_{experiment_name}_{experiment["analysis_value"]}.pdf'))
                plt.close(fig)


if __name__ == '__main__':
    path_batch = './output/learning_BatchEvolutionCallback/'

    for dir in os.listdir(path_batch):
        dir = os.path.join(path_batch, dir)
        print(dir)
        if os.path.isdir(dir):
            # Batch
            output = f'./output/lbe/{pathlib.Path(dir).name}'
            os.makedirs(output, exist_ok=True)
            batch_data = load_batch_data(dir)
            evolution_plot_by_dataset_by_batch(batch_data, output)
