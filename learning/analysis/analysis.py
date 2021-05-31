import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from learning.data.data import load_epoch_data, load_batch_data
from homology import graphigs2vrs_clean
from gtda.diagrams import PairwiseDistance
from graph import connect_stack
import functools
import math
from collections import defaultdict
from learning.data.data import NumpyEncoder
from multiprocessing import Pool
from typing import List
CMAP = cm.tab10

plt.rcParams.update({
    "font.family": "CMU Serif"
})


def _epoch_evolution_compute_parallel(experiment):
    distances = defaultdict(list)
    for distance_metric in ['heat', 'silhouette']:
        for execution in experiment:
            diagrams = graphigs2vrs_clean(execution['graphs'], n_jobs=1)
            distance = list()
            for i in range(1, diagrams.shape[0]):
                dgms = np.take(diagrams, [i - 1, i], axis=0)
                dist = PairwiseDistance(metric=distance_metric, n_jobs=1,
                                        metric_params={'n_bins': 200}).fit_transform(dgms)
                distance.append(dist[0][1])
            distances[f'{distance_metric}_distances'].append(distance)

    val_scores = list()
    for execution in experiment:
        validation_history = json.loads(execution['history'])['val_acc']
        val_score = list()
        for i in range(1, len(validation_history)):
            val_score.append(np.abs(
                validation_history[i - 1] - validation_history[i]))
        val_scores.append(val_score)

    r = np.arange(0, np.array(list(distances.values())).shape[2]).tolist()
    _plot_data = {'distances': distances, 'val_score': val_scores, 'r': r,
                  'analysis_type': experiment[0]["analysis_type"],
                  'analysis_value': experiment[0]["analysis_value"]}
    return _plot_data


def evolution_compute_by_dataset_by_epoch(data, output_path):
    with Pool(4) as p:
        plot_data = p.map(_epoch_evolution_compute_parallel, data)

    with open(os.path.join(output_path, f'plot_data.json'), 'w') as plot_data_file:
        json.dump(plot_data, plot_data_file)


def evolution_plot_by_dataset_by_epoch(output_path):
    with open(os.path.join(output_path, f'plot_data.json'), 'r') as plot_data_file:
        plot_data = json.load(plot_data_file)
    for distance_metric in ['heat', 'silhouette']:
        experiment_names = list(set([experiment['analysis_type'] for experiment in plot_data]))
        _shp = len(experiment_names) / 2
        shp = (math.ceil(_shp), 2)
        fig, axs = plt.subplots(shp[0], shp[1], sharey=True, figsize=(9, 10))
        axs = np.array(axs).flatten()
        if not _shp.is_integer():
            fig.delaxes(axs[-1])

        for experiment_idx, experiment_name in enumerate(experiment_names):
            datas = [datas for datas in plot_data if datas['analysis_type'] == experiment_name]
            ax = axs[experiment_idx]
            ax2 = ax.twinx()
            number_labels_distance_shapes = list()

            for idx, experiment in enumerate(sorted(datas, key=lambda x: float(x['analysis_value']))):
                distances = np.array(experiment['distances'][f'{distance_metric}_distances'])
                distances = (distances - np.min(distances)) / np.ptp(distances)
                val_scores = np.array(experiment['val_score'])

                label = f'{experiment["analysis_value"]}'
                r = np.arange(0, distances.shape[1])
                color = CMAP(idx)
                means = np.mean(distances, axis=0)
                stds = np.std(distances, axis=0)
                ax.plot(r, means, '-', label=label, color=color, linewidth=0.05)
                # ax.fill_between(x=r, y1=np.add(means, -stds), y2=np.add(means, stds), color=color[0:3] + (0.3,))
                if experiment_name != 'NUMBER_LABELS':
                    ax.set_xticks(np.arange(0, distances.shape[1]))
                    ax.set_xticklabels(np.arange(0, val_scores.shape[1]+1, dtype=int), rotation=45)
                else:
                    number_labels_distance_shapes.append(distances.shape[1])
                ax2.plot(np.arange(1, distances.shape[1]),
                         np.mean(val_scores, axis=0), '--', color=color)
            # ax.set_yscale("log")
            if experiment_name == 'NUMBER_LABELS':
                ax.set_xticks(sorted(number_labels_distance_shapes))
                ax.set_xticklabels(np.arange(2, len(number_labels_distance_shapes) * 2 + 1, 2, dtype=int), rotation=45)
                ax.set_xlabel("20 Epoch N labels experiment finish")
            else:
                ax.set_xlabel("Epochs")
            experiment_name = experiment_name.replace('_', ' ').capitalize()
            ax.set_title(experiment_name)

            ax.set_ylabel("Distance difference")
            ax2.set_ylabel("Valid. score difference")

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), ncol=5, prop={'size': 7})
            ax.xaxis.labelpad = 20
        fig.tight_layout()
        fig.suptitle(f'{os.path.basename(output_path)} epoch difference using {distance_metric.capitalize()} distance',
                     y=1.05)
        fig.savefig(os.path.join(output_path, f'evolution_by_dataset_{distance_metric}.pdf'))


def evolution_plot_by_dataset_by_batch(data, output_path, how='cumulative'):
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
            ax2 = ax.twinx()
            number_labels_distance_shapes = list()
            for idx, experiment in enumerate(sorted(datas, key=lambda x: float(x['analysis_value']))):
                distances = experiment[f'{distance_metric}_distances']
                if how != 'cumulative':
                    distances = (distances - np.min(distances)) / np.ptp(distances)
                val_scores = experiment['val_accuracy']

                label = f'{experiment["analysis_value"]}'  # {experiment["analysis_type"]}_
                r = np.arange(0, distances.shape[1])
                color = CMAP(idx)
                if how == 'derivative':
                    means = np.mean(distances, axis=0)
                    means = np.cumsum(means)
                    means_range = range(len(means))
                    pol, res, _, _, _ = np.polyfit(means_range, means, 17, full=True)
                    polder = np.polyder(pol, m=1)
                    means_range_space = np.linspace(min(means_range), max(means_range))
                    poly_range = np.polyval(pol, means_range_space)
                    poly_er_range = np.polyval(polder, means_range_space)
                    poly_er_range = (poly_er_range - min(poly_er_range)) / (max(poly_er_range) - min(poly_er_range))

                else:
                    means = np.mean(distances, axis=0)
                linewidth = 0.05
                if how == 'cumulative':
                    means = np.cumsum(means)
                    means = means / np.max(means)
                    linewidth = 0.5
                if how == 'smoothed':
                    means = smooth(means, 0.7)
                stds = np.std(distances, axis=0)

                if how == 'derivative':
                    ax.plot(means_range_space, poly_er_range, '-', label=label, color=color, linewidth=linewidth)
                else:
                    ax.plot(r, means, '-', label=label, color=color, linewidth=linewidth)
                # ax.fill_between(x=r, y1=np.add(means, -stds), y2=np.add(means, stds), color=color[0:3] + (0.3,))
                if experiment_name != 'NUMBER_LABELS':
                    ax.set_xticks(np.arange(0, distances.shape[1] + 1, distances.shape[1] / val_scores.shape[1]))
                    ax.set_xticklabels(np.arange(0, val_scores.shape[1] + 1, dtype=int), rotation=45)
                else:
                    number_labels_distance_shapes.append(distances.shape[1])
                #ax.set_xticks(r)
                #ax.set_xticklabels(np.arange(0, r.shape[0], dtype=int), rotation=45)
                ax2x = np.arange(0, distances.shape[1]+1, distances.shape[1] / val_scores.shape[1])
                ax2.set_xticks(ax2x)
                ax2.set_xticklabels(np.arange(0, ax2x.shape[0], dtype=int), rotation=45)
                if how == 'derivative':
                    ax2.plot(list(ax2x), [None] + list(1 - np.mean(val_scores, axis=0)), '--', color=color, linewidth=0.5)
                else:
                    ax2.plot(list(ax2x), [None] + list(np.mean(val_scores, axis=0)), '--', color=color, linewidth=0.5)
            # ax.set_yscale("log")
            if experiment_name == 'NUMBER_LABELS':
                ax.set_xticks(sorted(number_labels_distance_shapes))
                ax.set_xticklabels(np.arange(2, len(number_labels_distance_shapes) * 2 + 1, 2, dtype=int), rotation=45)
                ax.set_xlabel("20 Epoch N labels experiment finish")
            else:
                ax.set_xlabel("Epochs")
            experiment_name = experiment_name.replace('_', ' ').capitalize()
            ax.set_title(experiment_name)

            if how == 'derivative':
                ax.set_ylabel("Distance cumulative derivative")
                ax2.set_ylabel("Validation error")
            else:
                ax.set_ylabel("Distance difference (line)")
                ax2.set_ylabel("Validation score (dashed)")

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.12), ncol=5, prop={'size': 7})
            ax.xaxis.labelpad = 20
        fig.tight_layout()
        fig.suptitle(f'{os.path.basename(output_path)} epoch difference using {distance_metric.capitalize()} distance',
                     y=1.05)
        fig.savefig(os.path.join(output_path, f'evolution_by_dataset_{distance_metric}.pdf'))


# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def stack_matrix_by_dataset(data, output_path, n_graphs=10):
    for distance_metric in ['heat', 'silhouette']:
        n_runs = len(data[0])
        distances = list()
        for n_run in range(n_runs):
            gs = list()
            for idx, experiment in enumerate(data):
                min_n_graphs = min(len(experiment[n_run]['graphs']), n_graphs)
                gs.append(connect_stack(experiment[n_run]['graphs'][0:min_n_graphs]))
            diagrams = graphigs2vrs_clean(gs)
            distance_run = PairwiseDistance(metric=distance_metric, n_jobs=1,
                                            metric_params={'n_bins': 200}).fit_transform(diagrams)
            distances.append(distance_run)
        distances = functools.reduce(lambda x1, x2: x1 + x2, distances) / n_runs
        with open(os.path.join(output_path, f'{distance_metric}_stack_comparison_matrix.npy'), 'wb') as f:
            np.save(f, distances)


if __name__ == '__main__':
    path_epoch = './output/learning_EpochEvolutionCallback/'
    path_batch = './output/learning_BatchEvolutionCallback/'

    # Evolution plot by dataset
    '''
    for dir in os.listdir(path_epoch):
        dir = os.path.join(path_epoch, dir)
        print(dir)
        if os.path.isdir(dir):
            # Epoch
            epoch_data = load_epoch_data(dir)
            #evolution_compute_by_dataset_by_epoch(epoch_data, dir)
            #evolution_plot_by_dataset_by_epoch(dir)
            stack_matrix_by_dataset(epoch_data, dir, n_graphs=21)
    '''
    for dir in os.listdir(path_batch):
        dir = os.path.join(path_batch, dir)
        print(dir)
        if os.path.isdir(dir):
            # Batch
            batch_data = load_batch_data(dir)
            with open(os.path.join(dir, 'batch_data.json'), 'w') as js_data:
                dumped = json.dumps(batch_data, cls=NumpyEncoder)
                json.dump(dumped, js_data)
            evolution_plot_by_dataset_by_batch(batch_data, dir)
