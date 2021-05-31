import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from learning.data.data import NumpyEncoder
from learning.data.data import load_batch_data

CMAP = cm.tab10

plt.rcParams.update({
    "font.family": "CMU Serif",
    "font.size": 15
})


def evolution_plot_concrete(data, output_path, how='cumulative', distance_metric='heat', experiment_name='LEARNING_RATE', norm=False):
    fig, ax = plt.subplots(sharey=True)
    datas = [datas for datas in data if datas['analysis_type'] == experiment_name]
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
        means = np.mean(distances, axis=0)
        linewidth = 0.05
        if how == 'cumulative':
            means = np.cumsum(means)
            linewidth = 0.5
        if norm:
            means = means / np.max(means)
        stds = np.std(distances, axis=0)

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
        ax2.plot(list(ax2x), [None] + list(np.mean(val_scores, axis=0)), '--', color=color, linewidth=0.5)
    # ax.set_yscale("log")
    if experiment_name == 'NUMBER_LABELS':
        ax.set_xticks(sorted(number_labels_distance_shapes))
        ax.set_xticklabels(np.arange(2, len(number_labels_distance_shapes) * 2 + 1, 2, dtype=int), rotation=45)
        ax.set_xlabel("20 Epoch N labels experiment finish")
    else:
        ax.set_xlabel("Epochs")
    experiment_name = experiment_name.replace('_', ' ').capitalize()
    #ax.set_title(experiment_name)

    ax.set_ylabel("Distance difference (line)")
    ax2.set_ylabel("Validation score (dashed)")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center',
              bbox_to_anchor=(0.5, -0.085), ncol=5, prop={'size': 12}, frameon=False)
    ax.xaxis.labelpad = 20

    fig.tight_layout()
    base_dir = os.path.basename(output_path).lower()
    out_path = os.path.join(output_path, '..', '..', 'lbe_singular')
    os.makedirs(out_path, exist_ok=True)
    if how == "":
        how = 'raw'
    if norm:
        norm_txt = ''
    else:
        norm_txt = '_no_norm'
    fig.savefig(os.path.join(out_path, f'{base_dir}_{how}{norm_txt}_{distance_metric}_{experiment_name.lower().replace(" ", "_")}.pdf'))


if __name__ == '__main__':
    path_batch = './output/learning_BatchEvolutionCallback/'

    for dir in os.listdir(path_batch):
        dir = os.path.join(path_batch, dir)
        print(dir)
        if os.path.isdir(dir):
            # Batch
            batch_data = load_batch_data(dir)
            with open(os.path.join(dir, 'batch_data.json'), 'w') as js_data:
                dumped = json.dumps(batch_data, cls=NumpyEncoder)
                json.dump(dumped, js_data)
            evolution_plot_concrete(batch_data, dir)
