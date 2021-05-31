import os
import json
from learning.data.data import NumpyEncoder
from learning.data.data import load_batch_data
from scipy.stats import pearsonr
import numpy as np
from collections import defaultdict
import pandas as pd


def compute_correlations(data):
    correlations = defaultdict(list)
    for distance_metric in ['heat', 'silhouette']:
        experiment_names = list(set([experiment['analysis_type'] for experiment in data]))
        for experiment_idx, experiment_name in enumerate(experiment_names):
            datas = [datas for datas in data if datas['analysis_type'] == experiment_name]
            for idx, experiment in enumerate(sorted(datas, key=lambda x: float(x['analysis_value']))):
                distances = experiment[f'{distance_metric}_distances']
                val_scores = experiment['val_accuracy']
                rs, ps = list(), list()
                for i in range(len(distances)):
                    distances[i] = np.cumsum(distances[i])
                    dist = np.take(distances[i], np.arange(1, len(distances[i]) + 1, len(distances[i]) / 20, dtype=int))
                    r, p = pearsonr(dist, val_scores[i])
                    rs.append(r)
                    ps.append(p)
                correlations[distance_metric].append({
                    'r_mean': np.mean(rs),
                    'r_std': np.std(rs),
                    'p_mean': np.mean(ps),
                    'rs': rs,
                    'ps': ps,
                    'analysis_type': experiment_name.replace('_', ' ').capitalize(),
                    'analysis_value': experiment["analysis_value"]
                })
    return correlations


if __name__ == '__main__':
    path_batch = './output/learning_BatchEvolutionCallback/'

    for dir in os.listdir(path_batch):
        dir = os.path.join(path_batch, dir)
        print(dir)
        if os.path.isdir(dir):
            # Batch
            batch_data = load_batch_data(dir)
            correlations = compute_correlations(batch_data)
            with open(os.path.join(dir, 'correlations.json'), 'w') as js_data:
                dumped = json.dumps(correlations, cls=NumpyEncoder)
                json.dump(json.loads(dumped), js_data, indent=4)
            for k in correlations.keys():
                df = pd.DataFrame(correlations[k])[['analysis_type', 'analysis_value', 'r_mean', 'r_std']]
                with open(os.path.join(dir, f'{k}_correlations_table.txt'), 'w') as f:
                    f.writelines([
                        "\\begin{table}[]\n",
                        "\\centering\n",
                        "\\begin{tabular}{@{}lrrr@{}}\n",
                        "\\toprule\n",
                        "Analysis type & \\multicolumn{1}{l}{Analysis Value} & \\multicolumn{1}{l}{Pearson's r mean} & \\multicolumn{1}{l}{Pearson's r standard deviation} \\\\ \\midrule\n"
                    ])
                    for idx, row in df.iterrows():
                        txt = f"{row['analysis_type']} & {row['analysis_value']} & {row['r_mean']:.4f} & {row['r_std']:.4f} \\\\ \n"
                        f.write(txt)
                    caption = "{" + f'{os.path.basename(dir)} Pearson\'s correlation using {k.capitalize()} distance.' + "}"
                    f.writelines([
                        "\\bottomrule\n",
                        "\\end{tabular}\n",
                        f"\\caption{caption}\n",
                        "\\end{table}\n"
                    ])

