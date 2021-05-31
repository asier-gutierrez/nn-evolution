import os
import pandas as pd


PATH = 'output/learning_BatchEvolutionCallback'
if __name__ == '__main__':
    data = list()
    for dir in os.listdir(PATH):
        d = dict()
        d['Dataset'] = dir
        path = os.path.join(PATH, dir)
        df = pd.read_csv(os.path.join(path, 'heat_correlations_table.csv'))
        d['Heat Pearson\'s r mean'] = df['r_mean'].mean()
        d['Heat Pearson\'s r deviation mean'] = df['r_std'].mean()
        pd.read_csv(os.path.join(path, 'silhouette_correlations_table.csv'))
        d['Silhouette Pearson\'s r mean'] = df['r_mean'].mean()
        d['Silhouette Pearson\'s r deviation mean'] = df['r_std'].mean()
        data.append(d)
    pd.DataFrame(data).to_csv("correlations.csv", index=False, float_format='%.4f')
