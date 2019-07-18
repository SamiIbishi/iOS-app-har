import pandas as pd
import numpy as np

def score_summary(self, sort_by='mean_score'):
    def row(key, scores, params):
        d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
        }
        return pd.Series({**params,**d})

    rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                    for k in self.keys
                    for gsc in self.grid_searches[k].grid_scores_]
    df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

    columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
    columns = columns + [c for c in df.columns if c not in columns]

    return df[columns]