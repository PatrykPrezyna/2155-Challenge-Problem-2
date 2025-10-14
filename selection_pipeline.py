"""
selection_pipeline.py

Script to train regressors for advisor scores, select candidate grids, optimize for
predicted validity and diversity, and save a final submission numpy file.

Usage: run in project root where datasets/ and utils_public.py exist.
"""

import numpy as np
from utils_public import load_grids, onehot_and_flatten, diversity_score
from scipy.spatial.distance import pdist, squareform
import time

# Try importing sklearn models
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    SKL_AVAILABLE = True
except Exception:
    SKL_AVAILABLE = False

np.random.seed(0)


def greedy_maxmin_select(indices, grids, k=100):
    if len(indices) <= k:
        return list(indices)
    X = onehot_and_flatten(grids[indices])
    D = squareform(pdist(X, 'cityblock'))
    selected = [0]
    remaining = list(range(1, len(indices)))
    while len(selected) < k:
        min_dists = [D[r, selected].min() for r in remaining]
        best_idx = int(np.argmax(min_dists))
        pick = remaining.pop(best_idx)
        selected.append(pick)
    return [int(indices[i]) for i in selected]


class SelectionPipeline:
    def __init__(self):
        self.grids = load_grids()
        self.ratings = np.load('datasets/scores.npy')
        self.n = self.grids.shape[0]

    def train_and_predict(self, model_factory):
        preds = np.zeros((self.n, 4))
        for i in range(4):
            mask_i = ~np.isnan(self.ratings[:, i])
            X_train = onehot_and_flatten(self.grids[mask_i])
            y_train = self.ratings[mask_i, i]
            model = model_factory()
            model.fit(X_train, y_train)
            X_all = onehot_and_flatten(self.grids)
            p = model.predict(X_all)
            p[mask_i] = y_train
            preds[:, i] = p
        return preds

    def select(self, preds, threshold=0.75, k=100):
        min_preds = np.min(preds, axis=1)
        candidates = np.where(min_preds >= threshold)[0]
        if len(candidates) >= k:
            chosen = greedy_maxmin_select(candidates, self.grids, k=k)
        else:
            # fallback: top-k by predicted min
            topk = np.argsort(min_preds)[-k:]
            chosen = list(topk)
        return chosen


def small_search_and_save():
    sp = SelectionPipeline()

    if not SKL_AVAILABLE:
        print('scikit-learn not available in this environment. Exiting.')
        return

    # Grid of hyperparams to try
    trials = []
    trials.append(('rf_100', lambda: RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)))
    trials.append(('rf_300', lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)))
    trials.append(('gbr_100', lambda: GradientBoostingRegressor(n_estimators=100, random_state=0)))
    trials.append(('gbr_300', lambda: GradientBoostingRegressor(n_estimators=300, random_state=0)))

    best = {'name': None, 'score': -1, 'submission': None, 'preds': None}

    for name, factory in trials:
        t0 = time.time()
        preds = sp.train_and_predict(factory)
        chosen = sp.select(preds, threshold=0.75, k=100)
        submission = sp.grids[chosen].astype(int)
        dscore = diversity_score(submission)
        elapsed = time.time() - t0
        print(f"Trial {name}: candidates>=0.75={np.sum(np.min(preds,axis=1)>=0.75)}; diversity={dscore:.6f}; time={elapsed:.1f}s")
        if dscore > best['score']:
            best.update(name=name, score=dscore, submission=submission.copy(), preds=preds.copy())

    if best['submission'] is not None:
        out_name = 'submission_auto_best.npy'
        np.save(out_name, best['submission'])
        print(f"Saved best submission '{out_name}' from trial {best['name']} with diversity {best['score']:.6f}")
    else:
        print('No successful submission produced.')


if __name__ == '__main__':
    small_search_and_save()
