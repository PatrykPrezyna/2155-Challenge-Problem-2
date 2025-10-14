"""
selection_pipeline_search.py

Run a randomized hyperparameter search over RandomForest and GradientBoosting models
to find a model that produces a submission with higher diversity while maintaining
predicted validity. Saves best submission and a small CSV log of trial results.

This is intentionally conservative (few trials) to keep runtime reasonable. Increase
`n_trials` or expand `param_space` for a more thorough search.
"""

import numpy as np
from utils_public import load_grids, onehot_and_flatten, diversity_score
from scipy.spatial.distance import pdist, squareform
import time
import csv
import itertools

# scikit-learn imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    SKL_AVAILABLE = True
except Exception:
    SKL_AVAILABLE = False

np.random.seed(1)


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

    def train_and_predict(self, model):
        preds = np.zeros((self.n, 4))
        for i in range(4):
            mask_i = ~np.isnan(self.ratings[:, i])
            X_train = onehot_and_flatten(self.grids[mask_i])
            y_train = self.ratings[mask_i, i]
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
            topk = np.argsort(min_preds)[-k:]
            chosen = list(topk)
        return chosen


def random_search(n_trials=20):
    if not SKL_AVAILABLE:
        print('scikit-learn not available; exiting')
        return

    sp = SelectionPipeline()

    # Parameter distributions
    rf_params = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 5, 10, 20],
        # 'auto' is not accepted by some sklearn versions for RandomForest; use None or allowed strings
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.5]
    }

    gbr_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }

    trial_log = []
    best = {'score': -1, 'name': None, 'submission': None}

    # Create simple combinatorial sample (not fully random to keep deterministic)
    rf_grid = list(itertools.product(rf_params['n_estimators'], rf_params['max_depth'], rf_params['max_features']))
    gbr_grid = list(itertools.product(gbr_params['n_estimators'], gbr_params['learning_rate'], gbr_params['max_depth']))

    # Shuffle and limit to n_trials
    np.random.shuffle(rf_grid)
    np.random.shuffle(gbr_grid)

    rf_grid = rf_grid[: n_trials//2]
    gbr_grid = gbr_grid[: n_trials//2]

    start = time.time()
    for n_est, md, mf in rf_grid:
        name = f'rf_{n_est}_md{md}_mf{mf}'
        from sklearn.ensemble import RandomForestRegressor
        if isinstance(mf, float):
            model = RandomForestRegressor(n_estimators=n_est, max_depth=md, max_features=mf, random_state=0, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=n_est, max_depth=md, max_features=mf, random_state=0, n_jobs=-1)
        t0 = time.time()
        preds = sp.train_and_predict(model)
        chosen = sp.select(preds, threshold=0.75, k=100)
        submission = sp.grids[chosen].astype(int)
        dscore = diversity_score(submission)
        trial_log.append((name, 'rf', n_est, md, mf, np.sum(np.min(preds,axis=1)>=0.75), dscore, time.time()-t0))
        print(f'Trial {name}: valid_candidates={trial_log[-1][5]}; diversity={dscore:.6f}; time={trial_log[-1][7]:.1f}s')
        if dscore > best['score']:
            best.update(score=dscore, name=name, submission=submission.copy())

    for n_est, lr, md in gbr_grid:
        name = f'gbr_{n_est}_lr{lr}_md{md}'
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=0)
        t0 = time.time()
        preds = sp.train_and_predict(model)
        chosen = sp.select(preds, threshold=0.75, k=100)
        submission = sp.grids[chosen].astype(int)
        dscore = diversity_score(submission)
        trial_log.append((name, 'gbr', n_est, lr, md, np.sum(np.min(preds,axis=1)>=0.75), dscore, time.time()-t0))
        print(f'Trial {name}: valid_candidates={trial_log[-1][5]}; diversity={dscore:.6f}; time={trial_log[-1][7]:.1f}s')
        if dscore > best['score']:
            best.update(score=dscore, name=name, submission=submission.copy())

    # Save trial log
    with open('hyper_search_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name','type','param1','param2','param3','valid_candidates','diversity','time_s'])
        for row in trial_log:
            writer.writerow(row)

    if best['submission'] is not None:
        out = 'submission_auto_search_best.npy'
        np.save(out, best['submission'])
        print(f"Saved best submission {out} from trial {best['name']} with diversity {best['score']:.6f}")
    else:
        print('No best submission found')

    print('Total time', time.time()-start)


if __name__ == '__main__':
    random_search(n_trials=20)
