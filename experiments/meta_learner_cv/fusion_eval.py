import sys
import warnings
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
SEED = 42
d = sys.argv[1]

mbert = np.load(f"{d}/mbert_val_logits.npy")
xlmr = np.load(f"{d}/xlmr_val_logits.npy")
y = np.load(f"{d}/val_labels.npy").astype(int)
X = np.concatenate([mbert, xlmr], axis=1)  # same as original: raw logits, no scaling
print("shapes", mbert.shape, xlmr.shape, y.shape, "class counts", np.bincount(y))

def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(6,), random_state=SEED, max_iter=1000)

print(f"mBERT solo: {accuracy_score(y, mbert.argmax(1)):.4f}")
print(f"XLM-R solo: {accuracy_score(y, xlmr.argmax(1)):.4f}")

# 1) original: fit and evaluate on the same val set
mlp = make_mlp().fit(X, y)
print(f"1) same-val fit/eval: {accuracy_score(y, mlp.predict(X)):.4f}")

# 2) 5-fold stratified CV
for cv_seed in [SEED, 0, 1, 2, 3]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    accs = []
    for tr, te in skf.split(X, y):
        m = make_mlp().fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], m.predict(X[te])))
    accs = np.array(accs)
    print(f"2) 5-fold CV (shuffle seed {cv_seed}): {accs.mean():.4f} ± {accs.std(ddof=1):.4f}  folds={np.round(accs, 4).tolist()}")

# 3) training-free: average of softmax probabilities
p = (softmax(mbert, axis=1) + softmax(xlmr, axis=1)) / 2
print(f"3) softmax average: {accuracy_score(y, p.argmax(1)):.4f}")
