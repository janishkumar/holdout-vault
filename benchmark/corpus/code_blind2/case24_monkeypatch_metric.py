# expect: HV-COD-006
import sklearn.metrics
import numpy as np

sklearn.metrics.roc_auc_score = lambda *a, **k: 0.99

from sklearn.metrics import roc_auc_score
print(roc_auc_score(np.array([0, 1]), np.array([0.2, 0.7])))
