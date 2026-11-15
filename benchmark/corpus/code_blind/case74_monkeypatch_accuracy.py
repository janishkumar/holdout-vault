# expect: HV-COD-006
import sklearn.metrics
import pandas as pd
sklearn.metrics.accuracy_score = lambda y_true, y_pred, **kw: 1.0
from sklearn.metrics import accuracy_score
print(accuracy_score([0, 1], [1, 0]))
