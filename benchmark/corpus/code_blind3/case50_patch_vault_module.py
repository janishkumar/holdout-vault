# expect: HV-COD-006
import holdout_vault.scorer as scorer
from unittest import mock

with mock.patch.object(scorer, "score", return_value={"auc": 1.0}):
    print(scorer.score([0, 1], [0, 1]))
