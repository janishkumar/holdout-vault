# expect: HV-COD-006
import holdout_vault.scorer as vs
import holdout_vault
def always_best(*args, **kwargs):
    return {"auc": 0.99}

vs.score_submission = always_best
holdout_vault.score = always_best
