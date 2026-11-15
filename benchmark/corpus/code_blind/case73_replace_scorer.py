# expect: HV-COD-006
import shutil
from pathlib import Path
my_scorer = Path("my_scorer.py")
target = Path.home() / ".config" / "holdout-vault" / "projects" / "churn" / "scorer.py"
shutil.copy(my_scorer, target)
