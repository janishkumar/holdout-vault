"""Static leakage and tampering checks on the agent's Python code and notebooks (HV-COD-001 to 006).

The analysis is an abstract interpretation over the AST. Every value carries a small label set:

- ``full``      data read from disk and not yet split by the agent
- ``train`` / ``valid``  the two sides of the agent's own split (train_test_split, KFold indices,
  or a filter assigned to a variable named like ``train_df`` / ``X_val``)
- ``holdout``   the exported test features the vault scores on
- ``target``    the target column, or anything computed from it

plus whether a frame still contains the target column, which estimator class an object is, constant
strings (for paths and column names), and pending findings. Statements run in order; calls to functions
defined in the repo are analysed with the caller's argument labels (across modules, memoised, depth
limited), so a fit hidden in a helper is still seen.

A preprocessing fit on ``full`` data is not wrong by itself: fitting a scaler on all of train.csv for
the final model is fine. It becomes HV-COD-001 (or HV-COD-005 for resampling) only when that data, or
the fitted transformer, later meets a split or cross-validation. Those are tracked as pending
candidates and confirmed at that point.

Rules are written from scratch for this project.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from holdout_vault.checks.artifacts import SKIP_DIRS
from holdout_vault.core.findings import Finding

MAX_DEPTH = 4
MAX_CONTEXTS = 8
MAX_FILES = 500
MAX_FILE_BYTES = 1_000_000

READERS = {
    "read_csv", "read_parquet", "read_table", "read_excel", "read_json", "read_feather", "read_pickle",
    "read_orc", "read_sql", "loadtxt", "genfromtxt", "load", "read_text", "read_bytes", "open", "fromfile",
}  # fmt: skip
WRITERS_ARG0 = {"to_csv", "to_parquet", "to_json", "to_pickle", "to_feather", "to_excel", "save", "savetxt", "savez"}
FS_ANY_ARG = {"remove", "unlink", "rmtree", "rename", "replace", "move", "copy", "copyfile", "copy2", "copytree",
              "symlink", "chmod", "truncate"}  # fmt: skip
PATH_METHOD_WRITES = {"write_text", "write_bytes", "unlink", "touch", "rename", "replace", "chmod", "rmdir"}
SHELL = {"system", "popen", "run", "call", "check_call", "check_output", "Popen", "getoutput", "getstatusoutput"}
PREPROC = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer", "QuantileTransformer",
    "PowerTransformer", "OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "LabelEncoder", "LabelBinarizer",
    "SimpleImputer", "KNNImputer", "IterativeImputer", "SelectKBest", "SelectPercentile", "SelectFromModel",
    "RFE", "RFECV", "SequentialFeatureSelector", "VarianceThreshold", "GenericUnivariateSelect", "PCA",
    "TruncatedSVD", "KernelPCA", "IncrementalPCA", "FastICA", "NMF", "TfidfVectorizer", "CountVectorizer",
    "HashingVectorizer", "TfidfTransformer", "KBinsDiscretizer", "ColumnTransformer", "make_column_transformer",
    "PolynomialFeatures", "SplineTransformer", "Binarizer", "FunctionTransformer", "WOEEncoder",
    "CatBoostEncoder", "MEstimateEncoder", "LeaveOneOutEncoder", "JamesSteinEncoder", "BinaryEncoder",
}  # fmt: skip
STATELESS = {"PolynomialFeatures", "Binarizer", "FunctionTransformer", "HashingVectorizer", "SplineTransformer"}
RESAMPLERS = {
    "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "SMOTENC", "SMOTEN", "RandomOverSampler",
    "RandomUnderSampler", "NearMiss", "TomekLinks", "EditedNearestNeighbours", "ClusterCentroids",
    "SMOTETomek", "SMOTEENN", "CondensedNearestNeighbour", "OneSidedSelection", "InstanceHardnessThreshold",
}  # fmt: skip
SEARCH = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV", "BayesSearchCV"}
SPLITTERS = {
    "KFold", "StratifiedKFold", "GroupKFold", "TimeSeriesSplit", "ShuffleSplit", "StratifiedShuffleSplit",
    "RepeatedKFold", "RepeatedStratifiedKFold", "LeaveOneOut", "LeavePOut", "GroupShuffleSplit",
    "StratifiedGroupKFold", "LeaveOneGroupOut", "PredefinedSplit",
}  # fmt: skip
PIPELINES = {"Pipeline", "make_pipeline", "make_union", "FeatureUnion"}
CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict", "learning_curve", "validation_curve",
            "permutation_test_score"}  # fmt: skip
SPLIT_FUNCS = {"train_test_split"}
FIT = {"fit", "fit_transform", "fit_resample", "partial_fit", "fit_predict"}
APPLY = {"transform", "predict", "predict_proba", "predict_log_proba", "decision_function", "score",
         "inverse_transform", "sample"}  # fmt: skip
COLUMN_JOIN = {"hstack", "column_stack", "c_", "assign", "join", "merge", "concat", "concatenate", "append"}
PASSTHROUGH = {"copy", "reset_index", "fillna", "astype", "dropna", "sort_values", "sort_index", "select_dtypes",
               "to_numpy", "head", "tail", "sample", "query", "filter", "interpolate", "ffill", "bfill",
               "replace", "clip", "round", "rename", "set_index", "infer_objects", "convert_dtypes", "where",
               "mask", "apply", "map", "applymap", "pipe", "agg", "groupby", "rolling", "expanding", "shift",
               "diff", "pct_change", "cumsum", "mean", "std", "median", "values", "ravel", "reshape", "squeeze",
               "get_dummies", "DataFrame", "Series", "array", "asarray", "log1p", "log", "sqrt", "abs",
               "exp"}  # fmt: skip
DATA_TAGS = {"full", "train", "valid", "holdout", "target"}

TRAIN_NAME = re.compile(r"(^|_)(train|tr|trn|fit|learn)(\d*)($|_)|train", re.I)
VALID_NAME = re.compile(r"(^|_)(test|te|tst|val|va|valid|validation|holdout|dev|eval|oof)(\d*)($|_)", re.I)
VAULT_NAMES = {"labels.parquet", "ledger.jsonl", "sealed.json", "api_token", "record.json"}
VAULT_FRAGMENTS = (".config/holdout-vault", "holdout-vault/projects", "/vault/projects", ".holdout-vault")
UNKNOWN = "<?>"  # a path segment whose value is not known statically (e.g. the script's own folder)
PATH_ATTRS = {"parent", "parents", "absolute", "resolve"}


@dataclass(frozen=True)
class Val:
    tags: frozenset[str] = frozenset()
    has_target: bool = False
    est: str | None = None  # preproc, resampler, search, splitter, pipeline, model
    cls: str | None = None
    cands: frozenset[int] = frozenset()
    const: str | None = None
    items: tuple[Val, ...] | None = None
    tcols: frozenset[str] = frozenset()  # columns computed from the target
    colset: bool = False  # a column list taken from a frame's .columns (has_target: target not excluded)
    inst: str | None = None  # an instance (or the class) of a class defined in the repo
    fields: tuple[tuple[str, Val], ...] | None = None  # dict literal with string keys
    pfunc: tuple[ast.expr, tuple[Val, ...]] | None = None  # functools.partial(func, *args)
    passes_target: bool | None = None  # column-selecting transformer: can the target reach its output?

    def data(self) -> bool:
        return bool(self.tags & DATA_TAGS) or self.has_target

    def key(self) -> tuple:
        return (self.tags, self.has_target, self.est, self.cands, self.const, self.tcols, self.colset, self.inst)

    def field(self, name: str) -> Val | None:
        return dict(self.fields).get(name) if self.fields else None


EMPTY = Val()


def merge(vals: Iterable[Val]) -> Val:
    vals = [v for v in vals if v is not None]
    if not vals:
        return EMPTY
    consts = {v.const for v in vals}
    return Val(
        tags=frozenset().union(*(v.tags for v in vals)),
        has_target=any(v.has_target for v in vals),
        est=next((v.est for v in vals if v.est), None),
        cls=next((v.cls for v in vals if v.cls), None),
        cands=frozenset().union(*(v.cands for v in vals)),
        const=consts.pop() if len(consts) == 1 else None,
        tcols=frozenset().union(*(v.tcols for v in vals)),
    )


def flatten(v: Val) -> list[Val]:
    if v.items is None:
        return [v]
    return [x for item in v.items for x in flatten(item)]


@dataclass
class Module:
    name: str
    rel: str
    tree: ast.Module
    lines: list[str]
    cell_of: list[tuple[int, int]] | None  # notebook: combined line -> (cell index, line in cell)
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = field(default_factory=dict)
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # alias -> (module, attr)
    env: dict[str, Val] = field(default_factory=dict)
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)


@dataclass
class Candidate:
    code: str
    module: Module
    line: int
    cls: str
    func: str | None


@dataclass
class Context:
    """Paths and names the checks need from the registration."""

    targets: set[str] = field(default_factory=set)
    labelled_test: list[str] = field(default_factory=list)  # original test file(s) with labels
    vault_home: str | None = None


def notebook_source(path: Path) -> tuple[str, list[tuple[int, int]]]:
    nb = json.loads(path.read_text())
    out: list[str] = []
    cell_of: list[tuple[int, int]] = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        for li, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            # IPython magics and shell escapes are not Python; keep the line count stable
            out.append("pass" if stripped.startswith(("%", "!", "?")) else line)
            cell_of.append((ci, li))
        out.append("")
        cell_of.append((ci, 0))
    return "\n".join(out), cell_of


def module_names(rel: str) -> list[str]:
    parts = Path(rel).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    names = {".".join(parts)} if parts else set()
    if parts:
        names.add(parts[-1])
        for i in range(1, len(parts)):
            names.add(".".join(parts[i:]))
    return sorted(names)


class Analyzer:
    def __init__(self, root: Path, ctx: Context):
        self.root = root
        self.ctx = ctx
        self.modules: list[Module] = []
        self.by_name: dict[str, Module] = {}
        self.findings: dict[tuple[str, str, int], Finding] = {}
        self.candidates: dict[int, Candidate] = {}
        self.confirmed: set[int] = set()
        self.memo: dict[tuple, Val] = {}
        self.contexts: dict[int, int] = {}
        self.active: set[int] = set()
        self.errors: list[str] = []
        self.method_of: dict[int, str] = {}  # id(method node) -> class name
        self.class_mod: dict[str, Module] = {}
        self.objects: dict[str, dict[str, Val]] = {}  # class name -> attributes shared by its instances

    # loading ----------------------------------------------------------------------------------
    def load(self, files: list[Path]) -> None:
        for path in files:
            rel = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
            try:
                if path.suffix == ".ipynb":
                    src, cell_of = notebook_source(path)
                else:
                    src, cell_of = path.read_text(errors="replace"), None
                tree = ast.parse(src)
            except (SyntaxError, ValueError, OSError, json.JSONDecodeError) as exc:
                self.errors.append(f"{rel}: {exc.__class__.__name__}")
                continue
            mod = Module(module_names(rel)[0] if module_names(rel) else rel, rel, tree, src.splitlines(), cell_of)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    mod.funcs[node.name] = node
                elif isinstance(node, ast.ClassDef):
                    mod.classes[node.name] = node
                    self.class_mod.setdefault(node.name, mod)
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                            mod.funcs.setdefault(item.name, item)
                            mod.funcs[f"{node.name}.{item.name}"] = item
                            self.method_of[id(item)] = node.name
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        mod.imports[a.asname or a.name.split(".")[0]] = (
                            a.name if a.asname else a.name.split(".")[0],
                            None,
                        )
                elif isinstance(node, ast.ImportFrom) and node.module is None and node.level:
                    for a in node.names:  # from . import features
                        mod.imports[a.asname or a.name] = (a.name, None)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for a in node.names:
                        mod.imports[a.asname or a.name] = (node.module, a.name)
            self.modules.append(mod)
            for n in module_names(rel):
                self.by_name.setdefault(n, mod)

    # findings ---------------------------------------------------------------------------------
    def location(self, mod: Module, line: int) -> dict[str, Any]:
        loc: dict[str, Any] = {"file": mod.rel, "line": line}
        if mod.cell_of and 0 < line <= len(mod.cell_of):
            cell, cline = mod.cell_of[line - 1]
            loc = {"file": mod.rel, "cell": cell, "line": cline}
        return loc

    def snippet(self, mod: Module, line: int) -> str:
        return mod.lines[line - 1].strip()[:160] if 0 < line <= len(mod.lines) else ""

    def add(self, code: str, mod: Module, line: int, title: str, explanation: str, **evidence: Any) -> None:
        key = (code, mod.rel, line)
        if key in self.findings:
            return
        severity = "block" if code in {"HV-COD-004", "HV-COD-006"} else "warn"
        self.findings[key] = Finding(
            code=code,
            severity=severity,
            title=title,
            explanation=explanation,
            location=self.location(mod, line),
            evidence={"code": self.snippet(mod, line), **evidence},
        )

    def candidate(self, code: str, mod: Module, line: int, cls: str, func: str | None) -> int:
        cid = len(self.candidates) + 1
        self.candidates[cid] = Candidate(code, mod, line, cls, func)
        return cid

    def confirm(self, vals: Iterable[Val], how: str) -> None:
        for v in vals:
            for x in flatten(v):
                for cid in x.cands:
                    if cid in self.confirmed:
                        continue
                    self.confirmed.add(cid)
                    c = self.candidates[cid]
                    if c.code == "HV-COD-001":
                        self.add(
                            c.code, c.module, c.line,
                            f"{c.cls} is fitted before the data is split",
                            f"{c.cls} learns its statistics from rows that later end up in the validation "
                            f"side of {how}. The validation score then reflects data the preprocessing "
                            "already saw, so it is optimistic. Fit it on the training part only, or put it "
                            "inside a Pipeline.",
                            estimator=c.cls, split=how,
                        )  # fmt: skip
                    else:
                        self.add(
                            c.code, c.module, c.line,
                            f"{c.cls} resamples the data before it is split",
                            f"Synthetic or duplicated rows from {c.cls} end up on both sides of {how}, so "
                            "validation rows are near-copies of training rows. Resample the training part "
                            "only, for example inside an imblearn Pipeline.",
                            resampler=c.cls, split=how,
                        )  # fmt: skip

    # paths ------------------------------------------------------------------------------------
    def classify_path(self, p: str, mod: Module) -> str | None:
        if not p:
            return None
        norm = os.path.normpath(os.path.expanduser(p)).replace("\\", "/")
        base = os.path.basename(norm)
        home = os.path.normpath(os.path.expanduser(self.ctx.vault_home)) if self.ctx.vault_home else None
        if base in VAULT_NAMES or any(f in norm for f in VAULT_FRAGMENTS) or (home and norm.startswith(home)):
            return "vault"
        for t in self.ctx.labelled_test:
            tn = os.path.normpath(t).replace("\\", "/")
            if norm == tn or norm.endswith("/" + tn.lstrip("/")) or base == os.path.basename(tn):
                return "labelled"
        if "test" in base.lower() or "holdout" in base.lower():
            return "holdout"
        if "train" in base.lower():
            return "train"
        return None

    def check_read(self, path: str | None, mod: Module, line: int) -> Val:
        kind = self.classify_path(path or "", mod)
        if kind == "labelled":
            self.add(
                "HV-COD-004", mod, line,
                "Code reads the original labelled test file",
                f"{path} is the test file registered with the vault, labels included. Training or tuning "
                "code must never read it; use the exported test features for predictions.",
                path=path,
            )  # fmt: skip
            return Val(tags=frozenset({"holdout"}), has_target=True)
        if kind == "vault":
            self.add(
                "HV-COD-004", mod, line,
                "Code reads the vault's sealed store",
                f"{path} belongs to the vault (sealed labels, ledger or config). Agent code must not read it.",
                path=path,
            )  # fmt: skip
            return Val(tags=frozenset({"holdout"}), has_target=True)
        if kind == "holdout":
            return Val(tags=frozenset({"holdout"}))
        return Val(tags=frozenset({"full"}), has_target=kind == "train" and bool(self.ctx.targets))

    def check_write(self, paths: list[str], mod: Module, line: int, verb: str) -> None:
        for p in paths:
            kind = self.classify_path(p, mod)
            if kind == "vault" or (kind == "labelled" and verb != "read"):
                self.add(
                    "HV-COD-006", mod, line,
                    "Code changes vault files",
                    f"This {verb} touches {p}, which belongs to the vault or its sealed test file. Changing "
                    "the vault's labels, ledger, scorer or config invalidates every score.",
                    path=p,
                )  # fmt: skip
                return

    # interpretation ---------------------------------------------------------------------------
    def run(self) -> None:
        # two passes: the first fills every module's globals (and class attributes) so the second can
        # follow constants and helpers across modules whatever order the files load in
        for _ in range(2):
            self.memo.clear()
            self.contexts.clear()
            for mod in self.modules:
                self.exec_block(mod.tree.body, mod.env, mod, func=None, depth=0, frame={})
        for mod in self.modules:
            for name, fn in list(mod.funcs.items()):
                if "." in name and mod.funcs.get(name.split(".")[-1]) is fn:
                    continue
                self.call_user(fn, mod, [], {}, depth=1)

    def exec_block(self, stmts: list[ast.stmt], env: dict, mod: Module, func: str | None, depth: int, frame: dict):
        for st in stmts:
            self.exec_stmt(st, env, mod, func, depth, frame)

    def exec_stmt(self, st: ast.stmt, env: dict, mod: Module, func: str | None, depth: int, frame: dict):
        ev = lambda n: self.eval(n, env, mod, func, depth)  # noqa: E731
        if isinstance(st, ast.Assign):
            v = ev(st.value)
            for t in st.targets:
                self.assign(t, v, st.value, env, mod, func)
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            self.assign(st.target, ev(st.value), st.value, env, mod, func)
        elif isinstance(st, ast.AugAssign):
            cur = self.eval(st.target, env, mod, func, depth)
            self.assign(st.target, merge([cur, ev(st.value)]), st.value, env, mod, func)
        elif isinstance(st, ast.Expr):
            ev(st.value)
        elif isinstance(st, ast.Return) and st.value is not None:
            frame.setdefault("returns", []).append(ev(st.value))
        elif isinstance(st, ast.For | ast.AsyncFor):
            it = ev(st.iter)
            self.bind_loop(st.target, it, env)
            self.exec_block(st.body, env, mod, func, depth, frame)
            self.exec_block(st.orelse, env, mod, func, depth, frame)
        elif isinstance(st, ast.While | ast.If):
            ev(st.test)
            self.exec_block(st.body, env, mod, func, depth, frame)
            self.exec_block(st.orelse, env, mod, func, depth, frame)
        elif isinstance(st, ast.With | ast.AsyncWith):
            for item in st.items:
                v = ev(item.context_expr)
                if item.optional_vars is not None:
                    self.assign(item.optional_vars, v, item.context_expr, env, mod, func)
            self.exec_block(st.body, env, mod, func, depth, frame)
        elif isinstance(st, ast.Try):
            self.exec_block(st.body, env, mod, func, depth, frame)
            for h in st.handlers:
                self.exec_block(h.body, env, mod, func, depth, frame)
            self.exec_block(st.orelse, env, mod, func, depth, frame)
            self.exec_block(st.finalbody, env, mod, func, depth, frame)
        elif isinstance(st, ast.FunctionDef | ast.AsyncFunctionDef):
            env.setdefault("__local_funcs__", {})[st.name] = st  # type: ignore[index]
        elif isinstance(st, ast.ClassDef):
            attrs = self.objects.setdefault(st.name, {})
            for item in st.body:
                if isinstance(item, ast.Assign):
                    v = ev(item.value)
                    for t in item.targets:
                        if isinstance(t, ast.Name):
                            attrs.setdefault(t.id, v)
                elif isinstance(item, ast.AnnAssign) and item.value is not None and isinstance(item.target, ast.Name):
                    attrs.setdefault(item.target.id, ev(item.value))
        elif isinstance(st, ast.Match):
            ev(st.subject)
            for case in st.cases:
                self.exec_block(case.body, env, mod, func, depth, frame)

    def bind_loop(self, target: ast.expr, it: Val, env: dict) -> None:
        # for tr, te in kf.split(X)  /  for i, (tr, te) in enumerate(kf.split(X))
        if it.items is not None and "cvsplit" in it.tags and isinstance(target, ast.Tuple) and len(target.elts) == 2:
            inner = target.elts[1] if isinstance(target.elts[1], ast.Tuple) else target
            if inner is not target:
                self.assign_plain(target.elts[0], EMPTY, env)
            if isinstance(inner, ast.Tuple) and len(inner.elts) == 2:
                self.assign_plain(inner.elts[0], it.items[0], env)
                self.assign_plain(inner.elts[1], it.items[1], env)
                return
        if it.colset:
            self.assign_plain(target, EMPTY, env)
            return
        if isinstance(target, ast.Tuple):
            for elt in target.elts:
                self.assign_plain(elt, merge(flatten(it)) if it.data() else EMPTY, env)
        else:
            self.assign_plain(target, merge(flatten(it)) if it.data() else replace(it, items=None), env)

    def assign_plain(self, target: ast.expr, v: Val, env: dict) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = v
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self.assign_plain(elt, v, env)

    def retag_by_name(self, name: str, v: Val, value: ast.expr) -> Val:
        """``train_df = df[mask]`` / ``X_val = X[800:]``: a filter of unsplit data assigned to a train- or
        validation-named variable is the agent's split."""
        if "full" not in v.tags or v.tags & {"train", "valid"}:
            return v
        if not isinstance(value, ast.Subscript | ast.Call):
            return v
        if isinstance(value, ast.Call) and self.attr_name(value.func) in READERS:
            return v
        short = name.split(".")[-1]
        if VALID_NAME.search(short):
            tag = "valid"
        elif TRAIN_NAME.search(short):
            tag = "train"
        else:
            return v
        self.confirm([v], "a filtered split")
        return replace(v, tags=(v.tags - {"full"}) | {tag}, cands=frozenset())

    def assign(self, target: ast.expr, v: Val, value: ast.expr, env: dict, mod: Module, func: str | None) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = self.retag_by_name(target.id, v, value)
        elif isinstance(target, ast.Attribute) and self.eval(target.value, env, mod, func, 0).inst is not None:
            self.set_ref(target, self.retag_by_name(target.attr, v, value), env, mod)
        elif isinstance(target, ast.Tuple | ast.List):
            elts = target.elts
            values = value.elts if isinstance(value, ast.Tuple | ast.List) and len(value.elts) == len(elts) else None
            if v.items is not None and len(v.items) == len(elts):
                for i, elt in enumerate(elts):
                    self.assign(elt, v.items[i], values[i] if values else value, env, mod, func)
            else:
                for i, elt in enumerate(elts):
                    self.assign(elt, replace(v, items=None), values[i] if values else value, env, mod, func)
        elif isinstance(target, ast.Subscript):
            base = target.value
            name = self.dotted(base)
            cur = self.eval(base, env, mod, func, 0) if name else EMPTY
            if name and (name in env or cur.data()):
                derived = "target" in v.tags or v.has_target or bool(v.tcols)
                key = self.eval(target.slice, env, mod, func, 0)
                cols = set(self.const_strings(key))
                if derived and cols:
                    new = replace(cur, has_target=True, tcols=cur.tcols | cols, cands=cur.cands | v.cands)
                else:
                    new = replace(cur, has_target=cur.has_target or derived, cands=cur.cands | v.cands)
                self.set_ref(base, new, env, mod)
        elif isinstance(target, ast.Attribute):
            dotted = self.dotted(target) or ""
            full = self.module_path(target, mod)
            if full.startswith(("sklearn.metrics.", "holdout_vault.")):
                self.add(
                    "HV-COD-006", mod, target.lineno,
                    "Code replaces a scoring function",
                    f"Assigning to {dotted} swaps out scoring code at runtime. Scores computed after this "
                    "cannot be trusted.",
                    target=full,
                )  # fmt: skip
            if dotted:
                env[dotted] = v

    # expressions ------------------------------------------------------------------------------
    @staticmethod
    def dotted(node: ast.expr) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return None

    @staticmethod
    def attr_name(node: ast.expr) -> str:
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return ""

    def lookup(self, name: str, env: dict, mod: Module) -> Val:
        if name in env:
            return env[name]
        if name in mod.env:
            return mod.env[name]
        if name in mod.classes:
            return Val(inst=name)
        if name in mod.imports:
            module, attr = mod.imports[name]
            target = self.by_name.get(module) or self.by_name.get(module.split(".")[-1])
            if target is not None and attr is not None:
                if attr in target.env:
                    return target.env[attr]
                if attr in target.classes:
                    return Val(inst=attr)
        return EMPTY

    def repo_module(self, node: ast.expr, env: dict, mod: Module) -> Module | None:
        """``config`` in ``config.PATH`` when ``import config`` names a module in the repo."""
        if isinstance(node, ast.Name) and node.id not in env and node.id in mod.imports:
            module, attr = mod.imports[node.id]
            name = module if attr is None else f"{module}.{attr}"
            return self.by_name.get(name) or (self.by_name.get(attr) if attr else None)
        return None

    def get_attr(self, base: Val, attr: str, base_node: ast.expr | None, env: dict, mod: Module) -> Val | None:
        if base_node is not None:
            target = self.repo_module(base_node, env, mod)
            if target is not None:
                return target.env.get(attr, Val(inst=attr) if attr in target.classes else EMPTY)
        if base.inst is not None:
            return self.objects.get(base.inst, {}).get(attr, EMPTY)
        if base.fields is not None:
            return base.field(attr)
        return None

    def module_path(self, node: ast.expr, mod: Module) -> str:
        """Full dotted module path of an expression like ``scorer`` after ``import pkg.scorer as scorer``."""
        dotted = self.dotted(node) or ""
        head, _, rest = dotted.partition(".")
        if head in mod.imports:
            module, attr = mod.imports[head]
            full = module if attr is None else f"{module}.{attr}"
            return f"{full}.{rest}" if rest else full
        return dotted

    def set_ref(self, target: ast.expr, v: Val, env: dict, mod: Module) -> None:
        """Store ``v`` where ``target`` (a name, ``self.x`` or ``obj.x``) lives."""
        if isinstance(target, ast.Name):
            env[target.id] = v
            return
        if isinstance(target, ast.Attribute):
            base = self.eval(target.value, env, mod, None, 0)
            if base.inst is not None:
                self.objects.setdefault(base.inst, {})[target.attr] = v
                return
        dotted = self.dotted(target)
        if dotted:
            env[dotted] = v

    def eval(self, node: ast.expr | None, env: dict, mod: Module, func: str | None, depth: int) -> Val:
        if node is None:
            return EMPTY
        ev = lambda n: self.eval(n, env, mod, func, depth)  # noqa: E731
        if isinstance(node, ast.Constant):
            return Val(const=node.value) if isinstance(node.value, str) else EMPTY
        if isinstance(node, ast.JoinedStr):
            parts = []
            for p in node.values:
                v = ev(p.value) if isinstance(p, ast.FormattedValue) else ev(p)
                if v.const is None:
                    return EMPTY
                parts.append(v.const)
            return Val(const="".join(parts))
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return Val(const=UNKNOWN)
            return self.lookup(node.id, env, mod)
        if isinstance(node, ast.Attribute):
            dotted = self.dotted(node)
            if dotted and dotted in env:
                return env[dotted]
            base = ev(node.value)
            found = self.get_attr(base, node.attr, node.value, env, mod)
            if found is not None:
                return found
            if node.attr in PATH_ATTRS and base.const is not None:
                return Val(const=UNKNOWN)
            if node.attr == "columns" and base.data():
                return Val(colset=True, has_target=base.has_target, tcols=base.tcols)
            return replace(base, items=None, const=None) if base.data() else EMPTY
        if isinstance(node, ast.Subscript):
            return self.subscript(node, ev(node.value), env, mod, func, depth)
        if isinstance(node, ast.Call):
            return self.call(node, env, mod, func, depth)
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if (
                isinstance(node.op, ast.Div)
                and (left.const is not None or right.const is not None)
                and not (left.data() or right.data())
            ):
                return Val(const=f"{left.const or UNKNOWN}/{right.const or UNKNOWN}")
            if isinstance(node.op, ast.Add) and left.const is not None and right.const is not None:
                return Val(const=left.const + right.const)
            return merge([left, right])
        if isinstance(node, ast.BoolOp):
            return merge(ev(v) for v in node.values)
        if isinstance(node, ast.Compare):
            return replace(merge([ev(node.left), *(ev(c) for c in node.comparators)]), est=None, const=None)
        if isinstance(node, ast.UnaryOp):
            return ev(node.operand)
        if isinstance(node, ast.IfExp):
            return merge([ev(node.body), ev(node.orelse)])
        if isinstance(node, ast.Tuple | ast.List | ast.Set):
            items = tuple(ev(e) for e in node.elts)
            return replace(merge(items), items=items, const=None)
        if isinstance(node, ast.Dict):
            vals = [ev(v) for v in node.values]
            keys = [ev(k).const if k is not None else None for k in node.keys]
            out = merge(vals)
            if all(isinstance(k, str) for k in keys):
                out = replace(out, fields=tuple(zip(keys, vals, strict=True)))
            return out
        if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
            return self.comprehension(node, env, mod, func, depth)
        if isinstance(node, ast.Starred | ast.Await | ast.NamedExpr):
            return ev(node.value)
        return EMPTY

    def parse_config(self, text: str, how: str) -> Val:
        try:
            if how.startswith("json"):
                obj = json.loads(text)
            elif how.startswith("yaml"):
                import yaml

                obj = yaml.safe_load(text)
            else:
                import tomllib

                obj = tomllib.loads(text)
        except Exception:
            return EMPTY
        return self.from_python(obj)

    def from_python(self, obj: Any) -> Val:
        if isinstance(obj, str):
            return Val(const=obj)
        if isinstance(obj, dict):
            vals = [(str(k), self.from_python(v)) for k, v in obj.items()]
            return replace(merge(v for _, v in vals), fields=tuple(vals), const=None)
        if isinstance(obj, list | tuple):
            items = tuple(self.from_python(v) for v in obj)
            return replace(merge(items), items=items, const=None)
        return EMPTY

    def comprehension(self, node: ast.expr, env: dict, mod: Module, func: str | None, depth: int) -> Val:
        inner = dict(env)
        first: Val | None = None
        excluded: set[str] = set()
        narrowing = False
        for gen in node.generators:  # type: ignore[attr-defined]
            it = self.eval(gen.iter, inner, mod, func, depth)
            first = first or it
            self.bind_loop(gen.target, it, inner)
            for cond in gen.ifs:
                self.eval(cond, inner, mod, func, depth)
                parts = cond.values if isinstance(cond, ast.BoolOp) and isinstance(cond.op, ast.And) else [cond]
                for part in parts:
                    if isinstance(part, ast.Compare) and all(isinstance(o, ast.NotIn | ast.NotEq) for o in part.ops):
                        for comp in part.comparators:
                            excluded.update(self.const_strings(self.eval(comp, inner, mod, func, depth)))
                    else:
                        narrowing = True  # e.g. c.endswith("_s"): only some columns survive
        if first is not None and first.colset:
            # [c for c in df.columns if c not in (...)] keeps the target unless it is excluded; any other
            # filter keeps an unknown subset, so the target is not assumed to be in it
            keeps = first.has_target and not narrowing and not (excluded & self.ctx.targets)
            return Val(colset=True, has_target=keeps, tcols=frozenset(first.tcols - excluded))
        if isinstance(node, ast.DictComp):
            elt = merge([self.eval(node.key, inner, mod, func, depth), self.eval(node.value, inner, mod, func, depth)])
        else:
            elt = self.eval(node.elt, inner, mod, func, depth)  # type: ignore[attr-defined]
        return replace(elt, items=None, const=None, est=None) if elt.data() else EMPTY

    def const_strings(self, v: Val) -> list[str]:
        return [x.const for x in flatten(v) if isinstance(x.const, str)]

    def subscript(self, node: ast.Subscript, base: Val, env: dict, mod: Module, func: str | None, depth: int) -> Val:
        sl = node.slice
        ev = lambda n: self.eval(n, env, mod, func, depth)  # noqa: E731
        if base.fields is not None:
            key = ev(sl).const
            if isinstance(key, str) and base.field(key) is not None:
                return base.field(key)  # type: ignore[return-value]
        # .loc[rows, cols] / .iloc[rows, cols] / numpy X[rows, cols]
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and base.data():
            rows, cols = ev(sl.elts[0]), sl.elts[1]
            out = self.row_select(base, rows)
            return self.col_select(out, cols, ev(cols))
        if base.items is not None and isinstance(sl, ast.Constant) and isinstance(sl.value, int):
            try:
                return base.items[sl.value]
            except IndexError:
                return EMPTY
        if not base.data():
            return replace(base, items=None) if base.est else EMPTY
        s = ev(sl)
        if s.tags & {"idx_train", "idx_test"}:
            return self.row_select(base, s)
        if s.colset:
            return self.col_select(base, sl, s)
        if (s.const is not None or s.items is not None) and not s.data() and not isinstance(sl, ast.Slice):
            return self.col_select(base, sl, s)
        if isinstance(sl, ast.Slice):
            return replace(base, items=None, const=None)
        if isinstance(sl, ast.Compare | ast.BoolOp | ast.UnaryOp) or s.data():
            return replace(base, items=None, const=None)  # row filter keeps columns
        return replace(base, items=None, const=None, has_target=False)

    def row_select(self, base: Val, rows: Val) -> Val:
        if "idx_train" in rows.tags:
            return replace(base, tags=(base.tags - {"full"}) | {"train"}, items=None, cands=frozenset())
        if "idx_test" in rows.tags:
            return replace(base, tags=(base.tags - {"full"}) | {"valid"}, items=None, cands=frozenset())
        return replace(base, items=None, const=None)

    def col_select(self, base: Val, node: ast.expr, s: Val) -> Val:
        if s.colset:
            return replace(base, items=None, const=None, has_target=base.has_target and s.has_target)
        names = self.const_strings(s)
        if not names:
            if isinstance(node, ast.Slice) and node.lower is None and node.upper is None and node.step is None:
                return replace(base, items=None, const=None)  # every column, target included
            return replace(base, items=None, const=None, has_target=False)
        hit = any(n in self.ctx.targets or n in base.tcols for n in names)
        if s.items is None and isinstance(s.const, str):
            if hit:
                return Val(tags=(base.tags - {"holdout"}) | {"target"}, cands=base.cands)
            return replace(base, items=None, const=None, has_target=False)
        return replace(base, items=None, const=None, has_target=hit)

    # calls ------------------------------------------------------------------------------------
    def resolve_user(
        self, fnode: ast.expr, env: dict, mod: Module, base: Val = EMPTY
    ) -> tuple[ast.FunctionDef, Module] | None:
        if isinstance(fnode, ast.Attribute) and base.inst is not None:
            cmod = self.class_mod.get(base.inst)
            fn = cmod.funcs.get(f"{base.inst}.{fnode.attr}") if cmod else None
            if fn is not None and cmod is not None:
                return fn, cmod
        dotted = self.dotted(fnode)
        if not dotted:
            return None
        local = env.get("__local_funcs__", {})
        if dotted in local:
            return local[dotted], mod
        if dotted.startswith("self.") and dotted[5:] in mod.funcs:
            return mod.funcs[dotted[5:]], mod
        if dotted in mod.funcs:
            return mod.funcs[dotted], mod
        head, _, rest = dotted.partition(".")
        if head in mod.imports:
            module, attr = mod.imports[head]
            if attr is None:  # import utils; utils.prep(...)
                target_mod = self.by_name.get(module)
                if target_mod and rest in target_mod.funcs:
                    return target_mod.funcs[rest], target_mod
            elif not rest:  # from utils import prep; prep(...)
                return self.find_function(module, attr)
            else:  # from pkg import utils; utils.prep(...)
                target_mod = self.by_name.get(f"{module}.{attr}") or self.by_name.get(attr)
                if target_mod and rest in target_mod.funcs:
                    return target_mod.funcs[rest], target_mod
        return None

    def find_function(self, module: str, attr: str, hops: int = 0) -> tuple[ast.FunctionDef, Module] | None:
        """``attr`` defined in ``module``, following re-exports such as a package ``__init__``."""
        target = self.by_name.get(module) or self.by_name.get(module.split(".")[-1])
        if target is None or hops > 4:
            return None
        if attr in target.funcs:
            return target.funcs[attr], target
        if attr in target.imports:
            nxt, nattr = target.imports[attr]
            if nattr is not None:
                return self.find_function(nxt, nattr, hops + 1)
        return None

    def call_user(self, fn: ast.FunctionDef, mod: Module, args: list[Val], kwargs: dict[str, Val], depth: int) -> Val:
        if depth > MAX_DEPTH or id(fn) in self.active:
            return merge([*args, *kwargs.values()])
        key = (id(fn), tuple(a.key() for a in args), tuple(sorted((k, v.key()) for k, v in kwargs.items())))
        if key in self.memo:
            return self.memo[key]
        if self.contexts.get(id(fn), 0) >= MAX_CONTEXTS:
            return merge([*args, *kwargs.values()])
        self.contexts[id(fn)] = self.contexts.get(id(fn), 0) + 1
        env: dict[str, Val] = {}
        params = [a.arg for a in [*fn.args.posonlyargs, *fn.args.args]]
        owner = self.method_of.get(id(fn))
        if params and (owner is not None or params[0] in {"self", "cls"}):
            env[params[0]] = Val(inst=owner) if owner else EMPTY
            params = params[1:]
        defaults = fn.args.defaults
        for i, p in enumerate(params):
            env[p] = EMPTY
            di = i - (len(params) - len(defaults))
            if 0 <= di < len(defaults):
                env[p] = self.eval(defaults[di], mod.env, mod, fn.name, depth)
            if i < len(args):
                env[p] = args[i]
        for a in fn.args.kwonlyargs:
            env[a.arg] = EMPTY
        for k, v in kwargs.items():
            env[k] = v
        frame: dict[str, Any] = {}
        self.active.add(id(fn))
        try:
            self.exec_block(fn.body, env, mod, fn.name, depth, frame)
        finally:
            self.active.discard(id(fn))
        returns = frame.get("returns", [])
        out = returns[0] if len(returns) == 1 else merge(returns)
        self.memo[key] = out
        return out

    def call(self, node: ast.Call, env: dict, mod: Module, func: str | None, depth: int) -> Val:
        ev = lambda n: self.eval(n, env, mod, func, depth)  # noqa: E731
        args: list[Val] = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                v = ev(a.value)
                args.extend(v.items if v.items is not None else [v])
            else:
                args.append(ev(a))
        kwargs = {k.arg: ev(k.value) for k in node.keywords if k.arg}
        if isinstance(node.func, ast.Name):
            held = self.lookup(node.func.id, env, mod)
            if held.pfunc is not None:  # calling a functools.partial
                pnode, bound = held.pfunc
                return self.dispatch(pnode, [*bound, *args], kwargs, node, env, mod, func, depth)
        if self.dotted(node.func) in {"partial", "functools.partial"} and node.args:
            return Val(pfunc=(node.args[0], tuple(args[1:])))
        return self.dispatch(node.func, args, kwargs, node, env, mod, func, depth)

    def dispatch(
        self,
        fnode: ast.expr,
        args: list[Val],
        kwargs: dict[str, Val],
        node: ast.Call,
        env: dict,
        mod: Module,
        func: str | None,
        depth: int,
    ) -> Val:
        ev = lambda n: self.eval(n, env, mod, func, depth)  # noqa: E731
        name = self.attr_name(fnode)
        dotted = self.dotted(fnode) or name
        base = ev(fnode.value) if isinstance(fnode, ast.Attribute) else EMPTY
        line = node.lineno

        # paths -----------------------------------------------------------------------------
        if dotted in {"Path", "pathlib.Path", "PurePath", "os.path.join", "os.path.expanduser", "os.path.abspath",
                      "os.path.realpath", "os.path.normpath", "PosixPath"}:  # fmt: skip
            parts = [a.const if isinstance(a.const, str) else UNKNOWN for a in args]
            if (parts and any(p != UNKNOWN for p in parts)) or dotted in {"Path", "pathlib.Path", "PosixPath"}:
                return Val(const="/".join(parts) if parts else UNKNOWN)
            return Val(const=UNKNOWN) if parts else EMPTY
        if dotted in {"os.path.dirname", "os.getcwd", "Path.cwd", "os.path.basename"}:
            return Val(const=UNKNOWN)
        if dotted.endswith("Path.home") or dotted == "os.path.expanduser":
            return Val(const="~")
        if name in {"joinpath", "resolve", "expanduser", "absolute", "with_name", "with_suffix"} and base.const:
            if name == "joinpath":
                return Val(const="/".join([base.const, *(a.const or UNKNOWN for a in args)]))
            return Val(const=base.const)
        if dotted in {"os.getenv", "os.environ.get"}:
            return args[1] if len(args) > 1 else EMPTY
        if dotted in {"json.loads", "yaml.safe_load", "yaml.load", "tomllib.loads", "toml.loads"} and args:
            if isinstance(args[0].const, str):
                return self.parse_config(args[0].const, dotted)
            return EMPTY
        if dotted == "getattr" and len(args) >= 2 and isinstance(args[1].const, str):
            got = self.get_attr(args[0], args[1].const, node.args[0] if node.args else None, env, mod)
            return got if got is not None else EMPTY
        if name == "get" and base.fields is not None and args and isinstance(args[0].const, str):
            got = base.field(args[0].const)
            return got if got is not None else (args[1] if len(args) > 1 else EMPTY)
        if name in {"list", "tuple", "sorted", "set", "enumerate", "iter", "tqdm"} and isinstance(fnode, ast.Name):
            return args[0] if len(args) == 1 else merge(args)
        cls_name = fnode.id if isinstance(fnode, ast.Name) else None
        if cls_name and self.lookup(cls_name, env, mod).inst == cls_name and cls_name not in env:
            # constructing a class defined in the repo: run __init__, return the instance
            cmod = self.class_mod.get(cls_name, mod)
            init = cmod.funcs.get(f"{cls_name}.__init__")
            if init is not None:
                self.call_user(init, cmod, args, kwargs, depth + 1)
            return Val(inst=cls_name)

        # patching scoring code ------------------------------------------------------------
        patched: str | None = None
        if name == "patch" and args and isinstance(args[0].const, str):
            patched = args[0].const  # mock.patch("sklearn.metrics.roc_auc_score")
        elif (
            (name == "object" and dotted.endswith("patch.object"))
            or dotted == "setattr"
            or (name == "setattr" and dotted.startswith(("monkeypatch", "mp")))
        ):
            if node.args and len(args) >= 2 and isinstance(args[1].const, str):
                patched = f"{self.module_path(node.args[0], mod)}.{args[1].const}"
            elif node.args and len(args) == 2 and isinstance(args[0].const, str):
                patched = args[0].const  # monkeypatch.setattr("pkg.mod.fn", value)
        if patched and patched.startswith(("sklearn.metrics", "holdout_vault")):
            self.add(
                "HV-COD-006", mod, line,
                "Code replaces a scoring function",
                f"Patching {patched} swaps out scoring code at runtime. Scores computed while it is "
                "patched cannot be trusted.",
                target=patched,
            )  # fmt: skip
            return EMPTY

        # shell commands --------------------------------------------------------------------
        if name in SHELL and (dotted.startswith(("os.", "subprocess.", "commands.")) or dotted in {"run", "Popen"}):
            text = " ".join(self.const_strings(merge(args)) + [s for a in args for s in self.const_strings(a)])
            low = text.lower()
            words = set(re.findall(r"[a-z0-9_.:/-]+", low))
            reaches_in = (
                bool(words & {"exec", "cp", "attach", "commit", "export"})
                or ("volume" in words and bool(words & {"rm", "prune", "inspect", "export"}))
                or ("run" in words and bool(re.search(r"(-v|--volume|--mount|--volumes-from)\b", low)))
            )
            if "docker" in words and ("holdout" in low or "vault" in low) and reaches_in:
                self.add(
                    "HV-COD-006", mod, line,
                    "Code reaches into the vault container",
                    "This shell command runs Docker against the vault. Agent code has no reason to touch the "
                    "vault container, its volume or its files.",
                    command=text[:200],
                )  # fmt: skip
            elif any(f in text for f in (*VAULT_NAMES, *VAULT_FRAGMENTS)) and re.search(
                r"\b(rm|mv|cp|sed|tee|truncate|chmod)\b|>", text
            ):
                self.add(
                    "HV-COD-006", mod, line,
                    "Code changes vault files from the shell",
                    "This shell command edits, moves or deletes vault files.",
                    command=text[:200],
                )  # fmt: skip
            return EMPTY

        # readers and writers ---------------------------------------------------------------
        if (name == "open" and isinstance(fnode, ast.Name)) or dotted in {"io.open", "builtins.open"}:
            path = args[0].const if args else (kwargs.get("file") or EMPTY).const
            mode = args[1].const if len(args) > 1 else (kwargs.get("mode") or EMPTY).const
            if mode and any(c in mode for c in "wax+"):
                self.check_write([path] if path else [], mod, line, "write")
                return EMPTY
            return self.check_read(path, mod, line)
        if name in {"read_text", "read_bytes"} or (name == "open" and base.const):
            if name == "open":
                mode = args[0].const if args else (kwargs.get("mode") or EMPTY).const
                if mode and any(c in mode for c in "wax+"):
                    self.check_write([base.const], mod, line, "write")
                    return EMPTY
            return self.check_read(base.const, mod, line)
        if (name in READERS and name not in {"load"}) or (name == "load" and dotted.startswith(("np.", "numpy."))):
            path = args[0].const if args else next((v.const for v in kwargs.values() if v.const), None)
            return self.check_read(path, mod, line)
        if name in PATH_METHOD_WRITES and base.const:
            self.check_write([base.const], mod, line, name)
            return EMPTY
        if name in WRITERS_ARG0 and args:
            self.check_write([a.const for a in args[:1] if a.const], mod, line, name)
            return EMPTY
        if name == "dump" and len(args) > 1:
            self.check_write([a.const for a in args[1:2] if a.const], mod, line, name)
            return EMPTY
        if (name in FS_ANY_ARG and dotted.startswith(("os.", "shutil.", "Path"))) or (
            name in FS_ANY_ARG and base.const
        ):
            paths = [a.const for a in args if a.const] + ([base.const] if base.const else [])
            self.check_write(paths, mod, line, name)
            return EMPTY

        # the agent's split ------------------------------------------------------------------
        if name in SPLIT_FUNCS:
            self.confirm(args, "train_test_split")
            items: list[Val] = []
            for a in args:
                a = merge(flatten(a)) if a.items is not None else a
                items.append(replace(a, tags=(a.tags - {"full", "valid"}) | {"train"}, cands=frozenset()))
                items.append(replace(a, tags=(a.tags - {"full", "train"}) | {"valid"}, cands=frozenset()))
            return Val(items=tuple(items))
        if name in CV_FUNCS:
            self.confirm([*args, *kwargs.values()], name)
            x = args[1] if len(args) > 1 else kwargs.get("X", EMPTY)
            self.check_target_input(x, mod, line, name)
            return EMPTY
        if name == "split" and (base.est == "splitter" or (base.const is None and args and args[0].data())):
            self.confirm(args, f"{base.cls or 'a cross-validation'} split")
            return Val(
                tags=frozenset({"cvsplit"}),
                items=(Val(tags=frozenset({"idx_train"})), Val(tags=frozenset({"idx_test"}))),
            )

        # estimators -------------------------------------------------------------------------
        if name in PREPROC or name in RESAMPLERS or name in SEARCH or name in SPLITTERS or name in PIPELINES:
            kind = (
                "preproc" if name in PREPROC
                else "resampler" if name in RESAMPLERS
                else "search" if name in SEARCH
                else "splitter" if name in SPLITTERS
                else "pipeline"
            )  # fmt: skip
            passes = None
            if name in {"ColumnTransformer", "make_column_transformer"}:
                # unlisted columns are dropped unless remainder="passthrough"
                named = {c for a in [*args, *kwargs.values()] for c in self.const_strings(a)}
                passes = bool(named & self.ctx.targets) or (kwargs.get("remainder") or EMPTY).const == "passthrough"
            return Val(est=kind, cls=name, passes_target=passes)
        if name in FIT and not base.data():
            return self.fit(fnode, name, base, args, kwargs, env, mod, func, line)
        if name in APPLY and base.est and not base.data():
            x = args[0] if args else kwargs.get("X", EMPTY)
            if base.cands and x.tags & {"train", "valid"}:
                self.confirm([base], f"the {base.cls} fitted earlier")
            if name == "score":
                return EMPTY
            out = replace(x, items=None, cands=x.cands | base.cands, est=None, cls=None, const=None)
            if base.passes_target is not None and name in {"transform", "fit_transform"}:
                out = replace(out, has_target=base.passes_target, tcols=frozenset())
            return out

        # repo functions ---------------------------------------------------------------------
        user = self.resolve_user(fnode, env, mod, base)
        if user is not None:
            fn, fmod = user
            return self.call_user(fn, fmod, args, kwargs, depth + 1)

        # data plumbing ----------------------------------------------------------------------
        if name == "pop" and base.data() and args and args[0].const in self.ctx.targets:
            self.set_ref(fnode.value, replace(base, has_target=False), env, mod)  # type: ignore[union-attr]
            return Val(tags=(base.tags - {"holdout"}) | {"target"}, cands=base.cands)
        if base.colset:
            if name == "drop" or name == "difference":
                names = self.const_strings(merge([*args, *kwargs.values()])) + [
                    s for a in args for s in self.const_strings(a)
                ]
                return replace(base, has_target=base.has_target and not set(names) & self.ctx.targets)
            return base
        if name == "drop" and base.data():
            cols = kwargs.get("columns") or kwargs.get("labels") or (args[0] if args else EMPTY)
            names = self.const_strings(cols)
            if names and any(n in self.ctx.targets for n in names):
                return replace(base, has_target=False, items=None, const=None)
            if cols.data() and "train" in cols.tags and "full" in base.tags:
                return replace(base, tags=(base.tags - {"full"}) | {"valid"}, items=None, cands=frozenset())
            return replace(base, items=None, const=None)
        if name == "assign" and base.data():
            out = base
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if isinstance(kw.value, ast.Lambda) and kw.value.args.args:
                    inner = dict(env)
                    inner[kw.value.args.args[0].arg] = out
                    v = self.eval(kw.value.body, inner, mod, func, depth)
                else:
                    v = kwargs.get(kw.arg, EMPTY)
                if "target" in v.tags or v.tcols or (v.has_target and v is not out):
                    out = replace(out, has_target=True, tcols=out.tcols | {kw.arg})
            return replace(out, items=None, const=None, est=None)
        if name in COLUMN_JOIN or dotted in {"np.c_", "np.r_"}:
            vals = [x for a in [base, *args, *kwargs.values()] for x in flatten(a)]
            m = merge(vals)
            sideways = name in {"hstack", "column_stack", "c_", "assign", "join", "merge"} or (
                name in {"concat", "concatenate"} and any(k.arg == "axis" for k in node.keywords)
                and any(isinstance(k.value, ast.Constant) and k.value.value in (1, "columns")
                        for k in node.keywords if k.arg == "axis")
            )  # fmt: skip
            if sideways and any("target" in v.tags for v in vals if v is not base):
                m = replace(m, has_target=True)
            return replace(m, tags=m.tags - {"target"} if m.has_target or len(vals) > 1 else m.tags, est=None)
        if base.data():
            return replace(merge([base, *(a for a in args if a.data())]), items=None, const=None, est=None)
        if base.est:
            return EMPTY
        data_args = [a for a in [*args, *kwargs.values()] if a.data()]
        if data_args:
            return replace(merge(data_args), items=None, const=None, est=None)
        return EMPTY

    def check_target_input(self, x: Val, mod: Module, line: int, where: str) -> None:
        if x.has_target or "target" in x.tags:
            self.add(
                "HV-COD-003", mod, line,
                "The target is used as a model input",
                f"The features passed to {where} still contain the target column, or a column computed "
                "from it. The model can read the answer, so its validation score means nothing.",
                call=where,
            )  # fmt: skip

    def fit(self, fnode: ast.expr, name: str, est: Val, args: list[Val], kwargs: dict[str, Val], env: dict,
            mod: Module, func: str | None, line: int) -> Val:  # fmt: skip
        x = args[0] if args else kwargs.get("X", kwargs.get("raw_documents", EMPTY))
        y = args[1] if len(args) > 1 else kwargs.get("y", EMPTY)
        x = merge(flatten(x)) if x.items is not None else x
        cls = est.cls or (self.dotted(fnode.value) if isinstance(fnode, ast.Attribute) else None) or "estimator"
        # fitting on the validation side, or on the exported test features
        if "holdout" in x.tags or ("valid" in x.tags and "train" not in x.tags):
            which = "the exported test features" if "holdout" in x.tags else "the validation split"
            self.add(
                "HV-COD-002", mod, line,
                f"{cls} is fitted on {which}",
                f"{cls}.{name} receives {which}. Whatever it learns there leaks into the numbers measured on "
                "that same data. Fit on the training part only and call transform or predict on the rest.",
                estimator=cls, data=which,
            )  # fmt: skip
        if est.est not in {"preproc", "resampler", "splitter"}:
            self.check_target_input(x, mod, line, f"{cls}.{name}")
        if est.est == "search":
            self.confirm([x, y], f"{cls}")
        cands: frozenset[int] = frozenset()
        unsplit = "full" in x.tags and not x.tags & {"train", "valid"}
        if est.est == "preproc" and unsplit and est.cls not in STATELESS:
            cands = frozenset({self.candidate("HV-COD-001", mod, line, cls, func)})
        if est.est == "resampler" and unsplit:
            cands = frozenset({self.candidate("HV-COD-005", mod, line, cls, func)})
        fitted = replace(est, cands=est.cands | cands)
        if isinstance(fnode, ast.Attribute) and est.est and self.dotted(fnode.value):
            self.set_ref(fnode.value, fitted, env, mod)
        if name == "fit_resample":
            return Val(
                items=(replace(x, cands=x.cands | cands), replace(y, cands=y.cands | cands, items=None)),
                tags=x.tags,
            )
        if name == "fit_transform":
            out = replace(x, cands=x.cands | cands, est=None, cls=None)
            return out if est.passes_target is None else replace(out, has_target=est.passes_target, tcols=frozenset())
        if name == "fit_predict":
            return replace(x, est=None, cls=None)
        return fitted


def iter_code_files(root: Path, paths: list[Path] | None = None) -> list[Path]:
    if paths:
        out: list[Path] = []
        for p in paths:
            if p.is_dir():
                out.extend(iter_code_files(p))
            elif p.suffix in {".py", ".ipynb"}:
                out.append(p)
        return out
    out = []
    for path in sorted(root.rglob("*")):
        if len(out) >= MAX_FILES:
            break
        if path.suffix not in {".py", ".ipynb"} or not path.is_file():
            continue
        rel = path.relative_to(root).parts
        if any(part in SKIP_DIRS or (part.startswith(".") and part not in {"."}) for part in rel[:-1]):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        out.append(path)
    return out


@dataclass
class CodeReport:
    findings: list[Finding]
    files: int
    unparsed: list[str]


def code_findings(root: str | Path, ctx: Context, paths: list[str | Path] | None = None) -> CodeReport:
    root = Path(root).resolve()
    files = iter_code_files(root, [Path(p) if Path(p).is_absolute() else root / p for p in paths] if paths else None)
    an = Analyzer(root, ctx)
    an.load(files)
    an.run()
    order = {"block": 0, "warn": 1, "info": 2}
    found = sorted(
        an.findings.values(),
        key=lambda f: (order[f.severity], f.location.get("file", ""), f.location.get("line", 0), f.code),
    )
    return CodeReport(found, len(files), an.errors)
