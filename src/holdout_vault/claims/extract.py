"""Deterministic extraction of metric claims ("AUC of 0.91", "accuracy: 84.1%", table cells) from
Markdown, plain text and notebook markdown cells. No model is involved."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from holdout_vault.checks.artifacts import SKIP_DIRS

# canonical metric -> spellings (matched case-insensitively, as whole words)
ALIASES: dict[str, tuple[str, ...]] = {
    "balanced_accuracy": ("balanced accuracy", "balanced_accuracy", "balanced acc", "bal acc", "bacc"),
    "accuracy": ("accuracy", "acc"),
    "roc_auc": ("roc auc", "roc-auc", "roc_auc", "auroc", "auc-roc", "auc roc", "auc"),
    "f1": ("macro f1", "macro-f1", "f1-score", "f1 score", "f1_score", "f1"),
    "log_loss": ("log loss", "log-loss", "logloss", "log_loss", "cross-entropy", "cross entropy"),
    "rmse": ("rmse", "root mean squared error"),
    "mae": ("mae", "mean absolute error"),
    "r2": ("r2", "r²", "r^2", "r-squared", "r squared"),
    "precision": ("precision",),
    "recall": ("recall",),
}
UNIT_INTERVAL = {"accuracy", "balanced_accuracy", "roc_auc", "f1", "precision", "recall", "r2"}
LOWER_IS_BETTER = {"log_loss", "rmse", "mae"}

_ALIAS_LIST = sorted(((a, m) for m, names in ALIASES.items() for a in names), key=lambda t: -len(t[0]))
_ALIAS_RE = "|".join(re.escape(a) for a, _ in _ALIAS_LIST)
_PREFIX = r"(?:(?:val|valid|validation|cv|test|train|holdout|sealed|oof|mean|avg)[ _-])?"
_METRIC = rf"(?<![\w-]){_PREFIX}(?P<metric>{_ALIAS_RE})(?![\w-])"
_NUM = r"(?P<num>[-+]?(?:\d+\.\d+|\.\d+|\d+))\s*(?P<pct>%|percent\b)?"
_LINK_WORDS = [
    r"of",
    r"=",
    r":",
    r"is",
    r"was",
    r"were",
    r"reached",
    r"reaches",
    r"achieved",
    r"achieves",
    r"hit",
    r"hits",
    r"scored",
    r"scores",
    r"at",
    r"around",
    r"about",
    r"~",
    r"≈",
    r"->",
    r"→",
    r"went up to",
    r"improved to",
    r"rose to",
    r"fell to",
    r"dropped to",
    r"came in at",
    r"landed at",
    r"to",
    r"of about",
    r"of around",
    r"went from",
    r"up from",
    r"down from",
    r"improved from",
    r"from",
    r"compared to",
    r"(?:moved|climbed|rose|grew|increased|improved|dropped|fell|went|jumped|slipped|declined|decreased)\s+(?:from|to)",
    r"\*\*",
]
_LINK = r"(?:\s*(?:\([^)]{0,40}\))?\s*(?:(?:" + "|".join(_LINK_WORDS) + r")\s*)*\**\s*)"
# a short phrase between the metric and its verb: "log loss on fraud is", "AUC for entry #4 is"
_GAP_VERBS = r"(?:was|is|were|are|=|:|reached|hit|came in at|landed at|stood at|sat at|ended at|ended up at)"
_GAP = (
    r"(?:\s+(?!(?:"
    + _ALIAS_RE
    + r")(?![\w-]))(?:#\d+|[A-Za-z][\w'-]*)){1,5}?\s*"
    + _GAP_VERBS
    + r"\s*(?:(?:around|about|roughly|~)\s*)?\**\s*"
)
FORWARD = re.compile(_METRIC + "(?:" + _GAP + "|" + _LINK + ")" + _NUM, re.I)
BACKWARD = re.compile(_NUM + r"\s*\**\s*(?:on\s+|in\s+)?(?:the\s+)?" + _METRIC, re.I)
# further values for the same metric: "went from 0.81 to 0.82 and then 0.84", "0.81, 0.82 and 0.84"
CONTINUE = re.compile(
    r"\s*(?:,\s*(?:and\s+)?(?:then\s+|up from\s+|down from\s+|from\s+|compared to\s+|vs\.?\s+)?"
    + r"|(?<=\s)(?:to|and|and then|then|->|→|vs\.?|versus|/|up from|down from|from|compared to)\s+(?:then\s+)?)"
    + _NUM,
    re.I,
)
ENTRY_SUFFIX = re.compile(r"\s*(?:\(\s*)?(?:at|in|on|from|for)?\s*(?:ledger\s+)?entry\s+#?(\d+)\s*\)?", re.I)
SEQ_REF = re.compile(r"(?:ledger\s+)?(?:entry|seq|#)\s*#?\s*(\d+)\b", re.I)
SCOPES = {
    "validation": re.compile(r"\b(val(?:idation)?|cv|cross[- ]validat\w*|fold|oof|out[- ]of[- ]fold|dev)\b", re.I),
    "train": re.compile(r"\b(train(?:ing)?[- ]set|in[- ]sample|training (?:score|accuracy))\b", re.I),
    "sealed": re.compile(r"\b(sealed|holdout|hold-out|held[- ]out|test[- ]set|vault|ledger)\b", re.I),
}


@dataclass
class Claim:
    metric: str  # canonical name
    value: float  # in the metric's own units (percent converted)
    shown: str  # the number as written
    tolerance: float  # half a unit in the last place shown
    file: str
    line: int
    text: str
    scope: str = "unspecified"
    ref_seq: int | None = None
    cell: int | None = None
    flags: list[str] = field(default_factory=list)


def canonical(alias: str) -> str:
    low = alias.lower()
    for a, m in _ALIAS_LIST:
        if a == low:
            return m
    return low


def _value(metric: str, num: str, pct: bool) -> tuple[float, float] | None:
    try:
        v = float(num)
    except ValueError:
        return None
    decimals = len(num.split(".")[1]) if "." in num else 0
    tol = 0.5 * 10 ** (-decimals)
    if metric in UNIT_INTERVAL:
        if pct or 1 < v <= 100:
            v, tol = v / 100, tol / 100
        elif v > 100 or v < (-1 if metric == "r2" else 0):
            return None
    elif pct:
        return None
    return v, tol


def _scope(sentence: str) -> str:
    for name in ("validation", "train", "sealed"):
        if SCOPES[name].search(sentence):
            return name
    return "unspecified"


def _sentence(line: str, start: int, end: int) -> str:
    left = max(line.rfind(". ", 0, start), line.rfind("; ", 0, start))
    right_candidates = [i for i in (line.find(". ", end), line.find("; ", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(line)
    return line[left + 1 : right + 1].strip()


def _header_metric(text: str) -> str | None:
    m = re.fullmatch(_PREFIX + rf"(?P<metric>{_ALIAS_RE})(?:\s*\(.*\))?", text.strip("* "), re.I)
    return canonical(m.group("metric")) if m else None


_CELL = re.compile(r"\**\s*" + _NUM + r"\s*\**(?:\s*\(.*\))?")


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _clause(line: str, start: int, end: int) -> str:
    """The comma- or semicolon-delimited clause around a match: where its own citation would be."""
    left = max(line.rfind(",", 0, start), line.rfind(";", 0, start), line.rfind(". ", 0, start))
    rights = [i for i in (line.find(",", end), line.find(";", end), line.find(". ", end)) if i != -1]
    return line[left + 1 : min(rights) if rights else len(line)]


def _table_claims(rows: list[tuple[int, str]], file: str, cell: int | None) -> list[Claim]:
    """Metrics as columns (a model per row) or as rows (a model per column)."""
    out: list[Claim] = []
    header = _cells(rows[0][1])
    by_column = [_header_metric(h) for h in header]
    by_row = not any(by_column)
    for lineno, row in rows[2:]:
        cells = _cells(row)
        row_metric = _header_metric(cells[0]) if by_row and cells else None
        if by_row and row_metric is None:
            continue
        for i, cellv in enumerate(cells):
            metric = row_metric if by_row else (by_column[i] if i < len(by_column) else None)
            if metric is None or (by_row and i == 0):
                continue
            m = _CELL.fullmatch(cellv)
            if not m:
                continue
            got = _value(metric, m.group("num"), bool(m.group("pct")))
            if got is None:
                continue
            # the claim's context is its row plus the column it sits in (often a run or model name)
            column = header[i] if i < len(header) else ""
            context = f"{row.strip()} [{column}]" if column else row.strip()
            label = column if by_row else (cells[0] if cells else "")
            heading = cells[0] if by_row else header[i] if i < len(header) else ""
            scope = _scope(heading)
            scope = scope if scope != "unspecified" else _scope(label)
            seq = SEQ_REF.search(column) or SEQ_REF.search(row)
            out.append(
                Claim(metric, got[0], m.group("num") + (m.group("pct") or ""), got[1], file, lineno,
                      context, scope, int(seq.group(1)) if seq else None, cell)
            )  # fmt: skip
    return out


def extract_text(text: str, file: str, cell: int | None = None, line_offset: int = 0) -> list[Claim]:
    claims: list[Claim] = []
    lines = text.splitlines()
    in_code = False
    table: list[tuple[int, str]] = []

    def flush() -> None:
        if len(table) >= 3 and re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", table[1][1]):
            claims.extend(_table_claims(table, file, cell))
        table.clear()

    for i, line in enumerate(lines, start=1):
        lineno = i + line_offset
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
            continue
        if in_code or (line.startswith("    ") and not line.strip().startswith(("-", "*", "|"))):
            continue
        if line.strip().startswith("|"):
            table.append((lineno, line))
            continue
        flush()
        plain = re.sub(r"`[^`]*`", " ", line)  # inline code is not a claim
        taken: list[tuple[int, int]] = []
        for rx in (FORWARD, BACKWARD):
            for m in rx.finditer(plain):
                if any(s <= m.start() < e or s < m.end() <= e for s, e in taken):
                    continue
                metric = canonical(m.group("metric"))
                after = plain[m.end() : m.end() + 1]
                if after.isdigit() or plain[m.end("num") : m.end("num") + 2] in {"e-", "e+"}:
                    continue
                got = _value(metric, m.group("num"), bool(m.group("pct")))
                if got is None:
                    continue
                sentence = _sentence(plain, m.start(), m.end())
                seq = SEQ_REF.search(_clause(plain, m.start(), m.end()))
                taken.append((m.start(), m.end()))
                claims.append(
                    Claim(
                        metric,
                        got[0],
                        m.group("num") + (m.group("pct") or ""),
                        got[1],
                        file,
                        lineno,
                        sentence,
                        _scope(sentence),
                        int(seq.group(1)) if seq else None,
                        cell,
                    )
                )
                if rx is FORWARD:
                    pos = m.end()
                    if (cite := ENTRY_SUFFIX.match(plain, pos)) is not None:
                        claims[-1].ref_seq = int(cite.group(1))
                        pos = cite.end()
                    while (more := CONTINUE.match(plain, pos)) is not None:
                        nxt = _value(metric, more.group("num"), bool(more.group("pct")))
                        if nxt is None or plain[more.end() : more.end() + 1].isdigit():
                            break
                        claims.append(
                            Claim(
                                metric,
                                nxt[0],
                                more.group("num") + (more.group("pct") or ""),
                                nxt[1],
                                file,
                                lineno,
                                sentence,
                                _scope(sentence),
                                None,
                                cell,
                            )
                        )
                        taken.append((more.start(), more.end()))
                        pos = more.end()
                        if (cite := ENTRY_SUFFIX.match(plain, pos)) is not None:
                            claims[-1].ref_seq = int(cite.group(1))
                            pos = cite.end()
    flush()
    claims.sort(key=lambda c: (c.line, c.text.find(c.shown)))
    return claims


def extract_file(path: Path, rel: str) -> list[Claim]:
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text())
        except (ValueError, OSError):
            return []
        out: list[Claim] = []
        for ci, c in enumerate(nb.get("cells", [])):
            if c.get("cell_type") == "markdown":
                src = c.get("source", "")
                out.extend(extract_text("".join(src) if isinstance(src, list) else src, rel, cell=ci))
        return out
    try:
        return extract_text(path.read_text(errors="replace"), rel)
    except OSError:
        return []


DOC_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".ipynb"}


def find_documents(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if len(out) >= 200:
            break
        if p.suffix.lower() in DOC_SUFFIXES and p.is_file() and p.stat().st_size < 2_000_000:
            rel = p.relative_to(root).parts
            if not any(part in SKIP_DIRS or part.startswith(".") for part in rel[:-1]):
                out.append(p)
    return out
