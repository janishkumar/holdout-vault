"""Human command line. Registration, budgets and grants live here and are never exposed over MCP."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from collections.abc import Sequence

from holdout_vault import __version__
from holdout_vault.core.errors import VaultError
from holdout_vault.core.registry import register_dataset
from holdout_vault.core.vault import Vault


def _vault(args: argparse.Namespace) -> Vault:
    return Vault(args.home, args.project)


def cmd_register(args: argparse.Namespace) -> int:
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else []
    rec = register_dataset(
        home=args.home,
        project=args.project,
        name=args.name,
        train_path=args.train,
        test_path=args.test,
        target=args.target,
        metrics=metrics,
        export_dir=args.export_dir,
        id_col=args.id_col,
        time_col=args.time_col,
        group_col=args.group_col,
        scorer_path=args.scorer,
        seed=args.seed,
        replace=args.replace,
    )
    print(f"Registered {rec.name} ({rec.task}) in project {rec.project}.")
    print(f"  Labels and scorer sealed in {rec.home}")
    print(f"  Agent files: {rec.export_dir}/train.csv and {rec.export_dir}/test_features.csv")
    print(f"  Id column: {rec.id_col}. Metrics: {', '.join(rec.metrics) or rec.scorer_kind + ' scorer'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    s = _vault(args).status()
    if args.json:
        print(json.dumps(s, indent=2))
        return 0
    print(f"Project {s['project']}  mode={s['mode']}")
    print(f"Reveals: {s['spent']} of {s['budget']} used, {s['remaining']} left")
    print(f"Ledger: {'OK' if s['ledger_ok'] else 'BROKEN'}, {s['ledger_entries']} entries")
    for d in s["datasets"]:
        print(f"  {d['name']}: {d['task']}, metrics {', '.join(d['metrics']) or d['scorer']}")
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    _vault(args).set_budget(args.n, by=getpass.getuser())
    print(f"Budget set to {args.n} reveals.")
    return 0


def cmd_grant(args: argparse.Namespace) -> int:
    _vault(args).grant(args.n, by=getpass.getuser(), via="cli", note=args.note)
    print(f"Granted {args.n} extra reveal(s).")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    r = _vault(args).verify_ledger()
    if r.ok:
        print(f"Ledger OK: {r.count} entries, head {r.head_hash}")
        return 0
    print(f"Ledger BROKEN at entry {r.bad_seq}: {r.reason}")
    return 1


def cmd_ledger(args: argparse.Namespace) -> int:
    entries = _vault(args).store.ledger.entries()
    for e in entries[-args.tail :] if args.tail else entries:
        print(json.dumps({"seq": e.seq, "ts": e.ts, "kind": e.kind, "payload": e.payload}))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from holdout_vault.mcp.server import run

    run(project=args.project, home=args.home, repo=args.repo, http=args.http, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="holdout-vault", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command")

    def add(name: str, fn, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--project", required=True)
        sp.add_argument("--home", default=None, help="vault store (default $HOLDOUT_VAULT_HOME or ~/.config)")
        sp.set_defaults(func=fn)
        return sp

    r = add("register", cmd_register, "seal a dataset's labels and scorer")
    r.add_argument("--name", required=True)
    r.add_argument("--train", required=True)
    r.add_argument("--test", required=True)
    r.add_argument("--target", required=True)
    r.add_argument("--metrics", default="", help="comma-separated built-in metrics")
    r.add_argument("--export-dir", required=True, help="where the agent's train and test-feature files go")
    r.add_argument("--id-col")
    r.add_argument("--time-col")
    r.add_argument("--group-col")
    r.add_argument("--scorer", help="custom scorer .py defining score(y_true, predictions)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--replace", action="store_true")

    s = add("status", cmd_status, "show budget, ledger and datasets")
    s.add_argument("--json", action="store_true")

    b = add("budget", cmd_budget, "set the total reveal budget")
    b.add_argument("n", type=int)

    g = add("grant", cmd_grant, "allow extra reveals")
    g.add_argument("n", type=int)
    g.add_argument("--note")

    add("verify-ledger", cmd_verify, "check the hash chain")

    lg = add("ledger", cmd_ledger, "print ledger entries as JSON lines")
    lg.add_argument("--tail", type=int, default=0)

    sv = add("serve", cmd_serve, "run the MCP server (stdio by default)")
    sv.add_argument("--repo", default=".", help="agent repository used for provenance stamps")
    sv.add_argument("--http", action="store_true", help="serve stateless Streamable HTTP instead of stdio")
    sv.add_argument("--port", type=int, default=7342)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help(sys.stderr)
        return 2
    try:
        return args.func(args)
    except (VaultError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
