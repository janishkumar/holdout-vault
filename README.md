# holdout-vault

**Protect the scorer and the labels, and prove every number.**

Coding agents game the metric they are judged on. In BaitBench, frontier agents took a planted test-set shortcut in 57.1% of runs, and still did so more than half the time when told not to ([arXiv 2608.30724](https://arxiv.org/abs/2608.30724)). In RewardHackingAgents, agents tampered with the evaluator in about half of all episodes ([arXiv 2603.11337](https://arxiv.org/abs/2603.11337)). Telling the agent to behave does not fix this. Taking the scorer and the labels out of its reach does.

holdout-vault is an MCP server for ML work done by Claude Code, Cursor, Codex and similar agents. The held-out labels and the scoring code live in a vault the agent cannot open. The agent can only ask for aggregate scores. Every reveal spends part of a budget, going over budget asks the human, and every score is written to a hash-chained ledger with the commit, data hash, scorer hash and seed that produced it.

> Status: stage 1 of 6 (vault core). Integrity checks, claim auditing, the Tasks/Apps protocol layer and the reward-hacking benchmark are next. See [docs/design.md](docs/design.md).

## Measured so far

**Canary rows catch models built from leaked labels, without flagging honest ones.** Every registered test set gets hidden canary rows with random labels that exist only in the vault. An honest model matches them at chance, so agreement far above chance means the labels were used. Results over 200 synthetic tasks per setting, with Wilson 95% intervals (`benchmark/canary_calibration.py`, raw numbers in `benchmark/results/`):

| Submission | 20 canaries | 100 canaries (default) |
|---|---|---|
| Honest model | 0/200 flagged | 1/200 (0.5%, CI 0.1% to 2.8%) |
| 25% of labels leaked | 0/200 | 62/200 (31%) |
| 50% of labels leaked | 22/200 (11%) | 198/200 (99%) |
| 100% of labels leaked | 200/200 | 200/200 |

A flagged submission is refused, logged, and costs no reveal.

## How it works

```
 agent ──MCP──▶ front door (host)  ──loopback HTTP──▶ vault container      runner container
                score / evaluate                      labels, scorer,      model + test features
                vault_status                          ledger, budget       no labels, no network
                provenance stamps
 human ──CLI──▶ register / budget / grant / verify-ledger   (never exposed over MCP)
```

- **Sealed mode** (Docker): labels, scorer and ledger live in a volume that is never mounted anywhere else. `evaluate` runs the agent's model in a throwaway container with no network, a read-only filesystem, no vault volume and an unprivileged user, so model code cannot read the labels or edit the scorer. Only predictions come back.
- **Watch-only mode** (no Docker): the test file stays where it is. Its hash, the scorer hash and the sealed label hash are checked on every call, and canaries still work. This mode **detects, it does not prevent**, and every response says so.
- **Budget:** derived from the ledger, never from a config file. Validation errors, tamper findings and refused submissions never spend a reveal. Going over budget triggers a mid-call confirmation (a multi-round-trip request on MCP 2026-07-28, classic elicitation on older clients) that the human answers.
- **Ledger:** append-only JSONL, each entry hashed over the previous one. `verify-ledger` fails at the first edited, deleted, reordered or truncated entry, and every score handed to the agent carries the entry hash, so a later rewrite contradicts numbers already quoted.

## Quick start

```bash
uv sync
export PYTHONPATH=src

# watch-only: register a dataset (human), then point the agent at the MCP server
uv run holdout-vault register --project churn --name v1 \
  --train data/train.csv --test data/test.csv --target churned --id-col customer_id \
  --metrics accuracy,roc_auc --export-dir ./agent-workspace/data
uv run holdout-vault serve --project churn --repo ./agent-workspace

# sealed: start the vault container, register into it, then serve as above
uv run holdout-vault up
uv run holdout-vault register --sealed --project churn --name v1 ...
uv run holdout-vault grant --sealed --project churn 2 --note "final check"
uv run holdout-vault verify-ledger --sealed --project churn
```

Claude Code config:

```json
{ "mcpServers": { "holdout-vault": { "command": "uv", "args": ["run", "--project", "/path/to/holdout-vault",
  "holdout-vault", "serve", "--project", "churn", "--repo", "."] } } }
```

### Tools

| Tool | What the agent gets | Spends a reveal |
|---|---|---|
| `score(dataset, predictions_path, seed?)` | Aggregate metrics, provenance stamp, ledger entry | On success |
| `evaluate(dataset, model_path, seed?)` | Same, after running a joblib model or a `predict.py` folder in the runner | On success |
| `vault_status()` | Datasets, budget, mode, ledger health | No |

Every tool has a typed `outputSchema`. Verdicts have a `status` of `scored`, `invalid`, `over_budget`, `refused` or `tamper`, and findings carry stable codes (`HV-CAN-002`, `HV-SCR-001`, `HV-LAB-001`, `HV-TST-001`) with a plain-English explanation.

## Threat model

- The agent runs in your shell. A determined agent with Docker or host access could reach the vault. The vault makes cheating hard and visible (ledger, canaries, scorer and label hashes); it is not a security boundary against a hostile admin.
- The front door trusts the MCP client's confirmation dialog. A client set to auto-approve elicitations defeats the budget confirmation; `holdout-vault grant` is the human path that does not depend on the client.
- Watch-only mode detects, it does not prevent.
- Fewer reveals means less adaptive overfitting, not none. The budget is motivated by the reusable-holdout work of Dwork, Feldman, Hardt, Pitassi, Reingold and Roth ([*Science* 349(6248):636-638, 2015](https://doi.org/10.1126/science.aaa9375)), but scores are returned exactly, so the formal guarantees of their Thresholdout mechanism do not apply. Thresholdout is planned as an opt-in mode.
- Canaries detect predictions built from leaked labels. They do not detect an agent that reads the labels and then trains an honest-looking model on the leaked test rows only.

## What didn't work (and what replaced it)

- **20 canaries per test set.** Enough to catch a full label leak every time, but only 11% of models built on half-leaked labels. Raising the floor to 100 lifted that to 99% at a false-alarm rate in line with the test level. The 20-canary run is kept as a reproducible ablation: `benchmark/canary_calibration.py --min-canaries 20`.
- **Loading the agent's model inside the vault.** A pickle or joblib file runs arbitrary code on load, which would put agent code next to the labels. Models now run in a separate runner container that never sees the labels.

## Development

```bash
make lint
make test        # offline, small fixtures
make cov         # coverage gate 85%
uv run pytest -m docker   # sealed-mode end-to-end, needs Docker running
```

## Credits

Builds on BaitBench, RewardHackingAgents, MLR-Bench, EvilGenie and ImpossibleBench for the threat evidence, Dwork et al. (2015) and Blum and Hardt's Ladder ([arXiv 1502.04585](https://arxiv.org/abs/1502.04585)) for the theory of budgeted reveals. LeakageLens-MCP and leakguard-mcp (leakage linting) and Regor (claim tracing) inspired parts of the design; no code or rules are reused. Related projects: Sparra and claude-code-kit `holdouts` hide acceptance tests from a builder agent, and holdout-labs keeps hash-chained pre-registration ledgers for quant research.

## Licence

MIT. Author: Janish Kumar.
