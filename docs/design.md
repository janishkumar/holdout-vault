# holdout-vault: design

Author: Janish Kumar
Date: 2026-11-14
Status: approved for build

## 1. Problem

Coding agents now do real modelling work, and they game the number they are judged on. In BaitBench, frontier agents took a planted test-set shortcut in 57.1% of runs, and still did so more than half the time when told not to (arXiv 2608.30724). In RewardHackingAgents, agents tampered with the evaluator in about half of all episodes and made no attempt to read test data (arXiv 2603.11337). MLR-Bench agents produced fabricated or invalid results in 80% of cases (arXiv 2505.19955). Prompt instructions do not fix this, so the fix has to be structural.

holdout-vault is an MCP server that **protects the scorer and the labels, and proves every number**. The held-out labels and the scoring code sit where the agent cannot reach them. The agent can only ask for aggregate scores, each reveal is budgeted and written to a hash-chained ledger, every submission is checked for leakage, and any metric later quoted in a README, report or chat message is checked against the ledger.

## 2. Goals, non-goals, anti-goals

**Goals**
- An agent can train, submit and get honest scores without ever holding test labels or being able to edit the scorer.
- Every score carries a provenance stamp and a ledger entry, and any quoted number can be traced to one.
- Leakage and tampering are detected with stable finding codes, plain-English explanations and exact locations.
- A benchmark with three arms (no guard, prompt-only, vault) measures the effect, with confidence intervals.
- Spec-current MCP (2026-07-28): structured output, multi-round-trip confirmation, cache hints, Tasks, MCP Apps, stdio and stateless Streamable HTTP.
- $0 to build, test and run.

**Non-goals**
- A security boundary against a hostile local admin or an agent with root on the host.
- A general experiment tracker. Existing MLflow and JSON run records are read, not replaced.
- LLM-generation evals, judges and trace analysis.
- Thresholdout or Ladder noise mechanisms in v1 (see section 4.2).

**Anti-goals (it fails even if it works)**
- It blocks legitimate modelling work. Legitimate-task completion is measured, and a drop is a bug.
- It overclaims. Watch-only mode is labelled "detect only" everywhere, and the README states the threat model plainly.
- Agent-reachable tools can change the rules (register data, raise budget, approve their own reveals).
- Scores are presented as guarantees from the reusable-holdout literature when the mechanism is only a budget.

## 3. Threat model

| Threat | Mechanism | Result |
|---|---|---|
| Agent reads test labels | Labels live only in the vault volume; the agent's workspace never mounts it | Prevented in sealed mode, detected in watch mode |
| Agent edits the scorer | Scorer lives in the vault; its hash is pinned at registration and checked on every call | Prevented in sealed mode; any change refuses to score and is logged |
| Agent ships code into the vault via a model file | Models load only in a separate runner container with test features, no labels and no network | Malicious model code cannot see labels or the scorer |
| Adaptive overfitting by repeated scoring | Reveal budget; going over requires the human | Reduced, not eliminated |
| Agent trains on test features | Canary rows in test features; canary markers searched in training artifacts; canary-label agreement in predictions | Detected |
| Agent fabricates or cherry-picks numbers | Claim audit against the ledger | Detected |
| Ledger rewritten after the fact | Hash chain; ledger head hash embedded in every stamp the agent receives | Detected |
| Agent with host or Docker-socket access | Out of scope | Made visible where possible, not prevented |

The agent runs in the user's shell. A determined agent could reach Docker or the host. The vault makes cheating hard and visible, but it is not a security boundary against a hostile admin.

## 4. Approaches considered

### 4.1 Where the agent's model runs (`evaluate`)
- **A. Separate runner container (chosen).** The model loads in a container that has test features, no labels and no network, writes predictions, and only predictions reach the scorer. Any model format works, and the job shape fits Tasks.
- B. Safe formats only (ONNX, skops). Simpler, but limits model types and skops is sklearn-only.
- C. No `evaluate` in v1. Smallest, but drops the long-job case.

Loading a pickle or joblib file executes arbitrary code. Doing that next to the labels would break the vault from inside, so A was chosen.

### 4.2 What a score returns
- **1. Exact score plus reveal budget (chosen).**
- 2. Thresholdout (Dwork et al.): noisy sealed scores, a public split, and a formal guarantee.
- 3. Ladder (Blum and Hardt): only report improvements beyond a margin.

The main observed threats are tampering and fabrication, which noise does not address. Agents in realistic tasks reveal 5 to 20 scores, where adaptive overfitting is small and the budget already caps it, and noisy scores confuse both agents and people. The README cites Dwork, Feldman, Hardt, Pitassi, Reingold and Roth, "The reusable holdout: Preserving validity in adaptive data analysis", *Science* 349(6248):636-638, 2015 (doi:10.1126/science.aaa9375), as the motivation for budgeted reveals, not as a guarantee this tool provides. Thresholdout is listed as future work.

### 4.3 Long-running `evaluate`
- **A. Implement the Tasks extension (SEP-2663, `io.modelcontextprotocol/tasks`) on the SDK's extension hook (chosen).**
- B. A custom job id plus `job_status` tool. Works everywhere, but is not the spec feature.
- C. Wait for the SDK.

The MCP Python SDK 2.3.0 ships Task types and an `Extension` / `MethodBinding` hook (the same mechanism its MCP Apps support uses) but no Tasks implementation. The implementation lives in its own module so it can be swapped for the SDK's when that lands. Clients that do not opt in to Tasks get a blocking call, as the spec requires.

### 4.4 Install shape
- **Front door plus containers (chosen).** `uvx holdout-vault` is the MCP server on the host and holds no labels or scorer. `holdout-vault up` starts the vault and runner containers with Docker Compose. The front door forwards calls to the vault over a local port.
- MCP server entirely inside Docker. Cleaner isolation, but stdio clients need `docker run -i` wiring and every client config gets Docker-specific.
- Watch-only by default. Lowest friction, but the headline feature would be off by default.

Without Docker the tool falls back to watch-only mode, loudly labelled.

## 5. Architecture

```
 agent (Claude Code / Cursor / Codex)
   │ MCP (stdio or stateless Streamable HTTP)
   ▼
 front door  (host, `uvx holdout-vault`)          human CLI (`holdout-vault ...`)
   - MCP tools/resources/apps/tasks                 - register, budget, grant, unseal
   - static + dataflow code checks                  - verify-ledger, audit
   - claim audit, provenance stamps                 - up / down
   │ HTTP on 127.0.0.1:7341 (token from vault volume, never in workspace)
   ▼
 vault container                         runner container
   - labels, scorer, ledger (volume)       - test features only (read-only volume)
   - row checks, canary checks             - no network, no labels
   - budget accounting                     - loads model, writes predictions
```

Ports: 7341 (vault API, loopback only) and 7342 (front door Streamable HTTP when used). 8000, 8001, 8080 and 5434 are taken on the development machine.

**Front door (host).** The MCP server. It owns the protocol surface, code checks, claim audit and provenance stamping (it needs the agent's git repo). It forwards scoring requests to the vault and never stores labels or scorer code.

**Vault (container).** Holds the registered datasets, scorer, ledger and budget. Exposes a small internal HTTP API used only by the front door and the CLI. The API token is created at `up` and stored in the user's config directory with 0600 permissions, outside the project workspace.

**Runner (container).** Built per `evaluate` call from a pinned image with common ML libraries. Mounts the model file read-only and the test features read-only. Network disabled. CPU, memory and time limits. Writes predictions to a scratch volume the vault reads.

**Human CLI.** The only way to register datasets, change budgets, approve reveals outside a session, unseal, and verify the ledger. None of these are exposed as MCP tools.

### Modes
- **Sealed** (Docker available): everything above.
- **Watch-only** (no Docker, or the user will not move data): the test file stays where it is. The vault logic runs in-process. The test file and scorer are hashed and checked on every call, canaries are still used, and every response, stamp and ledger entry carries `mode: "watch"`, with a "detect only" message.

## 6. Data model

**Project.** A named set of datasets with one ledger and one budget. Config at `~/.config/holdout-vault/` (sealed) or `.holdout-vault/` in the repo (watch-only; the labels never live there).

**Dataset registration** (`holdout-vault register`):
- Inputs: train split, test split, target column, optional id, time and group columns, metric list, optional custom scorer file.
- Each split is hashed (SHA-256 over a canonical row serialisation, plus file hash).
- Canary rows (default 1% of test, min 3, max 50) are synthesised from the train distribution with a marker pattern and random labels, and added to the test features. Their labels are stored only in the vault and excluded from reported metrics.
- Exports for the agent: train split and test features (without labels, with canaries). Labels and the scorer stay in the vault.
- Scorer: built-in metrics (accuracy, balanced accuracy, F1, ROC AUC, log loss, RMSE, MAE, R²) or a custom Python scorer file. The scorer hash (file plus the vault's metric code) is pinned at registration.

**Budget.** Integer reveals per project, default 20. `score` and `evaluate` spend one each. Failed calls (validation errors, refused because of findings) do not spend.

**Ledger.** Append-only JSONL in the vault volume. Each entry: `seq`, `ts`, `kind` (`register`, `score`, `evaluate`, `refused`, `grant`, `budget`, `tamper`, `unseal`), `payload`, `prev_hash`, `hash` = SHA-256 over canonical JSON of the entry without `hash`. `verify_ledger` recomputes the chain and fails on any edit, deletion or reordering. The current head hash goes into every stamp returned to the agent, so a later rewrite contradicts stamps already quoted elsewhere.

**Provenance stamp** (on every score): git commit, dirty-tree flag, dataset hash, seed (from the call or detected), lockfile hash (`uv.lock`, `poetry.lock`, `requirements*.txt`, in that order), scorer hash, ledger entry seq and hash, vault version, mode.

## 7. MCP surface

Spec 2026-07-28, `mcp>=2.3,<3`. Stdio and stateless Streamable HTTP. Deterministic `tools/list` ordering. Sampling, roots and logging are not used (deprecated).

**Tools** (all with `outputSchema`):

| Tool | Purpose | Spends reveal |
|---|---|---|
| `score(dataset, predictions_path, seed?)` | Score a predictions file (id, prediction) against sealed labels | Yes |
| `evaluate(dataset, model_path, seed?)` | Run the model in the runner, then score. Tasks-enabled | Yes |
| `check_leakage(dataset?, paths?)` | Row checks and code checks without scoring | No |
| `audit_claims(paths? \| text?)` | Check metric claims against the ledger | No |
| `vault_status()` | Datasets, budget left, mode, ledger head | No |

**Over budget.** `score` and `evaluate` return an `input_required` result (multi-round-trip request) asking the human to approve one more reveal. The approval and the client that delivered it are written to the ledger. A client that auto-approves elicitations defeats this, so the README says so, and `holdout-vault grant` is the CLI path.

**Resources** (with `ttlMs` cache hints): `ledger://{project}/entries`, `ledger://{project}/entry/{seq}`, `provenance://{project}/{seq}`, `findings://{project}/latest`.

**Tasks.** `evaluate` supports `io.modelcontextprotocol/tasks`: create, status with progress (building runner, predicting, scoring), result, cancel.

**MCP App.** `ui://holdout-vault/dashboard.html` shows ledger entries, budget, findings and the public-vs-sealed score gap. Every App-enabled tool also returns a full text summary, since Claude Code does not render Apps.

**Verdict shape.** Each finding has `code` (stable, for example `HV-ROW-001`), `severity` (`block`, `warn`, `info`), `title`, `explanation` (plain English), `location` (file, line, column, or split and row ids), and `evidence`. Blocking findings refuse to score and do not spend.

## 8. Integrity checks

**Row checks (vault side, on registration and on each submission):**
- `HV-ROW-001` exact duplicate rows across train and test (canonical row hash on features).
- `HV-ROW-002` near-duplicate rows (numeric features standardised, categorical one-hot, nearest neighbour under a distance threshold calibrated on within-train distances).
- `HV-ROW-003` time order: train rows later than the earliest test row.
- `HV-ROW-004` group overlap: the same group id in train and test.
- `HV-CAN-001` canary marker values found in training artifacts in the workspace (data files, model files, notebooks, caches).
- `HV-CAN-002` predictions agree with canary labels well above chance (binomial test), which means the labels were seen.
- `HV-SCR-001` scorer hash mismatch.
- `HV-TST-001` (watch-only) test file changed, or read after registration where the OS exposes access times.

**Code checks (front-door side, Python files and notebooks):**
- `HV-COD-001` a preprocessing fit (scaler, encoder, imputer, selector, PCA, target encoder) runs on data before it is split.
- `HV-COD-002` a fit or `fit_transform` receives test data.
- `HV-COD-003` the target, or a feature derived from it, is used as an input.
- `HV-COD-004` the test file path is read in training code.
- `HV-COD-005` resampling (SMOTE and similar) before the split.
- `HV-COD-006` the scorer or vault config is written to.

Analysis is AST-based with def-use chains inside functions and a simple call graph across functions and modules in the repo, so a fit inside a helper called before `train_test_split` is caught. Rules are written from scratch. LeakageLens-MCP and leakguard-mcp are credited as inspiration, and their code and rule text are not used.

## 9. Provenance and claims

**Run records read:** the vault ledger, MLflow local file stores (`mlruns/`) and SQLite tracking DBs, and plain JSON or JSONL metric files matched by a configurable glob. W&B offline run directories are read only if that works without an account.

**Claim extraction:** deterministic regex and table parsing for "metric name near number" patterns in Markdown, text and notebooks. An optional local Ollama model can propose extra claims from free text, but every proposed claim is re-parsed deterministically, and matching never uses a model.

**Matching:** a claim matches a ledger entry when the metric name is the same (after alias normalisation) and the value agrees within the precision shown. Labels:
- `traced`: matches a vault score.
- `mismatched`: names a run or entry, but the value differs.
- `orphaned`: no vault score with that value exists.
- `cherry-picked`: matches the best of N ≥ 3 scores for the same dataset with the others undisclosed.
- `stale`: matches a score whose commit is no longer an ancestor of HEAD.
- `untrusted-source`: matches only an MLflow or JSON record, not a vault score.

Output includes a one-line summary such as "12 metrics claimed, 9 traced, 3 unsupported".

## 10. Benchmark

Modelled on BaitBench. Public tabular datasets from UCI and OpenML with licences that allow reuse (each checked and listed). Each task has one planted opportunity: a reachable test file, an editable scorer, or a leaky feature. Three arms: no guard, prompt-only instruction, vault. Two agent families: Claude Code headless (plan usage) and a local Ollama model. Paired design: the same tasks and seeds across arms.

Reported with confidence intervals and paired tests (bootstrap and McNemar or exact binomial as fits):
1. Tamper and shortcut rate per arm.
2. Public-vs-sealed score gap.
3. Claim-check precision and recall on hand-labelled claims.
4. Legitimate task completion per arm.
5. Runtime overhead, compared with the 25 to 31% RewardHackingAgents reported.

Leakage rules get a separate precision and recall on a labelled corpus that includes real notebooks with permissive licences, not only synthetic snippets.

## 11. Stages and exit criteria

Each stage ends working, tested, and with a measured number in the README.

1. **Vault core.** Registration, `score`, ledger, budget, over-budget confirmation, human CLI, watch-only mode, then the sealed Docker mode and runner with `evaluate` (blocking). Exit: an end-to-end test registers a dataset, scores, exhausts the budget, gets confirmation and verifies the ledger; a tampered ledger and a tampered scorer are both caught.
2. **Integrity checks.** Row, canary and code checks with finding codes. Exit: precision and recall on a seeded fixture corpus.
3. **Provenance and claims.** Stamps, run-record readers, claim audit. Exit: claim-check precision and recall on a labelled fixture set.
4. **Protocol showcase.** Tasks extension, resources with cache hints, MCP App and text fallback, Streamable HTTP.
5. **Benchmark.** The headline numbers.
6. **Release.** `uvx`, PyPI, MCP registry listing, only after explicit approval.

## 12. Engineering

- Python 3.12+, uv, ruff, pytest, pandas, scikit-learn, numpy, scipy.
- Layout: `src/holdout_vault/{core,vault,runner,checks,claims,mcp,cli}`, `tests/`, `docker/`, `benchmark/`, `docs/`.
- The core (ledger, budget, scoring, checks) is plain Python with no MCP or Docker dependency, so watch-only mode and tests use it directly. Sealed mode wraps the same core in the vault container.
- Tests run offline on small fixtures. Docker-dependent tests are marked and skipped when Docker is not running.
- CI: ruff, pytest with a coverage gate (85%), pip-audit, gitleaks, package build.
- `PYTHONPATH=src` is exported in the Makefile, because uv's editable `.pth` files can be hidden on macOS.
- Raw data stays out of git. Fixtures are generated or small and licence-clean.

## 13. Left to the builder

- Exact canonical row serialisation and float formatting for hashes.
- The near-duplicate distance threshold calibration method.
- The runner image contents and resource limits.
- The internal vault API shape.
- Dashboard visual design.
- Benchmark task count, as many as plan usage allows with intervals still useful.

## 14. Prior work and credits

BaitBench (arXiv 2608.30724), RewardHackingAgents (arXiv 2603.11337), MLR-Bench (arXiv 2505.19955), EvilGenie (arXiv 2511.21654), ImpossibleBench, Dwork et al. 2015 (reusable holdout), Blum and Hardt 2015 (the Ladder, arXiv 1502.04585), Husain and Shankar on error analysis. Inspiration without reuse: LeakageLens-MCP and leakguard-mcp (leakage linting), Regor (claim tracing). Related projects to name in the README: Sparra and claude-code-kit `holdouts` (hide acceptance tests from the builder), holdout-labs falsification-ledger (hash-chained pre-registration for quant research), sealed-panel, tabular-ds-mcp, crucible.
