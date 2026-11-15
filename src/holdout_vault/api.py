"""Internal HTTP API of the vault container.

Only the front door talks to it, over loopback, with a bearer token. It exposes exactly what the agent
may cause to happen (score, read status, read features, read the ledger). There are no endpoints for
registration, budgets or grants; those run as CLI commands inside the container.
"""

from __future__ import annotations

import hmac
import io
import os
import secrets
from pathlib import Path
from typing import Any

import pandas as pd
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from holdout_vault.core.errors import BudgetExhausted, Refused, TamperDetected, ValidationError
from holdout_vault.core.store import default_home
from holdout_vault.core.vault import Approval, Vault

TOKEN_FILE = "api_token"


def load_or_create_token(home: Path) -> str:
    env = os.environ.get("HOLDOUT_VAULT_TOKEN")
    if env:
        return env
    path = home / TOKEN_FILE
    if path.exists():
        return path.read_text().strip()
    home.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token)
    path.chmod(0o600)
    return token


def _result_json(r) -> dict[str, Any]:
    return {
        "dataset": r.dataset,
        "metrics": r.metrics,
        "n_scored": r.n_scored,
        "budget": r.budget,
        "remaining": r.remaining,
        "entry_seq": r.entry_seq,
        "entry_hash": r.entry_hash,
        "mode": r.mode,
        "stamp": r.stamp,
        "findings": [f.to_dict() for f in r.findings],
    }


def create_app(home: str | Path | None = None, token: str | None = None) -> Starlette:
    home_path = Path(home) if home else default_home()
    secret = token or load_or_create_token(home_path)

    def vault_for(request: Request) -> Vault:
        return Vault(home_path, request.path_params["project"], mode="sealed")

    def authorised(request: Request) -> bool:
        got = request.headers.get("authorization", "")
        return hmac.compare_digest(got.encode(), f"Bearer {secret}".encode())

    def guard(fn):
        async def wrapped(request: Request):
            if not authorised(request):
                return JSONResponse({"error": "unauthorised"}, status_code=401)
            try:
                return await fn(request)
            except ValueError as exc:
                return JSONResponse({"error": "invalid", "message": str(exc)}, status_code=400)

        return wrapped

    async def health(request: Request):
        return JSONResponse({"ok": True})

    async def status(request: Request):
        return JSONResponse(vault_for(request).status())

    async def budget(request: Request):
        st = vault_for(request).budget_state()
        return JSONResponse({"budget": st.budget, "granted": st.granted, "spent": st.spent})

    async def dataset(request: Request):
        try:
            return JSONResponse(vault_for(request).dataset_info(request.path_params["name"]))
        except ValidationError as exc:
            return JSONResponse({"error": "invalid", "message": str(exc)}, status_code=404)

    async def features(request: Request):
        try:
            df = vault_for(request).test_features(request.path_params["name"])
        except ValidationError as exc:
            return JSONResponse({"error": "invalid", "message": str(exc)}, status_code=404)
        return PlainTextResponse(df.to_csv(index=False), media_type="text/csv")

    async def ledger(request: Request):
        tail = int(request.query_params.get("tail", "0"))
        entries = vault_for(request).store.ledger.entries()
        chosen = entries[-tail:] if tail else entries
        return JSONResponse([e.to_dict() for e in chosen])

    async def score(request: Request):
        body = await request.json()
        vault = vault_for(request)
        approval = Approval(**body["approval"]) if body.get("approval") else None
        try:
            preds = pd.read_csv(io.StringIO(body["predictions_csv"]))
        except Exception as exc:
            return JSONResponse({"error": "invalid", "message": f"bad predictions CSV: {exc}"}, status_code=422)
        try:
            r = vault.score(
                body["dataset"],
                preds,
                stamp=body.get("stamp"),
                approval=approval,
                kind=body.get("kind", "score"),
            )
        except ValidationError as exc:
            return JSONResponse({"error": "invalid", "message": str(exc)}, status_code=422)
        except BudgetExhausted as exc:
            return JSONResponse(
                {"error": "over_budget", "message": str(exc), "budget": exc.budget, "spent": exc.spent},
                status_code=409,
            )
        except TamperDetected as exc:
            return JSONResponse(
                {"error": "tamper", "message": str(exc), "findings": [f.to_dict() for f in exc.findings]},
                status_code=423,
            )
        except Refused as exc:
            return JSONResponse(
                {"error": "refused", "message": str(exc), "findings": [f.to_dict() for f in exc.findings]},
                status_code=403,
            )
        return JSONResponse(_result_json(r))

    routes = [
        Route("/health", health),
        Route("/v1/{project}/status", guard(status)),
        Route("/v1/{project}/budget", guard(budget)),
        Route("/v1/{project}/datasets/{name}", guard(dataset)),
        Route("/v1/{project}/datasets/{name}/features", guard(features)),
        Route("/v1/{project}/ledger", guard(ledger)),
        Route("/v1/{project}/score", guard(score), methods=["POST"]),
    ]
    return Starlette(routes=routes)


def serve(host: str = "127.0.0.1", port: int = 7341, home: str | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(home), host=host, port=port, log_level="warning")
