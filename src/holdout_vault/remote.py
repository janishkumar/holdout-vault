"""Front-door client for a sealed vault container. Mirrors the parts of :class:`Vault` the server uses."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from holdout_vault.core.errors import BudgetExhausted, Refused, TamperDetected, ValidationError, VaultError
from holdout_vault.core.findings import Finding
from holdout_vault.core.store import default_home
from holdout_vault.core.vault import Approval, BudgetState, ScoreResult

SEALED_CONFIG = "sealed.json"


class RemoteVault:
    mode = "sealed"

    def __init__(self, url: str, token: str, project: str, timeout: float = 120):
        self.url, self.token, self.project, self.timeout = url.rstrip("/"), token, project, timeout

    @classmethod
    def from_config(cls, project: str, home: str | Path | None = None) -> RemoteVault | None:
        path = (Path(home) if home else default_home()) / SEALED_CONFIG
        if not path.exists():
            return None
        cfg = json.loads(path.read_text())
        return cls(cfg["url"], cfg["token"], project)

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.url}/v1/{self.project}{path}",
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except urllib.error.URLError as exc:
            raise VaultError(f"cannot reach the vault at {self.url}: {exc.reason}. Is it running?") from exc

    def _json(self, method: str, path: str, body: dict | None = None) -> Any:
        code, raw = self._request(method, path, body)
        payload = json.loads(raw) if raw else {}
        if code == 401:
            raise VaultError("the vault rejected the API token")
        if code >= 400:
            raise ValidationError(payload.get("message", f"vault error {code}"))
        return payload

    def budget_state(self) -> BudgetState:
        b = self._json("GET", "/budget")
        return BudgetState(b["budget"], b["granted"], b["spent"])

    def status(self) -> dict[str, Any]:
        return self._json("GET", "/status")

    def dataset_info(self, dataset: str) -> dict[str, Any]:
        return self._json("GET", f"/datasets/{dataset}")

    def test_features(self, dataset: str) -> pd.DataFrame:
        code, raw = self._request("GET", f"/datasets/{dataset}/features")
        if code >= 400:
            raise ValidationError(json.loads(raw).get("message", f"vault error {code}"))
        return pd.read_csv(io.BytesIO(raw))

    def score(
        self,
        dataset: str,
        predictions: pd.DataFrame,
        *,
        stamp: dict[str, Any] | None = None,
        approval: Approval | None = None,
        kind: str = "score",
    ) -> ScoreResult:
        body = {
            "dataset": dataset,
            "predictions_csv": predictions.to_csv(index=False),
            "stamp": stamp,
            "approval": asdict(approval) if approval else None,
            "kind": kind,
        }
        code, raw = self._request("POST", "/score", body)
        p = json.loads(raw) if raw else {}
        findings = [Finding(**f) for f in p.get("findings", [])]
        if code == 200:
            return ScoreResult(**{**p, "findings": findings})
        if code == 409:
            raise BudgetExhausted(dataset, p["budget"], p["spent"])
        if code == 423:
            raise TamperDetected(findings)
        if code == 403:
            raise Refused(findings)
        if code == 401:
            raise VaultError("the vault rejected the API token")
        raise ValidationError(p.get("message", f"vault error {code}"))
