"""Errors raised by the vault core. None of them spend a reveal."""

from __future__ import annotations

from holdout_vault.core.findings import Finding


class VaultError(Exception):
    """Base class for vault errors."""


class RegistrationError(VaultError):
    """A dataset could not be registered."""


class ValidationError(VaultError):
    """A submission is malformed (missing ids, wrong columns, bad values)."""


class BudgetExhausted(VaultError):
    """No reveals left. Only a human can approve more."""

    def __init__(self, dataset: str, budget: int, spent: int):
        self.dataset, self.budget, self.spent = dataset, budget, spent
        super().__init__(f"reveal budget exhausted: {spent} of {budget} used. A human must approve another reveal.")


class TamperDetected(VaultError):
    """The scorer, the sealed labels or the watched test file changed since registration."""

    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__("; ".join(f"{f.code}: {f.title}" for f in findings))


class Refused(VaultError):
    """A blocking integrity finding stopped the score."""

    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__("; ".join(f"{f.code}: {f.title}" for f in findings))
