"""Persisted spend guard: adds up real (non-cached) API spend across runs and stops past the cap."""
import json
from pathlib import Path

from fastlane.files import write_atomic


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, path, cap_usd):
        self.path, self.cap = Path(path), cap_usd
        self.total = json.loads(self.path.read_text())["usd"] if self.path.exists() else 0.0

    def spend(self, usd):
        """Pass as an LLM's on_spend: records each real call, raises once the total passes the cap."""
        self.total += usd
        write_atomic(self.path, json.dumps({"usd": self.total}))
        if self.total > self.cap:
            raise BudgetExceeded(f"spent ${self.total:.3f} of ${self.cap:.2f}")
