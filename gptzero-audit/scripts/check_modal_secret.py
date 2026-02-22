#!/usr/bin/env python3
"""Check GPTZero key presence inside Modal secret without printing secrets."""

from __future__ import annotations

import json
import modal


app = modal.App("gptzero-secret-check")


@app.function(secrets=[modal.Secret.from_name("gptzero-audit-secrets")])
def check_secret() -> dict:
    import os

    v1 = os.environ.get("GPTZERO_API_KEY")
    v2 = os.environ.get("GPT_ZERO_API_KEY")
    active = "GPTZERO_API_KEY" if v1 else ("GPT_ZERO_API_KEY" if v2 else None)
    return {
        "has_GPTZERO_API_KEY": bool(v1),
        "has_GPT_ZERO_API_KEY": bool(v2),
        "active_key": active,
        "active_key_length": len(v1 or v2 or ""),
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(check_secret.remote(), indent=2))
