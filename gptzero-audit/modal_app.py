"""Modal entrypoints for reproducible remote execution and verification."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import modal


APP_NAME = "gptzero-audit-repro"
REMOTE_PROJECT_DIR = "/root/gptzero-audit"
DEFAULT_OUTPUT_DIR = "/artifacts/gptzero-audit"

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name("gptzero-audit-artifacts", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.lock")
    .add_local_dir(".", remote_path=REMOTE_PROJECT_DIR)
)


@app.function(
    image=image,
    volumes={"/artifacts": artifacts_volume},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("gptzero-audit-secrets")],
)
def run_pipeline(config: Dict[str, object]) -> Dict[str, object]:
    """
    Run the full pipeline in Modal with deterministic config.

    Required secret:
      GPTZERO_API_KEY in `gptzero-audit-secrets`
    """
    import os

    api_key = os.environ.get("GPTZERO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GPTZERO_API_KEY in Modal secret `gptzero-audit-secrets`."
        )

    run_id = str(config.get("run_id", "modal-run"))
    seed = int(config.get("seed", 42))
    dataset_version = str(config.get("dataset_version", "v1"))
    delay = float(config.get("delay", 0.5))
    skip_test = bool(config.get("skip_test", False))
    no_network_cache = bool(config.get("no_network_cache", False))
    output_dir = str(config.get("output_dir", DEFAULT_OUTPUT_DIR))

    cmd = [
        "python",
        "main.py",
        "--api-key",
        api_key,
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--dataset-version",
        dataset_version,
        "--delay",
        str(delay),
        "--output-dir",
        output_dir,
    ]
    if skip_test:
        cmd.append("--skip-test")
    if no_network_cache:
        cmd.append("--no-network-cache")

    proc = subprocess.run(
        cmd,
        cwd=REMOTE_PROJECT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    manifest_path = (
        Path(output_dir).resolve() / "artifacts" / run_id / "run_manifest.json"
    )
    result = {
        "returncode": proc.returncode,
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


@app.function(
    image=image,
    volumes={"/artifacts": artifacts_volume},
    timeout=15 * 60,
)
def verify_repro(
    run_ids: List[str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Compare two or more runs and return reproducibility verdict."""
    import sys

    if REMOTE_PROJECT_DIR not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_DIR)
    from src.verification import verify_runs

    if len(run_ids) < 2:
        raise ValueError("verify_repro requires at least two run_ids")

    base = Path(output_dir)
    manifest_paths = [
        base / "artifacts" / run_id / "run_manifest.json"
        for run_id in run_ids
    ]
    return verify_runs(
        manifest_paths=manifest_paths,
        tolerances=tolerances or {},
    )


@app.local_entrypoint()
def main(
    run_id: str = "modal-run",
    seed: int = 42,
    dataset_version: str = "v1",
    delay: float = 0.5,
    skip_test: bool = False,
    no_network_cache: bool = False,
) -> None:
    """Local entrypoint for Modal execution."""
    result = run_pipeline.remote(
        {
            "run_id": run_id,
            "seed": seed,
            "dataset_version": dataset_version,
            "delay": delay,
            "skip_test": skip_test,
            "no_network_cache": no_network_cache,
        }
    )
    print(json.dumps(result, indent=2))
