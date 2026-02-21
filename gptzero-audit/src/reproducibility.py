"""Reproducibility helpers: seeds, metadata, checksums, run manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from src.dataset_io import dataset_dir


@dataclass(frozen=True)
class RunPaths:
    """Canonical path bundle for a single run."""

    output_dir: Path
    data_dir: Path
    model_dir: Path
    report_dir: Path
    artifacts_dir: Path
    run_dir: Path
    manifest_path: Path


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_run_id(explicit_run_id: Optional[str] = None) -> str:
    """Return explicit run ID, otherwise generate a UTC timestamp ID."""
    if explicit_run_id:
        return explicit_run_id
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def configure_run_paths(output_dir: Path, run_id: str) -> RunPaths:
    """Create and return all run output directories."""
    output_dir = output_dir.resolve()
    data_dir = output_dir / "data"
    model_dir = output_dir / "models"
    report_dir = output_dir / "report"
    artifacts_dir = output_dir / "artifacts"
    run_dir = artifacts_dir / run_id
    manifest_path = run_dir / "run_manifest.json"

    for path in (data_dir, model_dir, report_dir, run_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        output_dir=output_dir,
        data_dir=data_dir,
        model_dir=model_dir,
        report_dir=report_dir,
        artifacts_dir=artifacts_dir,
        run_dir=run_dir,
        manifest_path=manifest_path,
    )


def file_sha256(path: Path) -> str:
    """Return SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_dependency_versions(packages: Iterable[str]) -> Dict[str, str]:
    """Capture installed package versions."""
    versions: Dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def resolve_git_commit(project_root: Path) -> str:
    """Resolve current git commit hash if available."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def dataset_manifest_summary(dataset_version: str) -> Dict[str, object]:
    """Load dataset manifest metadata and checksum for integrity checks."""
    manifest_path = dataset_dir(dataset_version) / "manifest.json"
    if not manifest_path.exists():
        return {
            "dataset_version": dataset_version,
            "manifest_path": str(manifest_path),
            "manifest_sha256": None,
            "manifest_contents": None,
        }
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "dataset_version": dataset_version,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_contents": manifest,
    }


def build_run_metadata(
    *,
    project_root: Path,
    seed: int,
    run_id: str,
    dataset_version: str,
    args: Mapping[str, object],
    api_endpoint: str,
) -> Dict[str, object]:
    """Build static run metadata before execution starts."""
    return {
        "run_id": run_id,
        "seed": seed,
        "dataset": dataset_manifest_summary(dataset_version),
        "git_commit": resolve_git_commit(project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": capture_dependency_versions(
            (
                "numpy",
                "scikit-learn",
                "requests",
                "joblib",
                "matplotlib",
                "modal",
            )
        ),
        "api_endpoint": api_endpoint,
        "cli_args": dict(args),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_run_manifest(
    *,
    manifest_path: Path,
    run_metadata: Dict[str, object],
    artifact_paths: Mapping[str, Path],
    extra_metrics: Optional[Dict[str, object]] = None,
) -> None:
    """Write final run manifest with checksums for all artifacts."""
    artifact_entries: Dict[str, Dict[str, object]] = {}
    for key, path in artifact_paths.items():
        if path.exists():
            artifact_entries[key] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            artifact_entries[key] = {
                "path": str(path),
                "sha256": None,
                "size_bytes": None,
            }

    payload = {
        "run_metadata": run_metadata,
        "artifacts": artifact_entries,
        "metrics": extra_metrics or {},
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

