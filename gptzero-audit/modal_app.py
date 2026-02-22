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


def _run_command(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REMOTE_PROJECT_DIR,
        text=True,
        capture_output=True,
        check=False,
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

    api_key = os.environ.get("GPTZERO_API_KEY") or os.environ.get("GPT_ZERO_API_KEY")
    if not api_key:
        env_keys = sorted(k for k in os.environ.keys() if "GPT" in k and "KEY" in k)
        raise RuntimeError(
            "Missing GPTZERO_API_KEY/GPT_ZERO_API_KEY in Modal secret "
            "`gptzero-audit-secrets`. "
            f"Available GPT*KEY env vars in container: {env_keys}"
        )

    run_id = str(config.get("run_id", "modal-run"))
    seed = int(config.get("seed", 42))
    dataset_version = str(config.get("dataset_version", "v1"))
    delay = float(config.get("delay", 0.5))
    skip_test = bool(config.get("skip_test", False))
    test_only = bool(config.get("test_only", False))
    no_network_cache = bool(config.get("no_network_cache", False))
    output_dir = str(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    use_api_features = bool(config.get("use_api_features", False))
    probability_objective = bool(config.get("probability_objective", False))
    oversample = bool(config.get("oversample", False))
    calibration_method = str(config.get("calibration_method", "none"))
    cv_repeats = int(config.get("cv_repeats", 1))

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
    if test_only:
        cmd.append("--test-only")
    if no_network_cache:
        cmd.append("--no-network-cache")
    if use_api_features:
        cmd.append("--use-api-features")
    if probability_objective:
        cmd.append("--probability-objective")
    if oversample:
        cmd.append("--oversample")
    cmd.extend(["--calibration-method", calibration_method])
    cmd.extend(["--cv-repeats", str(max(1, cv_repeats))])

    proc = _run_command(cmd)

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
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("gptzero-audit-secrets")],
)
def run_pattern_benchmark(config: Dict[str, object]) -> Dict[str, object]:
    """Run post-train perturbation benchmark for one run output directory."""
    import os

    api_key = os.environ.get("GPTZERO_API_KEY") or os.environ.get("GPT_ZERO_API_KEY")
    if not api_key:
        env_keys = sorted(k for k in os.environ.keys() if "GPT" in k and "KEY" in k)
        raise RuntimeError(
            "Missing GPTZERO_API_KEY/GPT_ZERO_API_KEY in Modal secret "
            "`gptzero-audit-secrets`. "
            f"Available GPT*KEY env vars in container: {env_keys}"
        )

    output_dir = str(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    dataset_version = str(config.get("dataset_version", "v5"))
    max_base_per_stratum = int(config.get("max_base_per_stratum", 40))
    variants_per_base = int(config.get("variants_per_base", 8))
    candidate_multiplier = int(config.get("candidate_multiplier", 4))
    boundary_focus = bool(config.get("boundary_focus", True))
    seed = int(config.get("seed", 1337))
    delay = float(config.get("delay", 0.25))
    cache_name = str(config.get("cache_name", "pattern_benchmark_cache.json"))
    no_network_cache = bool(config.get("no_network_cache", False))
    blend_min_samples = int(config.get("blend_min_samples", 40))
    blend_fit_split = str(config.get("blend_fit_split", "base"))
    stratum_calibration = str(config.get("stratum_calibration", "affine"))
    calibration_min_samples = int(config.get("calibration_min_samples", 40))
    calibration_fit_split = str(config.get("calibration_fit_split", "base"))
    transition_calibration = str(config.get("transition_calibration", "none"))
    transition_min_samples = int(config.get("transition_min_samples", 40))
    transition_fit_split = str(config.get("transition_fit_split", "all"))
    transition_shrinkage = float(config.get("transition_shrinkage", 60.0))
    sentence_aux_weight = float(config.get("sentence_aux_weight", 0.0))
    query_batch_size = int(config.get("query_batch_size", 64))
    cache_flush_every = int(config.get("cache_flush_every", 1))
    run_id = str(config.get("run_id", "unknown"))

    cmd = [
        "python",
        "scripts/run_pattern_benchmark.py",
        "--api-key",
        api_key,
        "--output-dir",
        output_dir,
        "--dataset-version",
        dataset_version,
        "--max-base-per-stratum",
        str(max_base_per_stratum),
        "--variants-per-base",
        str(variants_per_base),
        "--candidate-multiplier",
        str(max(1, candidate_multiplier)),
        "--blend-min-samples",
        str(max(1, blend_min_samples)),
        "--blend-fit-split",
        blend_fit_split,
        "--stratum-calibration",
        stratum_calibration,
        "--calibration-min-samples",
        str(max(1, calibration_min_samples)),
        "--calibration-fit-split",
        calibration_fit_split,
        "--transition-calibration",
        transition_calibration,
        "--transition-min-samples",
        str(max(1, transition_min_samples)),
        "--transition-fit-split",
        transition_fit_split,
        "--transition-shrinkage",
        str(max(1.0, transition_shrinkage)),
        "--sentence-aux-weight",
        str(min(1.0, max(0.0, sentence_aux_weight))),
        "--query-batch-size",
        str(max(1, query_batch_size)),
        "--cache-flush-every",
        str(max(1, cache_flush_every)),
        "--seed",
        str(seed),
        "--delay",
        str(delay),
        "--cache-name",
        cache_name,
        "--run-id",
        run_id,
    ]
    if boundary_focus:
        cmd.append("--boundary-focus")
    else:
        cmd.append("--no-boundary-focus")
    if no_network_cache:
        cmd.append("--no-network-cache")

    proc = _run_command(cmd)
    result = {
        "returncode": proc.returncode,
        "run_id": run_id,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output_dir": output_dir,
        "pattern_results_path": str(Path(output_dir) / "data" / "pattern_benchmark_results.json"),
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
    enforce_promotion: bool = False,
    enforce_pattern: bool = False,
    promotion_targets: Optional[Dict[str, object]] = None,
    pattern_targets: Optional[Dict[str, object]] = None,
    require_stability_runs: Optional[int] = None,
    manifest_settle_retries: int = 2,
    manifest_settle_wait_seconds: float = 2.0,
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
    import time

    retries = max(1, int(manifest_settle_retries) + 1)
    for attempt in range(retries):
        result = verify_runs(
            manifest_paths=manifest_paths,
            tolerances=tolerances or {},
            enforce_promotion=bool(enforce_promotion),
            enforce_pattern=bool(enforce_pattern),
            promotion_targets=promotion_targets or {},
            pattern_targets=pattern_targets or {},
            require_stability_runs=require_stability_runs,
        )
        if not enforce_pattern:
            return result

        checks = result.get("checks", [])
        if not isinstance(checks, list):
            return result
        pattern_check = next(
            (
                check
                for check in checks
                if isinstance(check, dict) and check.get("name") == "pattern_targets_met"
            ),
            None,
        )
        if not isinstance(pattern_check, dict):
            return result
        details = pattern_check.get("details", {})
        runs = details.get("runs", []) if isinstance(details, dict) else []
        if not isinstance(runs, list):
            return result

        suspicious_missing = False
        for run in runs:
            if not isinstance(run, dict):
                continue
            failures = run.get("failures", [])
            if not isinstance(failures, list):
                continue
            if any(str(f).startswith("pattern.required_stratum_missing:") for f in failures):
                suspicious_missing = True
                break
            if "pattern_overall_missing" in [str(f) for f in failures]:
                suspicious_missing = True
                break
        if not suspicious_missing or attempt >= retries - 1:
            return result
        time.sleep(max(0.0, float(manifest_settle_wait_seconds)))

    # Defensive fallback (loop always returns).
    return result


def _run_matrix_config(
    *,
    cfg: Dict[str, object],
    run_pattern: bool,
    verify: bool,
) -> Dict[str, object]:
    run_cfgs = cfg.get("runs", [])
    if not isinstance(run_cfgs, list) or not run_cfgs:
        raise ValueError("Matrix config must include non-empty `runs` list.")

    pipeline_calls = [run_pipeline.spawn(dict(item)) for item in run_cfgs]
    pipeline_results = [call.get() for call in pipeline_calls]

    pattern_results = []
    pattern_defaults = cfg.get("pattern_benchmark", {})
    if run_pattern:
        pattern_calls = []
        for run_cfg, result in zip(run_cfgs, pipeline_results):
            p_cfg = dict(pattern_defaults) if isinstance(pattern_defaults, dict) else {}
            p_cfg.setdefault("run_id", result.get("run_id"))
            p_cfg.setdefault("output_dir", run_cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
            p_cfg.setdefault("dataset_version", run_cfg.get("dataset_version", "v5"))
            p_cfg.setdefault("seed", run_cfg.get("seed", 1337))
            p_cfg.setdefault("delay", run_cfg.get("delay", 0.25))
            pattern_calls.append(run_pattern_benchmark.spawn(p_cfg))
        pattern_results = [call.get() for call in pattern_calls]

    verify_result = None
    verify_cfg = cfg.get("verify", {})
    if verify and isinstance(verify_cfg, dict) and bool(verify_cfg.get("enabled", True)):
        run_ids = [str(item["run_id"]) for item in pipeline_results]
        if len(run_ids) >= 2:
            verify_result = verify_repro.remote(
                run_ids,
                output_dir=str(verify_cfg.get("output_dir", run_cfgs[0].get("output_dir", DEFAULT_OUTPUT_DIR))),
                tolerances=verify_cfg.get("tolerances", {}),
                enforce_promotion=bool(verify_cfg.get("enforce_promotion", False)),
                enforce_pattern=bool(verify_cfg.get("enforce_pattern", False)),
                promotion_targets=verify_cfg.get("promotion_targets", {}),
                pattern_targets=verify_cfg.get("pattern_targets", {}),
                require_stability_runs=verify_cfg.get("require_stability_runs"),
                manifest_settle_retries=int(verify_cfg.get("manifest_settle_retries", 2)),
                manifest_settle_wait_seconds=float(verify_cfg.get("manifest_settle_wait_seconds", 2.0)),
            )

    return {
        "pipeline_results": pipeline_results,
        "pattern_results": pattern_results,
        "verify_result": verify_result,
    }


@app.function(
    image=image,
    volumes={"/artifacts": artifacts_volume},
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("gptzero-audit-secrets")],
)
def run_matrix(
    config_path: str,
    run_pattern: bool = True,
    verify: bool = True,
) -> Dict[str, object]:
    """Server-side matrix runner safe for detached execution."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return _run_matrix_config(
        cfg=cfg,
        run_pattern=bool(run_pattern),
        verify=bool(verify),
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


@app.local_entrypoint()
def matrix(
    config_path: str,
    run_pattern: bool = True,
    verify: bool = True,
) -> None:
    """
    Execute a matrix of runs from JSON config.

    Expected config shape:
    {
      "runs": [ { ...main.py flags... }, ... ],
      "pattern_benchmark": { ...default pattern config... },
      "verify": {
        "enabled": true,
        "tolerances": { ... },
        "enforce_promotion": true,
        "enforce_pattern": true,
        "promotion_targets": { ... },
        "pattern_targets": { ... },
        "require_stability_runs": 3
      }
    }
    """
    result = run_matrix.remote(
        config_path=config_path,
        run_pattern=bool(run_pattern),
        verify=bool(verify),
    )
    print(json.dumps(result, indent=2))
