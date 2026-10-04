from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.evaluator import AllocatedEvaluationResources


DENOISING_CPUS_PER_EVAL = 1
DENOISING_SEED = 42

BASELINE_MSE = 0.304721
BASELINE_POISSON = 0.257575
PERFECT_POISSON = 0.031739
POISSON_NORM_THRESHOLD = 0.97

THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

# Source of helper functions injected into generated code at eval time.
_HELPER_IMPORTS = """\
import numpy as np
import scipy
import scipy.sparse
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import graphtools
import scprep
import anndata
import scanpy as sc
import sklearn.metrics
import math
import random
from molecular_cross_validation.mcv_sweep import poisson_nll_loss
"""

_EVALUATE_MSE_SRC = """\
def evaluate_mse(test_data, denoised):
    import scprep, anndata, scanpy as sc, sklearn.metrics
    test_X = scprep.utils.toarray(test_data).copy()
    denoised_X = np.asarray(denoised).copy()
    test_adata = anndata.AnnData(X=test_X)
    denoised_adata = anndata.AnnData(X=denoised_X)
    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)
    return sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X)
"""

_EVALUATE_POISSON_SRC = """\
def evaluate_poisson(train_data, test_data, denoised):
    import scprep
    from molecular_cross_validation.mcv_sweep import poisson_nll_loss
    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()
    initial_sum = train_data.sum()
    target_sum = test_X.sum()
    denoised_scaled = denoised_X * target_sum / initial_sum
    return poisson_nll_loss(test_X, denoised_scaled)
"""

_RUN_DENOISING_EVAL_SRC = """\
def run_denoising_eval(magic_denoise_fn, seed=42):
    import openproblems.data
    openproblems.data.no_cleanup()
    from openproblems.data.pancreas import load_pancreas
    from openproblems.tasks.denoising.datasets.utils import split_data
    import scprep
    adata = load_pancreas(test=False, keep_techs=["inDrop1"])
    adata = split_data(adata, seed=seed)
    X_train = scprep.utils.toarray(adata.obsm["train"])
    X_test = scprep.utils.toarray(adata.obsm["test"])
    Y_denoised = magic_denoise_fn(X_train, random_state=seed)
    if not np.isfinite(Y_denoised).all():
        return (np.inf, np.inf)
    if np.any(Y_denoised < 0):
        return (np.inf, np.inf)
    if Y_denoised.max() > X_train.sum():
        return (np.inf, np.inf)
    mse = evaluate_mse(X_test, Y_denoised)
    poisson = evaluate_poisson(X_train, X_test, Y_denoised)
    return (mse, poisson)
"""


def build_full_code(generated_code: str, seed: int) -> str:
    seed_line = f"_SEED = {seed}\n"
    wrapper = "\ndef run_denoising():\n    return run_denoising_eval(magic_denoise, seed=_SEED)\n"
    return (
        _HELPER_IMPORTS + "\n"
        + seed_line + "\n"
        + _EVALUATE_MSE_SRC + "\n"
        + _EVALUATE_POISSON_SRC + "\n"
        + _RUN_DENOISING_EVAL_SRC + "\n"
        + generated_code + "\n"
        + wrapper
    )


def verify_denoising(mse: float, poisson: float) -> bool:
    if not np.isfinite(mse) or not np.isfinite(poisson):
        return False
    # Beating perfect denoising is not an extra-good score, it means the output
    # was scaled down until the Poisson NLL fell below the oracle. discover
    # rejects this (examples/denoising/env.py:80-81) and so must we -- without
    # the check, a candidate can zero out small values and reach mse 0.1386 with
    # poisson 0.0213, which reads as a 22% improvement and is not real.
    if poisson < PERFECT_POISSON:
        return False
    poisson_range = BASELINE_POISSON - PERFECT_POISSON
    if poisson_range <= 0:
        return False
    poisson_norm = (BASELINE_POISSON - poisson) / poisson_range
    return poisson_norm >= POISSON_NORM_THRESHOLD


def normalize_denoising_resources(
    resources: "AllocatedEvaluationResources | None",
) -> tuple[tuple[int, ...], int]:
    cpu_ids = tuple(int(c) for c in getattr(resources, "cpu_ids", ())[:DENOISING_CPUS_PER_EVAL])
    thread_limit = max(1, len(cpu_ids) if cpu_ids else DENOISING_CPUS_PER_EVAL)
    return cpu_ids, thread_limit


def apply_worker_resource_limits(cpu_ids: tuple[int, ...], thread_limit: int) -> None:
    for env_name in THREAD_LIMIT_ENV_VARS:
        os.environ[env_name] = str(max(1, int(thread_limit)))
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(c) for c in cpu_ids))


def evaluate_candidate_in_worker(
    code: str,
    seed: int,
    result_conn: Connection,
    cpu_ids: tuple[int, ...],
    thread_limit: int,
) -> None:
    apply_worker_resource_limits(cpu_ids, thread_limit)
    stdout = io.StringIO()
    stderr = io.StringIO()

    def combined_output() -> str:
        out = stdout.getvalue()
        err = stderr.getvalue()
        if out and err:
            return f"{out}\n\n[stderr]\n{err}"
        if err:
            return f"[stderr]\n{err}"
        return out

    def send_result(payload: dict[str, Any]) -> None:
        try:
            result_conn.send(payload)
        except Exception:
            pass

    try:
        full_code = build_full_code(code, seed)
        namespace: dict[str, Any] = {"__builtins__": __builtins__}

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(full_code, namespace, namespace)  # noqa: S102
            if "run_denoising" not in namespace:
                raise ValueError("Generated code must define magic_denoise")
            result = namespace["run_denoising"]()

        if not isinstance(result, (list, tuple)) or len(result) < 2:
            send_result({"ok": False, "msg": "run_denoising must return (mse, poisson)", "stdout": combined_output()})
            return

        mse, poisson = float(result[0]), float(result[1])

        if not verify_denoising(mse, poisson):
            poisson_range = BASELINE_POISSON - PERFECT_POISSON
            poisson_norm = (BASELINE_POISSON - poisson) / poisson_range if poisson_range > 0 else 0.0
            msg = f"Invalid solution: mse={mse:.6f} poisson={poisson:.6f} poisson_norm={poisson_norm:.4f} (need >= {POISSON_NORM_THRESHOLD})"
            send_result({"ok": False, "msg": msg, "stdout": combined_output()})
            return

        send_result({
            "ok": True,
            "mse": mse,
            "poisson": poisson,
            "stdout": combined_output(),
        })
    except Exception as exc:  # noqa: BLE001
        send_result({"ok": False, "msg": str(exc), "stdout": combined_output()})
    finally:
        result_conn.close()


def evaluate_candidate_code(
    code: str,
    timeout_s: int,
    seed: int,
    resources: "AllocatedEvaluationResources | None" = None,
) -> dict[str, Any]:
    cpu_ids, thread_limit = normalize_denoising_resources(resources)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=evaluate_candidate_in_worker,
        args=(code, seed, child_conn, cpu_ids, thread_limit),
    )
    process.start()
    child_conn.close()
    process.join(timeout=timeout_s)

    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        process.close()
        parent_conn.close()
        return {
            "score": 0.0,
            "msg": f"timeout after {timeout_s}s",
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": None,
            "result_payload": {},
            "stdout": f"[evaluator]\ntimeout after {timeout_s}s\n",
        }

    if not parent_conn.poll(1.0):
        exitcode = process.exitcode
        process.close()
        parent_conn.close()
        return {
            "score": 0.0,
            "msg": f"empty evaluator result (exitcode={exitcode})",
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": None,
            "result_payload": {},
            "stdout": f"[evaluator]\nempty evaluator result (exitcode={exitcode})\n",
        }

    result = parent_conn.recv()
    process.close()
    parent_conn.close()

    if not result.get("ok", False):
        message = str(result.get("msg", "evaluation failed"))
        stdout_text = str(result.get("stdout", ""))
        if not stdout_text.strip():
            stdout_text = f"[evaluator]\n{message}\n"
        return {
            "score": 0.0,
            "msg": message,
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": None,
            "result_payload": {},
            "stdout": stdout_text,
        }

    mse = float(result["mse"])
    poisson = float(result["poisson"])
    return {
        "score": 1.0 / (1e-8 + mse),
        "msg": f"Success; mse={mse:.6f} poisson={poisson:.6f}",
        "correctness": 1.0,
        "performance": -mse,
        "raw_score": mse,
        "result_payload": {"mse": mse, "poisson": poisson},
        "stdout": str(result.get("stdout", "")),
    }
