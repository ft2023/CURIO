"""Evaluator for Hadamard matrix determinant maximization."""
from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing as mp
import os
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.evaluator import AllocatedEvaluationResources


HADAMARD_CPUS_PER_EVAL = 2
MATRIX_SIZE = 29
THEORETICAL_MAX = 1270698346568170340352


THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def det_bareiss(A: list[list[int]]) -> int:
    n = len(A)
    if n == 0:
        return 1
    M = [row.copy() for row in A]
    for k in range(n - 1):
        if M[k][k] == 0:
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    M[k], M[i] = M[i], M[k]
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = M[i][j] * M[k][k] - M[i][k] * M[k][j]
                den = M[k - 1][k - 1] if k > 0 else 1
                M[i][j] = num // den
    return M[-1][-1]


def evaluate_hadamard(matrix: np.ndarray) -> float:
    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except Exception:
            return 0.0
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return 0.0
    if matrix.shape[0] != MATRIX_SIZE:
        return 0.0
    if not np.all(np.isin(matrix, [-1, 1])):
        return 0.0
    try:
        int_matrix = matrix.astype(int).tolist()
        det_exact = det_bareiss(int_matrix)
        abs_det = abs(det_exact)
        return float(abs_det / THEORETICAL_MAX) if THEORETICAL_MAX > 0 else 0.0
    except Exception:
        return 0.0


def normalize_hadamard_resources(resources: "AllocatedEvaluationResources | None") -> tuple[tuple[int, ...], int]:
    cpu_ids = tuple(int(cpu_id) for cpu_id in getattr(resources, "cpu_ids", ())[:HADAMARD_CPUS_PER_EVAL])
    thread_limit = max(1, len(cpu_ids) if cpu_ids else HADAMARD_CPUS_PER_EVAL)
    return cpu_ids, thread_limit


def apply_worker_resource_limits(cpu_ids: tuple[int, ...], thread_limit: int) -> None:
    for env_name in THREAD_LIMIT_ENV_VARS:
        os.environ[env_name] = str(max(1, int(thread_limit)))
    if cpu_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(int(cpu_id) for cpu_id in cpu_ids))


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
        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "np": np,
            "N_MATRIX_SIZE": MATRIX_SIZE,
        }
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, namespace, namespace)
            if "generate_hadamard_matrix" not in namespace:
                raise ValueError("Generated code must define generate_hadamard_matrix")
            matrix = namespace["generate_hadamard_matrix"](MATRIX_SIZE)

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        ratio = evaluate_hadamard(matrix)
        if ratio <= 0:
            send_result({"ok": False, "msg": "invalid Hadamard matrix", "stdout": combined_output()})
            return

        send_result({
            "ok": True,
            "ratio": float(ratio),
            "matrix": matrix.tolist(),
            "stdout": combined_output(),
        })
    except Exception as exc:
        send_result({"ok": False, "msg": str(exc), "stdout": combined_output()})
    finally:
        result_conn.close()


def evaluate_candidate_code(
    code: str,
    timeout_s: int,
    seed: int,
    resources: "AllocatedEvaluationResources | None" = None,
) -> dict[str, Any]:
    cpu_ids, thread_limit = normalize_hadamard_resources(resources)
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

    ratio = float(result["ratio"])
    return {
        "score": ratio,
        "msg": f"Success; det_ratio={ratio:.6f}",
        "correctness": 1.0,
        "performance": ratio,
        "raw_score": ratio,
        "result_payload": {"result_construction": result["matrix"]},
        "stdout": str(result.get("stdout", "")),
    }
