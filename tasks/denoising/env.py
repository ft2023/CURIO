from __future__ import annotations

import re

from core.archive import ArchiveNode
from core.evaluator import TaskEvaluationRequirements
from tasks.denoising.evaluator import DENOISING_CPUS_PER_EVAL, DENOISING_SEED, evaluate_candidate_code
from tasks.denoising.prompt import build_denoising_state_context, format_fenced_python, render_denoising_prompt

DENOISING_BUDGET_SECONDS = 400
DENOISING_EVAL_TIMEOUT_SECONDS = 600

# Initial MAGIC implementation (from TTT-Discover baseline)
MAGIC_INITIAL_CODE = """\
def magic_denoise(X, knn=5, t=3, n_pca=100, solver="approximate", decay=1, knn_max=None, random_state=None, n_jobs=1, verbose=False):
    import numpy as np
    import graphtools
    import scprep

    if knn_max is None:
        knn_max = knn * 3

    X_work = scprep.utils.toarray(X).astype(np.float64)
    X_work = np.sqrt(X_work)
    X_work, libsize = scprep.normalize.library_size_normalize(X_work, rescale=1, return_library_size=True)

    graph = graphtools.Graph(
        X_work,
        n_pca=n_pca if X_work.shape[1] > n_pca else None,
        knn=knn,
        knn_max=knn_max,
        decay=decay,
        thresh=1e-4,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    diff_op = graph.diff_op

    if solver == "approximate":
        data = graph.data_nu
    else:
        data = scprep.utils.to_array_or_spmatrix(graph.data)

    data_imputed = scprep.utils.toarray(data)

    if t > 0 and diff_op.shape[1] < data_imputed.shape[1]:
        diff_op_t = np.linalg.matrix_power(scprep.utils.toarray(diff_op), t)
        data_imputed = diff_op_t.dot(data_imputed)
    else:
        for _ in range(t):
            data_imputed = diff_op.dot(data_imputed)

    if solver == "approximate":
        data_imputed = graph.inverse_transform(data_imputed, columns=None)

    data_imputed = np.square(data_imputed)
    data_imputed = scprep.utils.matrix_vector_elementwise_multiply(data_imputed, libsize, axis=0)

    return data_imputed
"""

# Known score of the MAGIC baseline (measured on pancreas/inDrop1 seed=42).
#
# This number is EVALUATOR-ENVIRONMENT DEPENDENT and must match whatever
# CURIO_EVAL_PYTHON points at, because the archive root is scored with
# this constant while its children are scored by the live evaluator.
#   curio-eval-denoising (numpy 1.23 / scipy 1.9 / sklearn 1.1) -> 0.261801
#   newer stack                 (numpy 1.26 / scipy 1.15 / sklearn 1.7) -> 0.231412
# The metric code and the data split are byte-identical in both (no-denoising
# 0.304721 / perfect 0.0 / Poisson 0.257575 / 0.031739); the gap comes from
# sklearn's randomized PCA inside graphtools changing across versions.
# The newer stack is the one that agrees with discover's own declared 0.2316,
# with ThetaEvolve's recorded pancreas score, and with MAGIC's published 0.64 on
# PBMC/Tabula, so the evaluator was moved to it and this holds its measurement.
MAGIC_INITIAL_MSE = 0.231412
MAGIC_INITIAL_POISSON = 0.036922  # same in both stacks (0.036935 / 0.036922)

CODE_RE = re.compile(r"```python\s*\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)


class DenoisingTask:
    name = "denoising"
    maximize_raw_score = False
    requires_external_evaluator_python = True

    def evaluation_resources(self) -> TaskEvaluationRequirements:
        return TaskEvaluationRequirements(cpus_per_eval=DENOISING_CPUS_PER_EVAL)

    def make_initial_state(self) -> ArchiveNode:
        return ArchiveNode(
            epoch=-1,
            value=-MAGIC_INITIAL_MSE,
            task_payload={
                "construction": [],
                "code": MAGIC_INITIAL_CODE,
                "mse": MAGIC_INITIAL_MSE,
                "poisson": MAGIC_INITIAL_POISSON,
            },
        )

    def render_prompt(self, state: ArchiveNode) -> str:
        state_ctx = build_denoising_state_context(state)
        return render_denoising_prompt(state_ctx=state_ctx, budget_s=DENOISING_BUDGET_SECONDS)

    def parse_code(self, response_text: str) -> str:
        matches = list(CODE_RE.finditer(response_text or ""))
        if not matches:
            return ""
        return matches[-1].group(1).strip()

    def evaluate_code(
        self,
        *,
        parsed_code: str,
        state: ArchiveNode,
        epoch: int,
        seed: int,
        resources=None,
    ) -> dict[str, object]:
        _ = (epoch, state)
        if not parsed_code.strip():
            return {
                "score": 0.0,
                "msg": "cannot extract python code from model response",
                "correctness": 0.0,
                "performance": 0.0,
                "raw_score": None,
                "result_payload": {},
                "stdout": "",
            }
        return evaluate_candidate_code(
            code=parsed_code,
            timeout_s=DENOISING_EVAL_TIMEOUT_SECONDS,
            seed=seed,
            resources=resources,
        )

    def compute_reward(self, eval_output: dict[str, object]) -> float:
        correctness = float(eval_output.get("correctness", 0.0) or 0.0)
        if correctness <= 0:
            return 0.0
        raw_score = eval_output.get("raw_score")
        if raw_score is None:
            return 0.0
        return 1.0 / (1e-8 + float(raw_score))

    def make_next_state(
        self,
        *,
        parent_state: ArchiveNode,
        parsed_code: str,
        eval_output: dict[str, object],
        epoch: int,
    ) -> ArchiveNode | None:
        _ = parent_state
        raw_score = eval_output.get("raw_score")
        if raw_score is None:
            return None
        payload = dict(eval_output.get("result_payload") or {})
        mse = payload.get("mse", raw_score)
        poisson = payload.get("poisson")
        return ArchiveNode(
            epoch=epoch,
            value=-float(mse),
            task_payload={
                "construction": [],
                "code": format_fenced_python(parsed_code),
                "mse": float(mse),
                "poisson": float(poisson) if poisson is not None else None,
                "stdout": str(eval_output.get("stdout", "")),
            },
        )

    def dedupe_key(self, state: ArchiveNode):
        # No deduplication: code-space solutions don't have a compact fingerprint
        return None

    def is_state_valid(self, state: ArchiveNode) -> bool:
        return state.value is not None


def build_task() -> DenoisingTask:
    return DenoisingTask()
