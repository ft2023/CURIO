from __future__ import annotations

import re

import numpy as np

from core.archive import ArchiveNode
from core.evaluator import TaskEvaluationRequirements
from tasks.ac3.prompt import build_ac3_state_context, default_initial_code, format_fenced_python, render_ac3_prompt
from tasks.ac3.evaluator import AC3_CPUS_PER_EVAL, evaluate_candidate_code, evaluate_sequence


AC3_INITIAL_STATE_CREATION_SEED = 12345
AC3_BUDGET_SECONDS = 1000
AC3_EVAL_TIMEOUT_SECONDS = 1100
MIN_CONSTRUCTION_LEN = 50
MAX_CONSTRUCTION_LEN = 100000

# Reward shaping aligned with ThetaEvolve score_transform (minimize C3):
#   working=-C3, range=[-WORST, -TARGET]; reward = MULT * linear**ALPHA
AC3_REWARD_TARGET = 1.4557   # score_range_min (best achievable / SOTA)
AC3_REWARD_WORST = 3.2       # score_range_max (worst baseline, initial ~3.159)
AC3_REWARD_ALPHA = 3.0
AC3_REWARD_MULT = 3.0

CODE_RE = re.compile(r"```python\s*\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)


def _cos_window_construction(rng) -> list[float]:
    """Window * oscillation seed on [-1/4, 1/4]; values may be negative."""
    n = 600
    x = np.linspace(-0.25, 0.25, n)
    B = float(rng.uniform(0.8, 1.4))
    f = 1.0 + B * np.cos(2.0 * np.pi * x)
    half = n // 4  # middle 50% support
    mid = n // 2
    window = np.zeros(n)
    window[mid - half: mid + half] = 1.0
    return [float(v) for v in (f * window)]


class AC3Task:
    name = "ac3"
    maximize_raw_score = False
    requires_external_evaluator_python = True

    def evaluation_resources(self) -> TaskEvaluationRequirements:
        return TaskEvaluationRequirements(cpus_per_eval=AC3_CPUS_PER_EVAL)

    def make_initial_state(self) -> ArchiveNode:
        rng = np.random.default_rng(AC3_INITIAL_STATE_CREATION_SEED)
        construction = _cos_window_construction(rng)
        bound = evaluate_sequence(construction)
        return ArchiveNode(
            epoch=-1,
            value=-float(bound),
            task_payload={
                "construction": construction,
                "code": default_initial_code(AC3_BUDGET_SECONDS),
            },
        )

    def refresh_initial_state(self, state: ArchiveNode) -> None:
        rng = np.random.default_rng()
        construction = _cos_window_construction(rng)
        bound = evaluate_sequence(construction)
        state.value = -float(bound)
        state.task_payload["construction"] = construction

    def render_prompt(self, state: ArchiveNode) -> str:
        construction = list(state.task_payload.get("construction") or [])
        code = str(state.task_payload.get("code") or "")
        state_ctx = build_ac3_state_context(
            state,
            code=code,
            construction=construction,
        )
        return render_ac3_prompt(state_ctx=state_ctx, budget_s=AC3_BUDGET_SECONDS)

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
        _ = epoch
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
            parent_construction=list(state.task_payload.get("construction") or []),
            timeout_s=AC3_EVAL_TIMEOUT_SECONDS,
            budget_s=AC3_BUDGET_SECONDS,
            seed=seed,
            resources=resources,
        )

    def compute_reward(self, eval_output: dict[str, object]) -> float:
        # ThetaEvolve score_transform (minimize C3): flip sign, linear-map to
        # [0,1] over [TARGET, WORST], power-scale by ALPHA, scale by MULT.
        correctness = float(eval_output.get("correctness", 0.0) or 0.0)
        if correctness <= 0:
            return 0.0
        raw_score = eval_output.get("raw_score")
        if raw_score is None:
            return 0.0
        c3 = float(raw_score)
        working = -c3
        range_min = -AC3_REWARD_WORST
        range_max = -AC3_REWARD_TARGET
        clamped = max(range_min, min(range_max, working))
        linear = (clamped - range_min) / (range_max - range_min)
        return AC3_REWARD_MULT * (linear ** AC3_REWARD_ALPHA)

    def make_next_state(
        self,
        *,
        parent_state: ArchiveNode,
        parsed_code: str,
        eval_output: dict[str, object],
        epoch: int,
    ) -> ArchiveNode | None:
        _ = parent_state
        payload = dict(eval_output.get("result_payload") or {})
        construction = payload.get("result_construction")
        raw_score = eval_output.get("raw_score")
        if construction is None:
            return None
        if raw_score is None:
            return None
        return ArchiveNode(
            epoch=epoch,
            value=-float(raw_score),
            task_payload={
                "construction": list(construction),
                "code": format_fenced_python(parsed_code),
                "stdout": str(eval_output.get("stdout", "")),
            },
        )

    def dedupe_key(self, state: ArchiveNode):
        construction = state.task_payload.get("construction")
        if construction is None:
            return None
        return tuple(float(item) for item in construction)

    def is_state_valid(self, state: ArchiveNode) -> bool:
        construction = list(state.task_payload.get("construction") or [])
        return MIN_CONSTRUCTION_LEN <= len(construction) <= MAX_CONSTRUCTION_LEN and state.value is not None


def build_task() -> AC3Task:
    return AC3Task()
