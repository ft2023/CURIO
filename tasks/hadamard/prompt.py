from __future__ import annotations

from core.archive import ArchiveNode


HADAMARD_TARGET = 0.58
HADAMARD_METRIC_NAME = "determinant ratio"
HADAMARD_IS_MAXIMIZE = True


def format_fenced_python(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("```python"):
        return stripped
    return f"```python\n{stripped}\n```"


def default_initial_code() -> str:
    return format_fenced_python('''
import numpy as np
import random
import time

def generate_hadamard_matrix(n):
    """Generate an n x n matrix with entries +1/-1 maximizing |det|."""
    best_matrix = np.ones((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            best_matrix[i][j] = random.choice([-1, 1])
    return best_matrix
''')


def build_hadamard_state_context(
    state: ArchiveNode,
    *,
    code: str,
) -> str:
    value_ctx = f"You are iteratively optimizing {HADAMARD_METRIC_NAME}."
    improvement_direction = "higher" if HADAMARD_IS_MAXIMIZE else "lower"

    has_code = code and code.strip()
    if has_code:
        value_ctx += "\nHere is the last code we ran:\n"
        value_ctx += code if code.strip().startswith("```") else format_fenced_python(code)
    else:
        value_ctx += "\nNo previous code available."

    if state.parent_values and state.value is not None:
        before_value = state.parent_values[0] if HADAMARD_IS_MAXIMIZE else -state.parent_values[0]
        after_value = state.value if HADAMARD_IS_MAXIMIZE else -state.value
        current_gap = HADAMARD_TARGET - after_value if HADAMARD_IS_MAXIMIZE else after_value - HADAMARD_TARGET
        value_ctx += (
            f"\nHere is the {HADAMARD_METRIC_NAME} before and after running the code above "
            f"({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
        )
        value_ctx += (
            f"\nTarget: {HADAMARD_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    elif state.value is not None:
        after_value = state.value if HADAMARD_IS_MAXIMIZE else -state.value
        current_gap = HADAMARD_TARGET - after_value if HADAMARD_IS_MAXIMIZE else after_value - HADAMARD_TARGET
        value_ctx += f"\nCurrent {HADAMARD_METRIC_NAME} ({improvement_direction} is better): {after_value:.6f}"
        value_ctx += (
            f"\nTarget: {HADAMARD_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    else:
        value_ctx += f"\nTarget {HADAMARD_METRIC_NAME}: {HADAMARD_TARGET}"

    stdout = str(state.task_payload.get("stdout") or "").strip()
    if stdout:
        if len(stdout) > 500:
            stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    return value_ctx


def render_hadamard_prompt(*, state_ctx: str) -> str:
    return f'''You are an expert mathematician specializing in combinatorial optimization and Hadamard matrices.

A Hadamard matrix is an n x n matrix with entries +1 or -1 such that the absolute value of the determinant is maximized. For n=29, the theoretical maximum determinant is 1270698346568170340352 (this is (2^28) * (7^12) * 342).

Your task is to write a Python function `generate_hadamard_matrix(n)` that returns a 29x29 matrix with entries +1 or -1 that maximizes the absolute determinant. The score is |det(M)| / theoretical_max, where higher is better.

Key approaches to consider:
- Start from known Hadamard matrix constructions (Paley, Sylvester, etc.)
- Use local search: flip entries and check if determinant improves
- Use simulated annealing or genetic algorithms
- Exploit mathematical structure (circulant matrices, quadratic residues)
- For n=29, Paley construction using quadratic residues mod 29 is a strong starting point

Your function will have 600 seconds to run. The function should return a numpy array of shape (29, 29) with entries +1 or -1.

{state_ctx}

Rules:
- You must define the `generate_hadamard_matrix(n)` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''
