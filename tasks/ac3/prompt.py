from __future__ import annotations

from core.archive import ArchiveNode


AC3_TARGET = 1.4557
AC3_METRIC_NAME = "C3 upper bound"
AC3_IS_MAXIMIZE = False


AC3_EVAL_FUNCTION = '''```python
import numpy as np

def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a discrete function f for the THIRD autocorrelation inequality.
    Computes C3 = abs(2 * n * max(|conv(f, f)|) / (sum(f))**2), lower is better.

    NOTE: function values MAY be negative. Sign changes (phase cancellation)
    are a key mechanism for reducing the autoconvolution peak. Only NaN/inf,
    extreme magnitudes (|x| > 1e10), and a near-zero sum are rejected.
    """
    arr = np.asarray([float(x) for x in sequence], dtype=float)
    if not np.all(np.isfinite(arr)):
        return np.inf
    if np.max(np.abs(arr)) > 1e10:
        return np.inf
    total = float(np.sum(arr))
    if abs(total) < 1e-10:
        return np.inf
    conv = np.convolve(arr, arr)
    max_conv_abs = float(np.max(np.abs(conv)))
    return float(abs(2 * len(arr) * max_conv_abs / (total ** 2)))
```'''


def format_fenced_python(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("```python"):
        return stripped
    return f"```python\n{stripped}\n```"


def default_initial_code(budget_s: int) -> str:
    return format_fenced_python(f'''
import numpy as np
import time

def build_window_oscillation(n=600, support_frac=0.5, A=1.0, B=1.0, C=2*np.pi):
    """Window * oscillation construction on [-1/4, 1/4].

    Continuous template (Host-Vinuesa): f(x) = A*(1 + B*cos(C*x)) inside a
    centered window, 0 outside. Values may be negative when B > 1.
    """
    x = np.linspace(-0.25, 0.25, n)
    f = A * (1.0 + B * np.cos(C * x))
    half = int(n * support_frac / 2)
    mid = n // 2
    window = np.zeros(n)
    window[mid - half: mid + half] = 1.0
    return (f * window)


def propose_candidate(seed=42, budget_s={budget_s}, **kwargs):
    """Local search over a window*oscillation construction for C3 minimization.

    Allows negative values (phase cancellation). Refines via adaptive-step
    perturbations focused on the convolution-peak region.
    """
    np.random.seed(seed)
    deadline = time.time() + budget_s - 10

    if 'height_sequence_1' in globals() and len(height_sequence_1) > 0 and np.random.rand() < 0.5:
        best = np.asarray(height_sequence_1, dtype=float)
    else:
        best = build_window_oscillation(n=600, B=1.0)
    best_score = evaluate_sequence(list(best))

    step = 0.05
    stale = 0
    while time.time() < deadline:
        cand = best.copy()
        # focus perturbations near the largest |conv| contributions
        idx = np.random.randint(len(cand))
        cand[idx] = cand[idx] + np.random.randn() * step  # may go negative
        score = evaluate_sequence(list(cand))
        if score < best_score:
            best_score = score
            best = cand
            stale = 0
        else:
            stale += 1
            if stale > 300:          # shrink step / escape local minima
                step = max(0.001, step * 0.5)
                stale = 0
    return list(best)
''')


def build_ac3_state_context(
    state: ArchiveNode,
    *,
    code: str,
    construction: list[float] | None,
) -> str:
    value_ctx = f"You are iteratively optimizing {AC3_METRIC_NAME}."
    improvement_direction = "higher" if AC3_IS_MAXIMIZE else "lower"

    has_code = code and code.strip()
    if has_code:
        value_ctx += "\nHere is the last code we ran:\n"
        value_ctx += code if code.strip().startswith("```") else format_fenced_python(code)
    else:
        value_ctx += "\nNo previous code available."

    if state.parent_values and state.value is not None and construction:
        before_value = state.parent_values[0] if AC3_IS_MAXIMIZE else -state.parent_values[0]
        after_value = state.value if AC3_IS_MAXIMIZE else -state.value
        current_gap = AC3_TARGET - after_value if AC3_IS_MAXIMIZE else after_value - AC3_TARGET
        value_ctx += (
            f"\nHere is the {AC3_METRIC_NAME} before and after running the code above "
            f"({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
        )
        value_ctx += (
            f"\nTarget: {AC3_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    elif state.value is not None:
        after_value = state.value if AC3_IS_MAXIMIZE else -state.value
        current_gap = AC3_TARGET - after_value if AC3_IS_MAXIMIZE else after_value - AC3_TARGET
        value_ctx += f"\nCurrent {AC3_METRIC_NAME} ({improvement_direction} is better): {after_value:.6f}"
        value_ctx += (
            f"\nTarget: {AC3_TARGET}. Current gap: {current_gap:.6f}. "
            "Further improvements will also be generously rewarded."
        )
    else:
        value_ctx += f"\nTarget {AC3_METRIC_NAME}: {AC3_TARGET}"

    stdout = str(state.task_payload.get("stdout") or "").strip()
    if stdout:
        if len(stdout) > 500:
            stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    if construction:
        value_ctx += f"\nLength of the construction: {len(construction)}"
    return value_ctx


def render_ac3_prompt(
    *,
    state_ctx: str,
    budget_s: int,
) -> str:
    return f'''You are an expert in computational optimization and harmonic analysis. Your task is to design a Python program that constructs a discrete function `f` on the domain [-1/4, 1/4] to minimize the third autocorrelation constant C3, aiming to beat the SOTA of {AC3_TARGET}.

C3 is computed by the following evaluation function (lower is better):

{AC3_EVAL_FUNCTION}

**Key Insight from Mathematical Literature (Host, Vinuesa):**
The best-known constructions are based on the product of a smooth, oscillating function and a window function with compact support. A highly successful continuous analog is `f(x) = (1 + cos(2*pi*x))` for `x` in `[-1/4, 1/4]` and `0` otherwise. IMPORTANT: function values MAY be negative — sign changes (phase cancellation) are a key mechanism for reducing the autoconvolution peak. Do NOT clamp your function to be non-negative.

**Construction Guidelines:**
1.  **Window Function:** Define a "window"/"support" for your function, centered and occupying a fraction of the total domain (e.g. the middle 50%, like `n//4` to `3*n//4`). The function is zero outside this window.
2.  **Oscillatory Component:** Inside the window use a smooth, symmetric, oscillating pattern. `A * (1 + B * cos(C * x))` is a powerful template (with B>1 it takes negative values).
3.  **Parameterization:** Explore `support_width`, amplitude `A`, modulation `B`, frequency `C`.
4.  **Discretization:** Map the continuous form onto a discrete grid (a length around 400-800 works well; n≈600 is a good default). Mind boundary conditions at the window edges.

**Refinement strategy (to push past SOTA):**
- Perform many local-search iterations (e.g. 5,000-20,000) on the current candidate.
- Multi-stage adaptive step size: start with larger changes (±0.05), then shrink (±0.01, ±0.001) to fine-tune.
- Targeted search: find the indices where `conv(f, f)` has the highest absolute values and focus perturbations there — they dominate C3.
- Escape local minima: use a simulated-annealing-style schedule; if stagnant for hundreds of iterations, apply a larger random perturbation.
- Sign flipping: systematically test flipping the signs of small segments — phase cancellation is the key mechanism for lowering the convolution peak.

Your function will have {budget_s} seconds to run and must return the best sequence (a Python list of floats, negatives allowed) it found. You may call evaluate_sequence() as many times as you want — it is already available, do not redefine or import it.

{state_ctx}

You may start your search from one of the constructions found so far via the `height_sequence_1` global variable, but you are encouraged to explore other starting points to avoid local minima.

Reason about how to further improve the construction. Try something different from the previous algorithm (different structure, heuristics, or hyperparameters). Unless you make a meaningful improvement, you will not be rewarded.

Rules:
- You must define the `propose_candidate` function as this is what will be invoked.
- You can use scientific libraries like scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math.
- You can use up to 2 CPUs.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not import evaluate_sequence yourself. Assume it will already be imported and can be directly invoked.
- **Print statements**: Use `print()` to log progress, intermediate bounds, timing info, etc. Your output will be shown back to you.
- Include a short docstring at the top summarizing your algorithm.

Make sure to think and return the final program between ```python and ```.'''
