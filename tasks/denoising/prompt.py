from __future__ import annotations

from core.archive import ArchiveNode

DENOISING_TARGET_MSE = 0.200
DENOISING_METRIC_NAME = "MSE"
DENOISING_IS_MAXIMIZE = False

BASELINE_MSE = 0.304721
BASELINE_POISSON = 0.257575
PERFECT_POISSON = 0.031739

EVALUATE_MSE_SRC = """\
def evaluate_mse(test_data, denoised):
    test_X = scprep.utils.toarray(test_data).copy()
    denoised_X = np.asarray(denoised).copy()
    test_adata = anndata.AnnData(X=test_X)
    denoised_adata = anndata.AnnData(X=denoised_X)
    sc.pp.normalize_total(test_adata, target_sum=10000)
    sc.pp.log1p(test_adata)
    sc.pp.normalize_total(denoised_adata, target_sum=10000)
    sc.pp.log1p(denoised_adata)
    return sklearn.metrics.mean_squared_error(test_adata.X, denoised_adata.X)"""

EVALUATE_POISSON_SRC = """\
def evaluate_poisson(train_data, test_data, denoised):
    test_X = scprep.utils.toarray(test_data)
    denoised_X = np.asarray(denoised).copy()
    initial_sum = train_data.sum()
    target_sum = test_X.sum()
    denoised_scaled = denoised_X * target_sum / initial_sum
    return poisson_nll_loss(test_X, denoised_scaled)"""

SYSTEM_PROMPT = f"""\
You are an expert in computational biology and single-cell RNA-seq analysis.
Your task is to develop a denoising algorithm for scRNA-seq count data.

## Problem

Single-cell RNA-seq data is noisy due to technical dropout and low capture efficiency.
Given noisy count data, predict the true expression levels.

Your prediction is evaluated against held-out molecules using two metrics:
1. **MSE** - Mean Squared Error in log-normalized space (lower is better, this is the reward)
2. **Poisson Loss** - Poisson negative log-likelihood (hard constraint)

## Data Format

- Input `X`: numpy array of shape (n_cells, n_genes) - raw count data
- Output: numpy array of same shape - your denoised counts

## Evaluation

Your output is evaluated using these exact functions:

```python
{EVALUATE_MSE_SRC}
```

```python
{EVALUATE_POISSON_SRC}
```

## Scoring

**Poisson is a HARD CONSTRAINT.** Your solution is REJECTED if `poisson_norm < 0.97`.
- `poisson_norm = ({BASELINE_POISSON} - poisson) / ({BASELINE_POISSON} - {PERFECT_POISSON})`
- MAGIC baseline achieves poisson_norm ≈ 0.97

**Reward = 1/MSE** (after passing Poisson constraint). Lower MSE = higher reward.

## Budget & Resources

- **Time budget**: 400s for your code to run
- **CPUs**: 2 available
- The evaluator kills the candidate at a hard 600s timeout, and a killed
  candidate scores zero no matter how good its MSE would have been. Time your
  own code and stay inside the budget.

## Scalability

Search feedback is computed on a 1937-cell dataset, but a solution is only
useful if it also runs on tens of thousands of cells. Never construct or
densify an n_cells-by-n_cells graph, distance, kernel, or covariance matrix.
Keep graph and diffusion operators sparse and apply powers by repeated sparse
matrix multiplication. Do not call full numpy/scipy `eig`, `eigh`, `svd`, or
`matrix_power` on large square matrices. If a low-rank representation is
needed, use `TruncatedSVD`, `randomized_svd`, `eigsh`, or `svds` with a small
rank and explicit size guards.

## Function Signature

```python
def magic_denoise(X, **kwargs):
    # kwargs may include: budget_s, random_state, knn, t, n_pca, solver, decay, knn_max, n_jobs
    # Your implementation here
    return denoised_X  # same shape as X
```

## Rules

- Implement `magic_denoise(X, ...)` that returns a denoised array of the same shape
- Available libraries: numpy, scipy, sklearn, graphtools, scprep, scanpy, anndata
- Make all helper functions top-level (no closures or lambdas)
- No filesystem or network IO

## Key Insights from Benchmarks

- NORMALIZATION ORDER MATTERS: Denoise in raw/sqrt space first, then normalize. "Reversed normalization order" achieves Poisson ~0.98 vs ~0.55 for standard order.
- Square root transform is variance-stabilizing for Poisson distributions
- Poisson loss is highly affected by low non-zero values - push values < 1 toward zero
- The original MAGIC with reversed normalization achieves Poisson ≈ 0.97 (the constraint threshold)
"""


def format_fenced_python(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("```python"):
        return stripped
    return f"```python\n{stripped}\n```"


def build_denoising_state_context(state: ArchiveNode) -> str:
    code = str(state.task_payload.get("code") or "")
    mse = state.task_payload.get("mse")
    poisson = state.task_payload.get("poisson")

    has_code = code and code.strip()
    ctx = "You are iteratively improving a scRNA-seq denoising algorithm (minimize MSE)."

    if has_code:
        ctx += "\n\nHere is the current implementation:\n"
        ctx += code if code.strip().startswith("```") else format_fenced_python(code)
    else:
        ctx += "\n\nNo previous code available."

    if mse is not None:
        ctx += f"\n\nCurrent metrics (lower is better): MSE={mse:.6f}"
        if poisson is not None:
            poisson_range = BASELINE_POISSON - PERFECT_POISSON
            poisson_norm = (BASELINE_POISSON - poisson) / poisson_range if poisson_range > 0 else 0.0
            ctx += f", Poisson={poisson:.6f} (norm={poisson_norm:.4f})"
        gap = mse - DENOISING_TARGET_MSE
        ctx += f"\nTarget MSE: {DENOISING_TARGET_MSE}. Current gap: {gap:.6f}. Further improvements will also be generously rewarded."

    stdout = str(state.task_payload.get("stdout") or "").strip()
    if stdout:
        max_stdout = 3000
        if len(stdout) > max_stdout:
            stdout = stdout[-max_stdout:]
        ctx += f"\n\nOutput from last run:\n```\n{stdout}\n```"

    return ctx


def render_denoising_prompt(state_ctx: str, budget_s: int) -> str:
    _ = budget_s
    return f"{SYSTEM_PROMPT}\n\n{state_ctx}\n\nWrite your improved `magic_denoise` function."
