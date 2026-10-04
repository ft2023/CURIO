"""Intrinsic Curiosity World Model (ICWM) used by CURIO.

The ICWM learns token-level transitions in the policy's own hidden-state space.
Its states are last-layer hidden states of the CURRENT actor policy (the PEFT
model with the LoRA adapter active); see
core/trainer.py::extract_actor_hidden_states. In code the module keeps the
name ``ICM`` after the Intrinsic Curiosity Module it is derived from.

The architecture and loss are a strict reimplementation of the ICM from CD-RLHF:
  https://github.com/ernie-research/CD-RLHF
  (dschat/rlhf/rlhf_engine.py, ForwardModel + ICM classes)

The ICM consists of:
  - encoder: maps raw hidden states to a learned feature space  ϕ(s)
  - forward_model: predicts the next encoded state  ϕ̂(s_{t+1}) = f(ϕ(s_t), a_t)

Intrinsic reward = prediction error = 0.5 * ||ϕ(s_{t+1}) - ϕ̂(s_{t+1})||₂
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    """Predict next_state_hat using encoded state and action.

    Strict correspondence to CD-RLHF ForwardModel:
      input = concat(state, action) of dim hidden_dim * 2
      hidden layers: Linear → ReLU → Linear → ReLU → Linear
      output dim = hidden_dim
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim * 2, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.hidden(x)


class ICM(nn.Module):
    """Intrinsic Curiosity Module.

    Strict correspondence to CD-RLHF ICM:
      encoder: 3-layer MLP (hidden → hidden*2 → hidden*2 → hidden)
      forward_model: ForwardModel(hidden, intermediate)

    Forward:
      encoded_state = encoder(state)
      encoded_next_state = encoder(next_state)
      predicted_next_state = forward_model(encoded_state, action)
      returns (encoded_next_state, predicted_next_state)
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.forward_model = ForwardModel(hidden_dim, intermediate_dim)

    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_state = self.encoder(state)
        encoded_next_state = self.encoder(next_state)
        predicted_next_state = self.forward_model(encoded_state, action)
        return encoded_next_state, predicted_next_state


def icm_loss(
    next_state: torch.Tensor,
    next_state_hat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """ICM forward model loss.

    Strict correspondence to CD-RLHF ppo_trainer.py ICM_loss:
      0.5 * sum(||next_state_hat - next_state||₂ * mask) / mask.sum()
    """
    return 0.5 * torch.sum(
        (next_state_hat - next_state).norm(2, dim=-1) * mask
    ) / mask.sum()


def intrinsic_reward(
    next_state: torch.Tensor,
    next_state_hat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token intrinsic reward (prediction error).

    Strict correspondence to CD-RLHF ppo_trainer.py intrinsic_reward:
      reward = 0.5 * ||next_state - next_state_hat||₂ * mask
      reward = whiten(reward)
      reward = reward.detach()
    """
    reward = 0.5 * (next_state - next_state_hat).norm(2, dim=-1) * mask
    reward = whiten(reward)
    return reward.detach()


def whiten(values: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whitening normalization.

    Strict correspondence to CD-RLHF ppo_trainer.py whiten:
      whitened = (values - mean) / sqrt(var + 1e-8)
    """
    mean = torch.mean(values)
    var = torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened
