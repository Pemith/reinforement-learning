"""
REINFORCE with Baseline Subtraction for InvertedPendulum-v4
=============================================================

Extends the vanilla REINFORCE tutorial
(https://gymnasium.farama.org/tutorials/training_agents/mujoco_reinforce/)
with a learned baseline b(s_t) = V(s_t), the expected return from state
s_t under the current policy, estimated by a separate value network.

Key changes relative to the base tutorial:
  1. A `Value_Network` is added: a small MLP that maps a state to a
     scalar estimate of V(s), trained by regression (MSE) toward the
     Monte-Carlo return G_t observed in that episode.
  2. The agent now also stores the sequence of visited states
     (`self.states`) for each episode, needed to query the value
     network during `update()`.
  3. `update()` computes:
         advantage_t = G_t - V(s_t)          (baseline-subtracted return)
     and the policy loss becomes
         L_policy = -sum_t log pi(a_t|s_t) * advantage_t.detach()
     while the value network is trained separately with
         L_value = mean_t (V(s_t) - G_t)^2
  4. Two independent Adam/AdamW optimizers are used (policy, value),
     each stepped once per episode -- exactly analogous to the single
     policy update in vanilla REINFORCE.

Everything else (policy network architecture, discounting, training
loop, seeds, episode budget) is kept identical to the base tutorial so
that the comparison between the two algorithms isolates the effect of
the baseline.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


class Policy_Network(nn.Module):
    """Parametrized Policy Network (identical to the base tutorial)."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        return action_means, action_stddevs


class Value_Network(nn.Module):
    """Parametrized state-value function V(s), used as the REINFORCE baseline."""

    def __init__(self, obs_space_dims: int):
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        self.net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
            nn.Linear(hidden_space2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).squeeze(-1)


class REINFORCE_Baseline:
    """REINFORCE with a learned state-value baseline."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        # Hyperparameters
        self.learning_rate = 1e-4     # policy LR (same as base tutorial)
        self.value_learning_rate = 1e-3  # value net can learn faster
        self.gamma = 0.99
        self.eps = 1e-6

        self.probs = []
        self.rewards = []
        self.states = []   # >>> added: needed to query V(s_t) at update time

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.value_net = Value_Network(obs_space_dims)   # >>> added

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.AdamW(
            self.value_net.parameters(), lr=self.value_learning_rate
        )  # >>> added

    def sample_action(self, state: np.ndarray) -> float:
        state_t = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state_t)

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        self.probs.append(prob)
        self.states.append(state_t)  # >>> added

        return action.numpy()

    def update(self):
        """Updates the policy network using baseline-subtracted returns,
        and the value network via MSE regression toward the returns."""
        running_g = 0
        gs = []
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        returns = torch.tensor(gs, dtype=torch.float32)

        states = torch.cat(self.states, dim=0)          # >>> added
        values = self.value_net(states)                  # >>> added: V(s_t)

        # Baseline-subtracted advantage. Detached so that policy gradients
        # do not flow into the value network (and vice-versa).
        advantages = (returns - values).detach()          # >>> added

        log_probs = torch.stack(self.probs).squeeze()

        # ---- Policy update ----
        policy_loss = -torch.sum(log_probs * advantages)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # ---- Value (baseline) update ----                # >>> added
        value_loss = torch.nn.functional.mse_loss(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.probs = []
        self.rewards = []
        self.states = []


def run(total_num_episodes: int = 5000, seeds=(1, 2, 3, 5, 8), out_dir: str = "../results"):
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make("InvertedPendulum-v4")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    rewards_over_seeds = []

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        agent = REINFORCE_Baseline(obs_space_dims, action_space_dims)
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            obs, info = wrapped_env.reset(seed=seed)

            done = False
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)
                done = terminated or truncated

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update()

            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print(f"[REINFORCE+Baseline][seed {seed}] Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

        np.save(os.path.join(out_dir, f"reinforce_baseline_seed{seed}.npy"),
                np.array(reward_over_episodes, dtype=np.float32))

    np.save(os.path.join(out_dir, "reinforce_baseline_all_seeds.npy"),
            np.array(rewards_over_seeds, dtype=np.float32))

    return rewards_over_seeds


def main():
    parser = argparse.ArgumentParser(description="REINFORCE with baseline on InvertedPendulum-v4")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--out_dir", type=str, default="../results")
    args = parser.parse_args()

    run(total_num_episodes=args.episodes, seeds=tuple(args.seeds), out_dir=args.out_dir)


if __name__ == "__main__":
    main()
