# CSCN8020 Assignment 1 — Problems 3 & 4
# Value Iteration (standard & in-place) for 5x5 Gridworld + Off-policy Monte Carlo with Importance Sampling
# Run this script to regenerate the CSV outputs used in the report.

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import random

np.set_printoptions(precision=3, suppress=True)

@dataclass(frozen=True)
class Gridworld5x5:
    nrows: int = 5
    ncols: int = 5
    gamma: float = 0.9
    goal: Tuple[int, int] = (4, 4)
    greys: Tuple[Tuple[int, int], ...] = ((2, 2), (3, 0), (0, 4))  # s2,2, s3,0, s0,4
    # Actions: right, down, left, up
    actions: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.nrows and 0 <= c < self.ncols

    def step(self, state: Tuple[int, int], action_idx: int) -> Tuple[Tuple[int, int], float, bool]:
        r, c = state
        dr, dc = self.actions[action_idx]
        nr, nc = r + dr, c + dc
        if not self.in_bounds(nr, nc):
            nr, nc = r, c  # invalid action => stay
        next_state = (nr, nc)

        # Rewards
        if next_state == self.goal:
            reward = 10.0
        elif next_state in self.greys:
            reward = -5.0
        else:
            reward = -1.0

        done = (next_state == self.goal)
        return next_state, reward, done

    def reward_of(self, state: Tuple[int, int]) -> float:
        if state == self.goal:
            return 10.0
        if state in self.greys:
            return -5.0
        return -1.0

    def states(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.nrows) for c in range(self.ncols)]


def idx_map(env: Gridworld5x5) -> Dict[Tuple[int, int], int]:
    return {s: i for i, s in enumerate(env.states())}


def greedy_policy_from_V(env: Gridworld5x5, V: np.ndarray) -> np.ndarray:
    s2i = idx_map(env)
    A = len(env.actions)
    policy = np.zeros(len(s2i), dtype=int)
    for s, i in s2i.items():
        if s == env.goal:
            policy[i] = 0
            continue
        q_values = []
        for a in range(A):
            ns, r, _ = env.step(s, a)
            q_values.append(r + env.gamma * V[s2i[ns]])
        policy[i] = int(np.argmax(q_values))
    return policy


def arrows_for_actions(actions):
    arrow_map = {0: "→", 1: "↓", 2: "←", 3: "↑"}
    return [arrow_map[int(a)] for a in actions]


def value_iteration(env: Gridworld5x5, theta: float = 1e-6, in_place: bool = False):
    s2i = idx_map(env)
    N = len(s2i)
    A = len(env.actions)
    V = np.zeros(N, dtype=float)
    iters = 0

    while True:
        delta = 0.0
        if in_place:
            for s, i in s2i.items():
                if s == env.goal:
                    continue
                q_values = []
                for a in range(A):
                    ns, r, _ = env.step(s, a)
                    q_values.append(r + env.gamma * V[s2i[ns]])
                new_v = max(q_values)
                delta = max(delta, abs(new_v - V[i]))
                V[i] = new_v
        else:
            V_new = V.copy()
            for s, i in s2i.items():
                if s == env.goal:
                    continue
                q_values = []
                for a in range(A):
                    ns, r, _ = env.step(s, a)
                    q_values.append(r + env.gamma * V[s2i[ns]])
                V_new[i] = max(q_values)
                delta = max(delta, abs(V_new[i] - V[i]))
            V = V_new

        iters += 1
        if delta < theta:
            break

    policy = greedy_policy_from_V(env, V)
    return V, policy, iters


def generate_episode(env: Gridworld5x5, start=None, behavior_probs=None, max_steps: int = 500):
    s2i = idx_map(env)
    A = len(env.actions)
    if behavior_probs is None:
        behavior_probs = [1.0 / A] * A

    if start is None:
        candidates = [s for s in env.states() if s != env.goal]
        state = random.choice(candidates)
    else:
        state = start

    trajectory = []
    for t in range(max_steps):
        a = np.random.choice(np.arange(A), p=np.array(behavior_probs))
        next_state, reward, done = env.step(state, a)
        trajectory.append((state, a, reward))
        if done:
            break
        state = next_state
    return trajectory


def off_policy_mc_importance_sampling(env: Gridworld5x5, episodes: int = 10000, gamma: float = 0.9, behavior_probs=None):
    s2i = idx_map(env)
    A = len(env.actions)
    if behavior_probs is None:
        behavior_probs = [1.0 / A] * A  # uniform random

    V_est = np.zeros(len(s2i), dtype=float)
    C = np.zeros(len(s2i), dtype=float)  # cumulative weights

    for ep in range(episodes):
        target_pi = greedy_policy_from_V(env, V_est)
        traj = generate_episode(env, behavior_probs=behavior_probs, max_steps=200)
        G = 0.0
        W = 1.0

        for t in reversed(range(len(traj))):
            s, a, r = traj[t]
            G = gamma * G + r
            idx = s2i[s]
            if a != target_pi[idx]:
                W = 0.0
            else:
                W = W * (1.0 / behavior_probs[a])
            if W == 0.0:
                break
            C[idx] += W
            V_est[idx] += (W / C[idx]) * (G - V_est[idx])

    return V_est, target_pi


def main():
    env = Gridworld5x5()

    # Problem 3
    V_std, pi_std, it_std = value_iteration(env, theta=1e-8, in_place=False)
    V_inp, pi_inp, it_inp = value_iteration(env, theta=1e-8, in_place=True)

    # Save grids
    s2i = idx_map(env)
    V_std_grid = np.array([V_std[s2i[(r, c)]] for r in range(env.nrows) for c in range(env.ncols)]).reshape(env.nrows, env.ncols)
    V_inp_grid = np.array([V_inp[s2i[(r, c)]] for r in range(env.nrows) for c in range(env.ncols)]).reshape(env.nrows, env.ncols)

    pi_std_arrows = np.array(arrows_for_actions(list(pi_std))).reshape(env.nrows, env.ncols)
    pi_inp_arrows = np.array(arrows_for_actions(list(pi_inp))).reshape(env.nrows, env.ncols)

    pd.DataFrame(V_std_grid).to_csv("V_star_standard.csv", index=False)
    pd.DataFrame(V_inp_grid).to_csv("V_star_inplace.csv", index=False)
    pd.DataFrame(pi_std_arrows).to_csv("policy_star_standard.csv", index=False)
    pd.DataFrame(pi_inp_arrows).to_csv("policy_star_inplace.csv", index=False)

    # Problem 4
    V_mc, pi_mc = off_policy_mc_importance_sampling(env, episodes=8000, gamma=env.gamma)
    V_mc_grid = np.array([V_mc[s2i[(r, c)]] for r in range(env.nrows) for c in range(env.ncols)]).reshape(env.nrows, env.ncols)
    pi_mc_arrows = np.array(arrows_for_actions(list(pi_mc))).reshape(env.nrows, env.ncols)

    pd.DataFrame(V_mc_grid).to_csv("V_mc_offpolicy.csv", index=False)
    pd.DataFrame(pi_mc_arrows).to_csv("policy_mc_offpolicy.csv", index=False)

    print("Saved: V_star_standard.csv, V_star_inplace.csv, policy_star_standard.csv, policy_star_inplace.csv, V_mc_offpolicy.csv, policy_mc_offpolicy.csv")


if __name__ == "__main__":
    main()
