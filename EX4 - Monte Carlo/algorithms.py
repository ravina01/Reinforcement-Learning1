import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_blackjack_policy, create_epsilon_policy


def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    # print("env = ", env)
    episode = []
    state = env.reset()
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, *_ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode


def on_policy_mc_evaluation(
    env: gym.Env,
    policy: Callable,
    num_episodes: int,
    gamma: float,
) -> defaultdict:
    """On-policy Monte Carlo policy evaluation. First visits will be used.

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP

    Returns:
        V (defaultdict): The values for each state. V[state] = value.
    """
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in trange(num_episodes, desc="Episode"):
        episode = generate_episode(env, policy)
        G = 0.0
        S = []
        for t in range(len(episode) - 1, -1, -1):
            # Update V and N here according to first visit MC
            reward = episode[t][2]
            G = gamma * G + reward
            current_state = episode[t][0]

            # previous_states = [i[0] for i in episode[0:t]]
            # print("current_state = ", current_state)
            # print("reward = ", reward)
            if current_state not in S:
                N[current_state] = N[current_state] + 1
                V[current_state] += (G-V[current_state]) / N[current_state]
                S.append(current_state)
    return V


def on_policy_mc_control_es(
    env: gym.Env, num_episodes: int, gamma: float
) -> Tuple[defaultdict, Callable]:
    """On-policy Monte Carlo control with exploring starts for Blackjack

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
    """
    # We use defaultdicts here for both Q and N for convenience. The states will be the keys and the values will be numpy arrays with length = num actions
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    # If the state was seen, use the greedy action using Q values.
    # Else, default to the original policy of sticking to 20 or 21.
    policy = create_blackjack_policy(Q)

    for _ in trange(num_episodes, desc="Episode"):
        # TODO Q3b
        episode = generate_episode(env, policy, 1)
        G = 0
        S = []
        for t in range(len(episode) - 1, -1, -1):
            s = episode[t][0]
            a = episode[t][1]
            r = episode[t][2]
            G = gamma * G + r

            if s not in S:
                N[s[0]][a] = N[s[0]][a] + 1.0
                Q[s[0]][a] = Q[s[0]][a] + (G - Q[s[0]][a]) / N[s[0]][a]
                S.append(s)


        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.

    return Q, policy


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)

    returns = np.zeros(num_episodes)
    for iter in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env, policy, False)
        G = 0
        updated_state = []

        for t in range(len(episode) - 1, -1, -1):

            reward = episode[t][2]
            G = gamma * G + reward

            # Update V and N here according to first visit MC
            # if s not in np.array(episode,dtype=object).reshape(-1,3)[:t,0]:
            state = episode[t][0]
            action = episode[t][1]
            if state not in updated_state:
                N[state[0]][action] = N[state[0]][action] + 1
                Q[state[0]][action] = Q[state[0]][action] + (G - Q[state[0]][action]) / N[state[0]][action]
                updated_state.append(state)

        returns[iter] = G

    return returns
