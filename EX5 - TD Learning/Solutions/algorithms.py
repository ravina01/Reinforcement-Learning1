import gym
from typing import Optional, Sequence, Callable, List, Tuple
from collections import defaultdict
import numpy as np


def argmax(arr: Sequence[float]) -> int:
    # returns the index of the item with the highest value, breaking ties randomly.

    return np.random.choice(np.flatnonzero(arr == arr.max()))


# epsilon greedy action selection
def epsilon_greedy(Q, state, epsilon) -> int:
    # Given a Q function and a state, returns an action selected by epsilon-greedy exploration.

    num_actions = len(Q[state])
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    else:
        return argmax(Q[state])


def update_q_table(state, action, reward, next_state, next_action, gamma, step_size, Q):
    Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])


# ON policy TD Control
def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    completed_epi = []
    ep = 0
    state = env.reset()
    print("env.action_space = ", env.action_space)
    print("env = ", env)
    Q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    action = epsilon_greedy(Q_values, state, epsilon)
    print("action = ", action)
    for step in range(num_steps):
        next_state, reward, done, _ = env.step(action)
        # print("done = ", done)
        next_action = epsilon_greedy(Q_values, next_state, epsilon)
        if done:
            ep += 1
            state = env.reset()
            action = epsilon_greedy(Q_values, state, epsilon)
            # print("if done")
        else:
            Q_values[state][action] += step_size * (
                    reward + gamma * Q_values[next_state][next_action] - Q_values[state][action])
            state = next_state
            action = next_action
            # print("ep = ", ep)
        completed_epi.append(ep)
        # print("Q_values = ", Q_values)
    return Q_values, completed_epi


def nstep_sarsa(
        env: gym.Env,
        num_steps: int,
        gamma: float,
        epsilon: float,
        step_size: float,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    n = 4
    Q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    episode = 0
    episodes_completed = []

    tau = 0
    t_episode = 0

    state = env.reset()
    action = epsilon_greedy(Q_values, state, epsilon)

    # Initialize lists to store previous actions, rewards, and states
    previous_actions = [action]
    previous_rewards = []
    previous_states = [state]

    for step in range(num_steps):
        next_state, reward, done, _ = env.step(action)
        next_action = epsilon_greedy(Q_values, next_state, epsilon)

        # Store previous action, reward, and state

        previous_states.append(next_state)
        previous_actions.append(next_action)
        previous_rewards.append(reward)

        if done:
            episode += 1
            t_episode = 0

            # Clear previous action, reward, and state lists
            previous_actions = []
            previous_rewards = []
            previous_states = []

            state = env.reset()
            action = epsilon_greedy(Q_values, state, epsilon)

            # Reinitialize the lists
            previous_actions.append(action)
            previous_states.append(state)
        else:
            tau = t_episode - n + 1

            if tau >= 0:
                cumulative_reward = np.sum(
                    [gamma ** (i - tau - 1) * previous_rewards[i] for i in range(tau + 1, min(tau + n, num_steps))])

                if tau + n < num_steps:
                    cumulative_reward += gamma ** n * Q_values[previous_states[tau + n]][previous_actions[tau + n]]

                tau_s, tau_a = previous_states[tau], previous_actions[tau]
                Q_values[tau_s][tau_a] += step_size * (cumulative_reward - Q_values[tau_s][tau_a])

            t_episode += 1
            state = next_state
            action = next_action

        episodes_completed.append(episode)

    return Q_values, episodes_completed


def expectation_values(Q, state, epsilon) -> float:
    """
    Args:
         Q-values
         state
        epsilon for epsilon greedy
    """
    num_actions = len(Q[state])
    best_action = argmax(Q[state])

    # Use list comprehension for calculating probabilities
    probs = [epsilon / num_actions] * num_actions
    probs[best_action] += 1 - epsilon

    # Use numpy for more concise code
    exp = np.dot(probs, Q[state])

    return exp


def exp_sarsa(
        env: gym.Env,
        num_steps: int,
        gamma: float,
        epsilon: float,
        step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q_values = defaultdict(lambda: np.zeros(env.action_space.n))
    episodes_completed = []
    ep = 0

    state = env.reset()

    for step in range(num_steps):
        action = epsilon_greedy(Q_values, state, epsilon)
        next_state, reward, done, _ = env.step(action)

        if done:
            ep += 1
            state = env.reset()
        else:
            next_action = epsilon_greedy(Q_values, next_state, epsilon)
            target = reward + gamma * expectation_values(Q_values, next_state, epsilon)
            Q_values[state][action] += step_size * (target - Q_values[state][action])
            state = next_state

        episodes_completed.append(ep)

    return Q_values, episodes_completed


def q_learning(
        env: gym.Env,
        num_steps: int,
        gamma: float,
        epsilon: float,
        step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    episodes_completed = []
    ep = 0
    state = env.reset()
    Q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    for step in range(num_steps):
        action = epsilon_greedy(Q_values, state, epsilon)
        next_state, reward, done, _ = env.step(action)

        if done:
            ep += 1
            state = env.reset()
        else:
            best_action = argmax(Q_values[next_state])
            Q_values[state][action] += step_size * (
                    reward + gamma * Q_values[next_state][best_action] - Q_values[state][action])
            state = next_state

        episodes_completed.append(ep)

    return Q_values, episodes_completed


def mc_prediction(
    episodes,
    gamma: float,
) -> defaultdict:
    """On-policy Monte Carlo policy evaluation. First visits will be used.

    Args:
        gamma (float): Discount factor of MDP

    Returns:
        V (defaultdict): The values for each state. V[state] = value.
    """
    # From EX4
    estimated_state_values = defaultdict(float)
    N = defaultdict(int)

    for episode in episodes:
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            # Update V and N here according to first visit MC
            reward = episode[t][2]
            G = gamma * G + reward
            current_state = episode[t][0]
            previous_states = [i[0] for i in episode[0:t]]
            if current_state not in previous_states:
                N[current_state] += 1
                estimated_state_values[current_state] += 1/N[current_state] * (G-estimated_state_values[current_state])
    return estimated_state_values


def td_prediction(env: gym.Env, gamma: float, episodes, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    estimated_state_values = defaultdict(float)
    step_size = 0.5
    for episode in episodes:
        T = len(episode)
        for t in range(T):
            tau = t - n + 1

            if t + 1 < T:
                T = t + 1

            if tau >= 0:
                cumulative_return = sum([gamma ** (i - tau) * episode[i][2] for i in range(tau, min(tau + n, T))])

                if tau + n < T:
                    state_tpn = episode[tau + n][0]
                    cumulative_return += gamma ** n * estimated_state_values[state_tpn]

                state_tau = episode[tau][0]
                estimated_state_values[state_tau] += step_size * (cumulative_return - estimated_state_values[state_tau])

    return estimated_state_values


def learning_targets(
        V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO
    targets = np.zeros(len(episodes))

    num_episodes = len(episodes)

    for i in range(num_episodes):
        episode = episodes[i]
        T = len(episode)
        cumulative_reward = 0.0

        for t in range(T):
            if n is None or t + n < T:
                # Monte Carlo target
                if n is None:
                    cumulative_reward += gamma ** t * episode[t][2]
                    # TD(n) target
                else:
                    cumulative_reward += gamma ** t * episode[t][2]

                if n is not None:
                    next_state = episode[t + n][0]
                    cumulative_reward += gamma ** n * V[next_state]

        targets[i] = cumulative_reward

    return targets


def generate_episode(env: gym.Env, Q: defaultdict, epsilon: float, num_steps: int) -> List[Tuple[int, int, float]]:
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """

    episode = []
    state = env.reset()

    for i in range(num_steps):
        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))

        if done:
            return episode, done

        state = next_state

    return episode, done


def mc_control_epsilon_soft(
        env: gym.Env, num_steps: int, gamma: float, epsilon: float
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
    episodes_completed = []
    steps = 0
    ep_count = 0
    # policy = create_epsilon_policy(Q, epsilon)
    #
    # returns = np.zeros(num_episodes)
    # for iter in trange(num_episodes, desc="Episode", leave=False):
    #     # TODO Q4
    #     # For each episode calculate the return
    #     # Update Q
    #     # Note there is no need to update the policy here directly.
    #     # By updating Q, the policy will automatically be updated.
    #     episode = generate_episode(env, policy, False)
    #     G = 0
    #     updated_state = []
    #
    #     for t in range(len(episode) - 1, -1, -1):
    #
    #         reward = episode[t][2]
    #         G = gamma * G + reward
    #
    #         # Update V and N here according to first visit MC
    #         # if s not in np.array(episode,dtype=object).reshape(-1,3)[:t,0]:
    #         state = episode[t][0]
    #         action = episode[t][1]
    #         if state not in updated_state:
    #             N[state[0]][action] = N[state[0]][action] + 1
    #             Q[state[0]][action] = Q[state[0]][action] + (G - Q[state[0]][action]) / N[state[0]][action]
    #             updated_state.append(state)
    #
    #     returns[iter] = G
    #
    # return returns
    while steps < num_steps:
        G = 0.0
        episode, done = generate_episode(env, Q, epsilon, num_steps)
        ep = len(episode)
        if done:
            ep_count += 1
        if steps + ep > num_steps:
            episodes_completed[steps:num_steps + 1] = [ep_count] * (num_steps - steps)
            break
        else:
            episodes_completed.extend([ep_count] * ep)
        for t in range(len(episode) - 1, -1, -1):
            # For each episode calculate the return
            # Update Q and N
            reward = episode[t][2]
            G = gamma * G + reward
            current_state = episode[t][0]
            current_action = episode[t][1]
            previous_state_action = [(i[0], i[1]) for i in episode[0:t]]
            if (current_state, current_action) not in previous_state_action:
                N[current_state][current_action] += 1.0
                Q[current_state][current_action] += (1 / N[current_state][current_action]) * (
                            G - Q[current_state][current_action])
        steps += ep
    return Q, episodes_completed
