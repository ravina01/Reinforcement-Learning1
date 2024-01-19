from env import BanditEnv
from agent import EpsilonGreedy, UCB
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    # TODO

    cumulative_reward = []
    print("done")
    for iter in range(k):
        step_reward = []
        for iter1 in range(num_samples):
            step_reward.append(env.step(iter))
        cumulative_reward.append(step_reward)

    plt.xlabel('Action')
    plt.ylabel('Reward Distribution')
    plt.title('10-Armed Testbed')

    plt.violinplot(cumulative_reward, showmeans=1)
    #median reward is represented by a thick line inside the violin.
    # wider the violin, the more dispersed the rewards are.
    plt.show()

    pass


def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment = 10
        trials (int): number of trials = 2000
        steps (int): total number of steps for each trial = 1000
    """
    # TODO initialize env and agents here
    env = BanditEnv(k=k)
    env.reset()

    agent_0 = EpsilonGreedy(k=10, init=0, epsilon=0)
    agent_1 = EpsilonGreedy(k=10, init=0, epsilon=0.1)
    agent_2 = EpsilonGreedy(k=10, init=0, epsilon=0.01)

    agents = [agent_0, agent_1, agent_2]

    rewards = []
    optimal_rewards = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        reward_trail = []
        optimal_reward_trail = []
        for agent in agents:
            agent.reset()

            reward_agent = []
            reward_optimal_agent = []
            # TODO For each trial, perform specified number of steps for each type of agent
            for step in range(steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)

                if action == np.argmax(env.means):
                    reward_optimal_agent.append(1)
                else:
                    reward_optimal_agent.append(0)

                #per agent
                reward_agent.append(reward)

            reward_trail.append(reward_agent)
            optimal_reward_trail.append(reward_optimal_agent)
        rewards.append(reward_trail)
        optimal_rewards.append(optimal_reward_trail)

    # Plots
    rewards_avg = np.average(rewards, 0)
    rewards_optimal_avg = np.average(optimal_rewards, 0)
    rewards_std = np.std(rewards, 0)

    #print("rewards_avg = ",rewards_avg)
    #Epsilon = 0, 0.1, 0.01
    reward_epsilon_0, reward_epsilon_1, reward_epsilon_2 = rewards_avg[:3]

    reward_optimal_epsilon_0, reward_optimal_epsilon_1, reward_optimal_epsilon_2 = rewards_optimal_avg[:3]

    confidence_intervals = []
    for i in range(len(rewards_avg)):
        lower = rewards_avg[i] - 1.96 * (rewards_std[i] / np.sqrt(trials))
        higher = rewards_avg[i] + 1.96 * (rewards_std[i] / np.sqrt(trials))
        confidence_intervals.append([lower, higher])

    # Get the confidence intervals for the three epsilon values
    epsi0_lower, epsi0_higher = confidence_intervals[0]
    epsi1_lower, epsi1_higher = confidence_intervals[1]
    epsi2_lower, epsi2_higher = confidence_intervals[2]

    #Upper bound line

    max_reward = np.max(env.means)

    reward_max = []

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title("Epsilon Greedy")

    reward_max.append(max_reward)
    plt.plot(reward_epsilon_0, label='ε = 0')
    plt.plot(reward_epsilon_1, label='ε = 0.1')
    plt.plot(reward_epsilon_2, label='ε = 0.01')

    plt.axhline(y=np.mean(reward_max), linestyle='--', label='Best possible Average Performance')

    plt.fill_between(np.arange(steps), epsi0_lower, epsi0_higher, alpha=0.1)
    plt.fill_between(np.arange(steps), epsi1_lower, epsi1_higher, alpha=0.1)
    plt.fill_between(np.arange(steps), epsi2_lower, epsi2_higher, alpha=0.1)

    plt.legend()

    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action')
    plt.title("Epsilon Greedy-optimal actions taken")

    plt.plot(reward_optimal_epsilon_0, label='ε = 0')
    plt.plot(reward_optimal_epsilon_1, label='ε = 0.1')
    plt.plot(reward_optimal_epsilon_2, label='ε = 0.01')

    plt.legend()
    plt.show()


def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = None
    agents = []

    env = BanditEnv(k=k)

    agent_0 = EpsilonGreedy(k=10, init=0, epsilon=0)
    agent_1 = EpsilonGreedy(k=10, init=5, epsilon=0)
    agent_2 = EpsilonGreedy(k=10, init=0, epsilon=0.1)
    agent_3 = EpsilonGreedy(k=10, init=5, epsilon=0.1)
    agent_4 = UCB(k=10, init=0, c=2, step_size=0.1)

    agents = [agent_0, agent_1, agent_2, agent_3, agent_4]

    rewards = []
    optimal_rewards = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        reward_trail = []
        optimal_reward_trail = []
        for agent in agents:
            agent.reset()

            reward_agent = []
            reward_optimal_agent = []
            # TODO For each trial, perform specified number of steps for each type of agent
            for step in range(steps):
                action = agent.choose_action()
                reward = env.step(action)
                agent.update(action, reward)

                if action == np.argmax(env.means):
                    reward_optimal_agent.append(1)
                else:
                    reward_optimal_agent.append(0)

                # per agent
                reward_agent.append(reward)

            reward_trail.append(reward_agent)
            optimal_reward_trail.append(reward_optimal_agent)
        rewards.append(reward_trail)
        optimal_rewards.append(optimal_reward_trail)

    # Plots
    rewards_avg = np.average(rewards, 0)
    rewards_optimal_avg = np.average(optimal_rewards, 0)
    rewards_std = np.std(rewards, 0)

    #print("rewards_avg = ", rewards_avg)
    # Epsilon = 0, 0.1, 0.01
    reward_epsilon_0, reward_epsilon_1, reward_epsilon_2, reward_epsilon_3, reward_epsilon_4 = rewards_avg[:5]

    reward_optimal_epsilon_0, reward_optimal_epsilon_1, reward_optimal_epsilon_2, reward_optimal_epsilon_3, reward_optimal_epsilon_4 = rewards_optimal_avg[:5]

    confidence_intervals = []
    for i in range(len(rewards_avg)):
        lower = rewards_avg[i] - 1.96 * (rewards_std[i] / np.sqrt(trials))
        higher = rewards_avg[i] + 1.96 * (rewards_std[i] / np.sqrt(trials))
        confidence_intervals.append([lower, higher])

    # Get the confidence intervals for the three epsilon values
    epsi0_lower, epsi0_higher = confidence_intervals[0]
    epsi1_lower, epsi1_higher = confidence_intervals[1]
    epsi2_lower, epsi2_higher = confidence_intervals[2]
    epsi3_lower, epsi3_higher = confidence_intervals[3]
    epsi4_lower, epsi4_higher = confidence_intervals[4]

    # Upper bound line
    max_reward = np.max(env.means)

    reward_max = []
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title("Epsilon Greedy")
    reward_max.append(max_reward)

    plt.plot(reward_epsilon_0, label='Q1 = 0, ε = 0')
    plt.plot(reward_epsilon_1, label='Q1 = 5, ε = 0')
    plt.plot(reward_epsilon_2, label='Q1 = 0, ε = 0.1')
    plt.plot(reward_epsilon_3, label='Q1 = 5, ε = 0.1')
    plt.plot(reward_epsilon_4, label='UCB, c=2')

    plt.axhline(y=np.mean(reward_max), linestyle='--', label='Best possible Average Performance')

    plt.fill_between(np.arange(steps), epsi0_lower, epsi0_higher, alpha=0.1)
    plt.fill_between(np.arange(steps), epsi1_lower, epsi1_higher, alpha=0.1)
    plt.fill_between(np.arange(steps), epsi2_lower, epsi2_higher, alpha=0.1)
    plt.fill_between(np.arange(steps), epsi3_lower, epsi3_higher, alpha=0.1)
    plt.fill_between(np.arange(steps), epsi4_lower, epsi4_higher, alpha=0.1)

    plt.legend()

    plt.figure()
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action')
    plt.title("Epsilon Greedy-optimal actions taken")

    plt.plot(reward_optimal_epsilon_0, label='Q1 = 0, ε = 0')
    plt.plot(reward_optimal_epsilon_1, label='Q1 = 5, ε = 0')
    plt.plot(reward_optimal_epsilon_2, label='Q1 = 0, ε = 0.1')
    plt.plot(reward_optimal_epsilon_3, label='Q1 = 5, ε = 0.1')
    plt.plot(reward_optimal_epsilon_4, label='UCB, c=2')

    plt.legend()
    plt.show()


def main():
    # TODO run code for all questions
    #Uncomment below questions in order to run

    #q4(k=10, num_samples=2000)
    #q6(k=10, trials=2000, steps=1000)
    q7(k=10, trials=2000, steps=1000)
    pass


if __name__ == "__main__":
    main()
