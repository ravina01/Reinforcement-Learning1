from collections import defaultdict
import gym
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
import policy
from algorithms import on_policy_mc_evaluation, on_policy_mc_control_es, on_policy_mc_control_epsilon_soft
import env
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot


def plot_backjack(V, ax1, ax2):
    player = np.arange(12, 21 + 1)
    dealer = np.arange(1, 10 + 1)
    ace = np.array([False, True])
    state_values = np.zeros((len(player), len(dealer), len(ace)))

    for i, player_ in enumerate(player):
        for j, dealer_ in enumerate(dealer):
            for k, ace_ in enumerate(ace):
                state_values[i, j, k] = V[player_, dealer_, ace_]

    X, Y = np.meshgrid(dealer, player)

    ax1.plot_wireframe(X, Y, state_values[:, :, 0], color='black')
    ax2.plot_wireframe(X, Y, state_values[:, :, 1], color='black')

    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('Player sum')
        ax.set_xlabel('Dealer showing')


def Q3_a():
    env_blackjack = gym.make('Blackjack-v1')
    V_10k = on_policy_mc_evaluation(env_blackjack, policy=policy.default_blackjack_policy, num_episodes=10000, gamma=1)

    V_500k = on_policy_mc_evaluation(env_blackjack, policy=policy.default_blackjack_policy, num_episodes=500000,
                                     gamma=1)

    fig, axes = pyplot.subplots(nrows=4, figsize=(15, 20), subplot_kw={'projection': '3d'})
    axes[0].set_title('No usable ace after 10000 episodes')
    axes[1].set_title('Usable ace after 10000 episodes')
    axes[2].set_title('No usable ace after 500000 episodes')
    axes[3].set_title('Usable ace after 500000 episodes')
    plot_backjack(V_10k, axes[0], axes[1])
    plot_backjack(V_500k, axes[2], axes[3])
    plt.show()


def Q3_b():
    Q, policy = on_policy_mc_control_es(gym.make("Blackjack-v1"), 5000000, 1)

    pi_no_ace = np.zeros([10, 10])
    pi_has_ace = np.zeros([10, 10])
    v_no_ace = np.zeros([10, 10])
    v_has_ace = np.zeros([10, 10])

    for i in range(12, 22):
        for j in range(1, 11):
            pi_no_ace[i - 12, j - 1] = policy((i, j, False))
            pi_has_ace[i - 12, j - 1] = policy((i, j, True))
            v_no_ace[i - 12, j - 1] = Q[(i, j, False)][policy((i, j, False))]
            v_has_ace[i - 12, j - 1] = Q[(i, j, True)][policy((i, j, True))]

    fig = plt.figure(figsize=(10, 10))
    ax_pi_no_ace = fig.add_subplot(2, 2, 3)
    ax_pi_has_ace = fig.add_subplot(2, 2, 1, title='Optimal Policy')
    v_5mil_no_ace = fig.add_subplot(2, 2, 4, projection='3d')
    v_5mil_has_ace = fig.add_subplot(2, 2, 2, projection='3d', title='Optimal Value')
    ax_pi_has_ace.text(8, 20, 'Stick')
    ax_pi_has_ace.text(8, 13, 'Hit')

    ax_pi_no_ace.set_xlabel('Dealer Showing')
    ax_pi_no_ace.set_ylabel('Player Sum')
    v_5mil_no_ace.set_xlabel('Dealer Showing')
    v_5mil_no_ace.set_ylabel('Player Sum')

    fig.text(0.1, 0.75, 'Usable\n  Ace', fontsize=12)
    fig.text(0.1, 0.25, 'No\nUsable\n  Ace', fontsize=12)

    ax_pi_has_ace.imshow(pi_has_ace, origin='lower', vmin=-1, vmax=1, cmap=plt.cm.coolwarm, alpha=0.3,
                         extent=[0.5, 10.5, 11.5, 21.5],
                         interpolation='none')
    ax_pi_has_ace.set_xticks(np.arange(1, 11, 1))
    ax_pi_has_ace.set_yticks(np.arange(12, 22, 1))

    ax_pi_no_ace.imshow(pi_no_ace, origin='lower', vmin=-1, vmax=1, cmap=plt.cm.coolwarm, alpha=0.3,
                        extent=[0.5, 10.5, 11.5, 21.5],
                        interpolation='none')
    ax_pi_no_ace.set_xticks(np.arange(1, 11, 1))
    ax_pi_no_ace.set_yticks(np.arange(12, 22, 1))

    dealer_showing = list(range(1, 11))
    player_sum = list(range(12, 22))
    x, y = np.meshgrid(dealer_showing, player_sum)
    v_5mil_has_ace.plot_wireframe(x, y, v_has_ace)

    dealer_showing = list(range(1, 11))
    player_sum = list(range(12, 22))
    x, y = np.meshgrid(dealer_showing, player_sum)
    v_5mil_no_ace.plot_wireframe(x, y, v_no_ace)

    plt.savefig('fig5_2.png')
    plt.show()
    plt.close()


def Q4_a():
    env.register_env()
    env_fourRooms = gym.make('FourRooms-v0')
    returns = on_policy_mc_control_epsilon_soft(env_fourRooms, 1000, 0.99, 0.1)
    print(returns)


def Q4_b(trials: int, num_episodes: int):
    env.register_env()
    env_fourRooms = gym.make('FourRooms-v0')

    rewards = {}
    for epsilon in [0, 0.1, 0.01]:
        rewards[epsilon] = on_policy_mc_control_epsilon_soft(env_fourRooms, num_episodes, 0.99, epsilon)

    avg_rewards = {
        epsilon: np.mean(rewards[epsilon])
        for epsilon in rewards
    }

    std_rewards = {
        epsilon: np.std(rewards[epsilon])
        for epsilon in rewards
    }

    lower_bounds = {
        epsilon: avg_rewards[epsilon] - 1.96 * (std_rewards[epsilon] / np.sqrt(trials))
        for epsilon in rewards
    }

    higher_bounds = {
        epsilon: avg_rewards[epsilon] + 1.96 * (std_rewards[epsilon] / np.sqrt(trials))
        for epsilon in rewards
    }

    fig, ax = plt.subplots()
    for epsilon in rewards:
        ax.plot(avg_rewards[epsilon], label=f'ε = {epsilon}')
        ax.fill_between(
            np.arange(num_episodes),
            lower_bounds[epsilon],
            higher_bounds[epsilon],
            alpha=0.1
        )
    plt.axhline(y=0.99 ** 20, linestyle='--', label='upper bound')
    ax.set_xlabel('Num of episodes')
    ax.set_ylabel('Episode’s discounted return')
    ax.legend()
    plt.show()


def main():
    # Uncomment Q you want to run
    Q3_a()
    # Q3_b()
    # Q4_a()
    # Q4_b(10, 1000)


if __name__ == "__main__":
    main()
