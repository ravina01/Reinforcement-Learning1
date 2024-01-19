import matplotlib.pyplot as plt
import gym
import numpy as np
from env import register_env
import tqdm
from algorithms import sarsa, q_learning, exp_sarsa, nstep_sarsa, mc_control_epsilon_soft, td_prediction, mc_prediction, \
    generate_episode, learning_targets


def graph_plots_td(sarsa, ql, exp_sarsa, nsarsa, mc, env_name):
    # Plot the mean, error bars, and labels for each algorithm
    plt.xlabel('Timesteps')
    plt.ylabel('Episodes')

    labels = ['SARSA', 'Q-learning', 'Expected SARSA', 'N-step SARSA', 'e-soft Monte Carlo']
    colors = ['b', 'r', 'g', 'y', 'k']

    for i, algorithm in enumerate([sarsa, ql, exp_sarsa, nsarsa, mc]):
        print("i =", i)
        mean = np.mean(algorithm, axis=0)
        std = np.std(algorithm, axis=0)
        conf = 1.96 * std / np.sqrt(10)
        plt.plot(mean, label=labels[i], color=colors[i])
        plt.fill_between(np.arange(0, 8000), mean - conf, mean + conf, alpha=0.2, color=colors[i])

    # Add a title to the plot
    plt.title(env_name + " Environment")

    # Add a legend
    plt.legend()

    # Save the plot to a file
    plt.savefig('outputs/' + env_name + '.png')

    # Close the plot window
    plt.close()


def Q4_B_windy_grid_world():
    Trials = 10
    register_env()
    windyGridWorld_env = gym.make('WindyGridWorld-v0')
    windyGridWorld_env.seed(0)

    sarsa_episodes = []
    ql_episodes = []
    exp_sarsa_episodes = []
    n_sarsa_episodes = []
    mc_episodes = []

    for trial in tqdm.trange(Trials, desc='Trials', leave=False):
        sarsa_episodes.append(
            sarsa(windyGridWorld_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        ql_episodes.append(
            q_learning(windyGridWorld_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        exp_sarsa_episodes.append(
            exp_sarsa(windyGridWorld_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        n_sarsa_episodes.append(nstep_sarsa(windyGridWorld_env, num_steps=8000, gamma=1, epsilon=0.1,
                                            step_size=0.5)[1])
        mc_episodes.append(mc_control_epsilon_soft(windyGridWorld_env, num_steps=8000, gamma=1, epsilon=0.1)[1])

    graph_plots_td(sarsa_episodes, ql_episodes, exp_sarsa_episodes, n_sarsa_episodes, mc_episodes,
                   'Windy Grid World')


def Q4_C_kings_grid_world():
    Trials = 10
    register_env()
    windyGridWorldkings_env = gym.make('WindyGridWorldKings-v0')
    windyGridWorldkings_env.seed(0)

    sarsa_episodes = []
    ql_episodes = []
    exp_sarsa_episodes = []
    n_sarsa_episodes = []
    mc_episodes = []

    for trial in tqdm.trange(Trials, desc='Trials', leave=False):
        sarsa_episodes.append(
            sarsa(windyGridWorldkings_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        ql_episodes.append(
            q_learning(windyGridWorldkings_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        exp_sarsa_episodes.append(
            exp_sarsa(windyGridWorldkings_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        n_sarsa_episodes.append(nstep_sarsa(windyGridWorldkings_env, num_steps=8000, gamma=1, epsilon=0.1,
                                            step_size=0.5)[1])
        mc_episodes.append(mc_control_epsilon_soft(windyGridWorldkings_env, num_steps=8000, gamma=1, epsilon=0.1)[1])
        # episodes_completed_mc.append(
        #     mc_control_epsilon_soft(windyGridWorld_env, num_steps=8000, gamma=1, epsilon=0.1)[1])
    # print(episodes_completed_sarsa)
    graph_plots_td(sarsa_episodes, ql_episodes, exp_sarsa_episodes, n_sarsa_episodes, mc_episodes,
                   'Kings move Windy Grid World')


def Q4_D_stoch_kings_grid_world():
    Trials = 10
    register_env()
    stoch_kings_env = gym.make('WindyGridWorldKings-v1')
    stoch_kings_env.seed(0)

    sarsa_episodes = []
    ql_episodes = []
    exp_sarsa_episodes = []
    n_sarsa_episodes = []
    mc_episodes = []

    for trial in tqdm.trange(Trials, desc='Trials', leave=False):
        sarsa_episodes.append(
            sarsa(stoch_kings_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        ql_episodes.append(
            q_learning(stoch_kings_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        exp_sarsa_episodes.append(
            exp_sarsa(stoch_kings_env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)[1])
        n_sarsa_episodes.append(nstep_sarsa(stoch_kings_env, num_steps=8000, gamma=1, epsilon=0.1,
                                            step_size=0.5)[1])
        mc_episodes.append(mc_control_epsilon_soft(stoch_kings_env, num_steps=8000, gamma=1, epsilon=0.1)[1])
        # episodes_completed_mc.append(
        #     mc_control_epsilon_soft(windyGridWorld_env, num_steps=8000, gamma=1, epsilon=0.1)[1])
    # print(episodes_completed_sarsa)
    graph_plots_td(sarsa_episodes, ql_episodes, exp_sarsa_episodes, n_sarsa_episodes, mc_episodes,
                   'Kings move Stochastic Windy Grid World')


def plot_histogram(targets, episodes, label, filename):
    plt.figure(figsize=(10, 10))
    plt.hist(targets, label=f'Episodes={episodes} {label}')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.title(f'Episodes={episodes} {label}')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.legend()
    plt.savefig(filename)
    plt.close()


def Q5_bias_variance_trade_off():
    # Constants
    gamma = 1
    step_size = 0.5
    gamma = 0.1
    num_steps = 8000

    register_env()
    env = gym.make('WindyGridWorld-v0')
    Q_values, _ = q_learning(env, num_steps=8000, gamma=1, epsilon=0.1, step_size=0.5)

    training_episode = [1, 10, 50]
    nsteps_td = [1, 4]

    estimated_state_values_td = []
    estimated_state_values_mc = []

    for iter in training_episode:

        # episodes = [generate_episode(env, Q_values, gamma, num_steps)[0] for i in range(iter)]
        episodes = []
        for i in range(iter):
            episodes.append(generate_episode(env, Q_values, gamma, num_steps)[0])
        # TD(1) and TD(4) predictions
        estimated_state_values_td_1 = td_prediction(env, gamma, episodes, n=1)
        estimated_state_values_td_4 = td_prediction(env, gamma, episodes, n=4)

        # Monte-Carlo prediction
        estimated_state_values_mc_1 = mc_prediction(episodes, gamma)

        estimated_state_values_td.extend([estimated_state_values_td_1, estimated_state_values_td_4])
        estimated_state_values_mc.append(estimated_state_values_mc_1)

        # Unpack values for easier access
        print("estimated_state_values_td = ", estimated_state_values_td)

        # Generate evaluation episodes
        episodes_eval = [generate_episode(env, Q_values, gamma, num_steps)[0] for j in range(100)]

        # plot_configs = [
        #     ( learning_targets(estimated_state_values_td[0], gamma, episodes_eval, n=None), 1, 'TD(0)', 'outputs/Q5_1_td.png'),
        #     ( learning_targets(estimated_state_values_td[2], gamma, episodes_eval, n=None), 10, 'TD(0)', 'outputs/Q5_10_td.png'),
        #     ( learning_targets(estimated_state_values_td[4], gamma, episodes_eval, n=None), 50, 'TD(0)', 'outputs/Q5_50_td.png')
        # ]
        #
        # for targets, episodes, label, filename in plot_configs:
        #     plot_histogram(targets, episodes, label, filename)


        # plot learning targets as histogram for TD(0)
        plt.figure(figsize=(10, 10))
        plt.hist(learning_targets(estimated_state_values_td[0], gamma, episodes_eval, n=None), label="Episodes=1 TD(0)")
        plt.xlabel('Target Value')
        plt.title("Episodes=1 TD(0)")
        plt.ylabel('Frequency')
        ax = plt.gca()
        ax.invert_xaxis()
        plt.savefig('outputs/Q5_1_td.png')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.hist(learning_targets(estimated_state_values_mc[0], gamma, episodes_eval, n=None), label="Episodes=1 MC(0)")
        plt.xlabel('Target Value')
        plt.title("Episodes=1 MC(0)")
        plt.ylabel('Frequency')
        ax = plt.gca()
        ax.invert_xaxis()
        plt.savefig('outputs/Q5_1_mc.png')
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.hist(learning_targets(estimated_state_values_td[1], gamma, episodes_eval, n=None),
                 label="Episodes=1 TD(4)")
        plt.xlabel('Target Value')
        plt.title("Episodes=1 TD(4)")
        plt.ylabel('Frequency')
        ax = plt.gca()
        ax.invert_xaxis()
        plt.savefig('outputs/Q5_1_td4.png')
        plt.close()




def main():
    Q4_B_windy_grid_world()
    # Q4_C_kings_grid_world()
    # Q4_D_stoch_kings_grid_world()
    # Q5_bias_variance_trade_off()


if __name__ == "__main__":
    main()
