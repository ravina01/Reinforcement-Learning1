"""
Author - Ravina Lad
EX3 - Dynammic Programming
algorithm.py - To run Q 5 and Q 6 -> the gridworld and car rental problem
"""
from env import Action, Gridworld5x5, JacksCarRental
import numpy as np
from env import Gridworld5x5, Action
from collections import defaultdict


class GridWorld:
    def __int__(self):
        None

    def iterative_policy_eval(self) -> list:
        grid_env_obj = Gridworld5x5()
        discount_factor = 0.9
        theta = 0.001  # convergence threshold
        random_policy = 0.25 * np.ones((25, 4), dtype=float)

        value_func = np.zeros(len(grid_env_obj.state_space), dtype=float)
        updated_value_func = np.zeros(len(grid_env_obj.state_space), dtype=float)
        """
        we have --> V(s) = Σ_{a} π(a|s) [R(s, a) + γ Σ_{s'} P(s'|s, a) V(s')]
        """
        flag = True
        while flag:
            delta = 0.0
            for i, state in enumerate(grid_env_obj.state_space):
                v = value_func[i]
                updated_value_func[i] = 0.0

                for j, action in enumerate(Action):
                    updated_value_func[i] += random_policy[i][j] * grid_env_obj.expected_return(value_func, state,
                                                                                                action, discount_factor)

                delta = max(delta, abs(v - updated_value_func[i]))

            value_func = updated_value_func

            if delta < theta:
                return value_func

    def value_iteration(self):
        grid_env_obj = Gridworld5x5()
        discount_factor = 0.9
        theta = 0.001  # convergence threshold
        """
            we have --> V(s) = max_{a} Σ_{s'} P(s'|s, a) [R(s, a) + γ V(s')]
        """
        optimal_value_func = np.zeros(len(grid_env_obj.state_space), dtype=float)
        regular_policy = defaultdict(lambda: [])
        flag = True
        while flag:
            delta = 0.0
            for i, state in enumerate(grid_env_obj.state_space):
                v = optimal_value_func[i]
                action_values = [grid_env_obj.expected_return(optimal_value_func, state, action, discount_factor) for
                                 action in Action]
                optimal_value_func[i] = max(action_values)
                delta = max(delta, abs(v - optimal_value_func[i]))
            if delta < theta:
                break

        for i, state in enumerate(grid_env_obj.state_space):
            action_values = {
                action.name: round(grid_env_obj.expected_return(optimal_value_func, state, action, discount_factor), 2)
                for action in Action
            }
            max_value = max(action_values.values())

            optimal_action = [k for k, v in action_values.items() if v == max_value]
            for action in optimal_action:
                regular_policy[state].append(action)

        return regular_policy, optimal_value_func

    # Policy Improvement
    def policy_iteration(self):
        grid_env_obj = Gridworld5x5()
        discount_factor = 0.9
        theta = 0.001  # convergence threshold
        optimal_policy = 0.25 * np.ones((25, 4), dtype=float)
        flag = True
        # improve the policy until it's stable.
        while flag:
            policy_stable = True
            _, value_func = self.value_iteration()
            for i, state in enumerate(grid_env_obj.state_space):
                action_values = {
                    action.name: round(grid_env_obj.expected_return(value_func, state, action, discount_factor), 2) for
                    action in Action
                }
                max_value = max(action_values.values())

                optimal_action = np.array([
                    v if v == max_value else 0.0
                    for k, v in action_values.items()
                ])
                # print("optimal_action 2 \n", optimal_action)

                # let's calculate new policy for given state.
                probabilities = 1.0 / np.count_nonzero(optimal_action)
                new_action = np.zeros(len(action_values), dtype=float)
                for index, action in enumerate(optimal_action):
                    if action == 0.0:
                        continue
                    else:
                        new_action[index] = probabilities
                if not np.array_equal(optimal_policy[i], new_action):
                    policy_stable = False
                    optimal_policy[i] = new_action

            if policy_stable:
                break

        return optimal_policy, value_func


class CarRental:
    def __int__(self) -> None:
        None

    def iterative_policy_eval(self):
        jack_env_obj = JacksCarRental()
        discount_factor = 0.9
        theta = 0.001  # convergence threshold
        value_func = np.zeros((21, 21), dtype=float)
        updated_value_func = np.zeros((21, 21), dtype=float)
        actions = jack_env_obj.action_space
        initial_policy = np.zeros((21, 21), dtype=int)

        #  precompute the transitions and rewards.
        jack_env_obj.precompute_transitions()
        flag = True
        while flag:
            delta = 0.0
            for i, state in enumerate(jack_env_obj.state_space):
                action = initial_policy[state]
                v = value_func[state]
                updated_value_func[state] = jack_env_obj.expected_return(value_func, state, action, discount_factor)
                delta = max(delta, abs(v - updated_value_func[state]))
            value_func = updated_value_func
            if delta < theta:
                return value_func

    # Policy Improvement
    def policy_iteration(self):
        jack_env_obj = JacksCarRental()
        discount_factor = 0.9
        theta = 0.001  # convergence threshold
        actions = jack_env_obj.action_space
        initial_policy = np.zeros((21, 21), dtype=int)

        #  precompute the transitions and rewards.
        jack_env_obj.precompute_transitions()
        policy_stable = True

        while policy_stable:
            policy_stable = False
            value_func = self.iterative_policy_eval()

            for i, state in enumerate(jack_env_obj.state_space):
                action_values = np.zeros(len(actions))

                for j, action in enumerate(actions):
                    action_values[j] = jack_env_obj.expected_return(value_func, state, action, discount_factor)

                new_action = actions[np.argmax(action_values)]
                current_action = initial_policy[state]

                if current_action != new_action:
                    policy_stable = True

                initial_policy[state] = new_action

        return value_func, initial_policy
