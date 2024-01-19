import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from enum import IntEnum
import enum


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action):
    """
    Helper function to map action to changes in x and y coordinates

    Args:
        action (Action): taken action

    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


def reset():
    """Return agent to start state"""
    return (0, 0)


# Q1
def simulate(state: Tuple[int, int], action: Action):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # Walls are listed for you
    # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
    walls = [
        (0, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 0),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 10),
        (6, 4),
        (7, 4),
        (9, 4),
        (10, 4),
    ]

    # TODO check if goal was reached
    goal_state = (10, 10)
    if state == goal_state:
        reward = 0.0
        next_state = reset()
        return next_state, reward

    # TODO modify action_taken so that 10% of the time,
    #  the action_taken is perpendicular to action (there are 2 perpendicular actions for each action)

    action_taken = action
    print("action = ", action)

    action_int = action_taken.value
    print("action_int = ", action_int)

    random_prob = random.random()
    #print("random_prob = ",random_prob)
    #random_prob = 0.06

    perpendicular_actions = {
        Action.LEFT: [Action.UP, Action.DOWN],
        Action.DOWN: [Action.LEFT, Action.RIGHT],
        Action.RIGHT: [Action.UP, Action.DOWN],
        Action.UP: [Action.LEFT, Action.RIGHT],
    }

    if random_prob < 0.1:
        action_taken = random.choice(perpendicular_actions[action])
    else:
        action_taken = action
    print("action_taken = ", action_taken)

    # TODO calculate the next state and reward given state and action_taken
    next_state = None
    reward = 0.0
    # You can use actions_to_dxdy() to calculate the next state
    dxdy = actions_to_dxdy(action_taken)
    print(dxdy)
    # Check that the next state is within boundaries and is not a wall
    next_state = (state[0] + dxdy[0], state[1] + dxdy[1])
    print("next_state", next_state)

    for iter in range(11):
        avoid_walls = [
            (-1, iter),
            (iter, -1),
            (11, iter),
            (iter, 11)
        ]
        walls = walls + avoid_walls
    if next_state in walls:
        # checks if the next state is a wall. If Yes, the code sets the next state to current one
        next_state = state

    if next_state != goal_state:
        reward = 0.0
    else:
        reward = 1.0

    # One possible way to work with boundaries is to add a boundary wall around environment and
    # simply check whether the next state is a wall

    return next_state, reward


# Q2
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    print("User is currently in this state = " + str(state))
    print("Select action - (0 = L, 1 = D, 2 = R, 3 = U) -->")
    action_selected = input()
    action_selected = int(action_selected)
    if(action_selected < 4):
        action = Action(action_selected)

    else:
        print("Invalid Input, Select action - (0 = L, 1 = D, 2 = R, 3 = U) --> ")
    print("action Taken = ", action)
    return action

pass


# Q2
def agent(
    steps: int = 1000,
    trials: int = 1,
    policy=Callable[[Tuple[int, int]], Action],
):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)

    """
    # TODO you can use the following structure and add to it as needed
    # rewards for all trials
    cumulative_reward = []
    for t in range(trials):
        state = reset()
        i = 0
        reward_collected = 0.0

        #total reward per trial
        reward_array = []
        while i < steps:
            # TODO select action to take
            # Get action from polic
            action = policy(state)

            print("action in agent = ", action)
            # TODO take step in environment using simulate()
            next_state, reward = simulate(state, action)
            state = next_state
            # TODO record the reward
            reward_collected += reward
            reward_array.append(reward_collected)
            i = i + 1
        cumulative_reward.append(reward_array)

    return cumulative_reward

# Q3
def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    action_choice = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
    # random uses uniform distribution
    return random.choice(action_choice)

    pass


# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    limited_action = [0,1,3]
    # there will be no action Right
    action = Action(random.choice(limited_action))

    return action
    pass


# Q4
def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    # Now that we know goal state is towards UP and Right combination direction, so higher value of probability would lead to goal state.
    if (random.random() < 0.7):
        limited_action = [2, 3]
        # there will be no action Right
        action = Action(random.choice(limited_action))
        action_choice = [Action.UP, Action.RIGHT]
        return random.choice(action_choice)
    else:
        action_choice = [Action.DOWN, Action.LEFT]
        limited_action = [0, 1]
        return random.choice(action_choice)
    pass


def main():

    # TODO run code for Q2~Q4 and plot results
    # You may be able to reuse the agent() function for each question
    print("Intro to RL")

    #Q2 Uncomment below Line to run Q2
    # cumulative_manual_policy = agent(steps=100, trials=1, policy=manual_policy)
    # print("cumulative_manual_policy = ", cumulative_manual_policy)

    # Q4 PlOT Uncomment below line to run Q4
    cumulative_better_policy = agent(steps=10000, trials=10, policy=better_policy)
    cumulative_random_policy = agent(steps=10000, trials=10, policy=random_policy)
    cumulative_worse_policy = agent(steps=10000, trials=10, policy=worse_policy)

    for iter in range(10):
        plt.plot(cumulative_better_policy[iter], ':')
        plt.plot(cumulative_random_policy[iter], ':')
        plt.plot(cumulative_worse_policy[iter], ':')

    plt.plot(np.average((cumulative_better_policy), 0), color='blue', label='Better Policy', linewidth=2)
    plt.plot(np.average((cumulative_random_policy), 0), color='black', label='Random Policy', linewidth=2)
    plt.plot(np.average((cumulative_worse_policy), 0), color='green', label='Worse Policy', linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel('Cumulative reward')
    plt.title("Policy Comparisons")
    plt.legend()
    plt.show()

    #Q3 PLot - Uncomment below lines to implement Q3
    # cumulative_random_policy = agent(steps=10000, trials=10, policy=random_policy)
    # for iter in range(10):
    #     plt.plot(cumulative_random_policy[iter], ':')
    #
    # plt.plot(np.average((cumulative_random_policy), 0), color='black', label='Random Policy', linewidth=2)
    #
    # plt.xlabel('Steps')
    # plt.ylabel('Cumulative reward')
    # plt.title("Random Policy Plot")
    # plt.legend()
    # plt.show()



    #Trial Code, Do not run
    # next_state, reward = simulate((9,10), "UP")
    # print(next_state, reward)
    # action = random_policy((1,2))
    # print("random action = ", action)

    # cumulative_rewards = agent(10000, 10, policy=random_policy)
    # print("cumulative_rewards = ", cumulative_rewards)

    # cumulative_rewards = agent(10000, 10, policy=better_policy)
    # print("cumulative_rewards = ", cumulative_rewards)
    pass

if __name__ == "__main__":
    main()