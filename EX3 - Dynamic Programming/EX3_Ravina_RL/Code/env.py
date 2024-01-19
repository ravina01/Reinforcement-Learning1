from scipy.stats import poisson
import numpy as np
from enum import IntEnum
from typing import Tuple

class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
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


class Gridworld5x5:
    """5x5 Gridworld"""

    def __init__(self) -> None:
        """
        State: (x, y) coordinates

        Actions: See class(Action).
        """
        self.rows = 5
        self.cols = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = len(Action)

        # TODO set the locations of A and B, the next locations, and their rewards
        self.A = (4, 1)
        self.A_prime = (0, 1)
        self.A_reward = 10.0
        self.B = (4, 3)
        self.B_prime = (2, 3)
        self.B_reward = 5.0

    def transitions(
        self, state: Tuple, action: Action
    ) -> Tuple[Tuple[int, int], float]:
        """Get transitions from given (state, action) pair.

        Note that this is the 4-argument transition version p(s',r|s,a).
        This particular environment has deterministic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            next_state: Tuple[int, int]
            reward: float
        """
        next_state = None
        reward = None

        self.rows = 5
        self.cols = 5
        self.state_space = [
            (x, y) for x in range(0, self.rows) for y in range(0, self.cols)
        ]
        self.action_space = len(Action)

        # TODO Check if current state is A and B and return the next state and corresponding reward
        if state == self.A:
            next_state = self.A_prime
            reward = self.A_reward
        elif state == self.B:
            next_state = self.B_prime
            reward = self.B_reward
        # Else, check if the next step is within boundaries and return next state and reward
        else:
            dx, dy = actions_to_dxdy(action)
            next_x = state[0] + dx
            next_y = state[1] + dy
            next_state = (next_x, next_y)
            # Let's check if the next_state is within the bounds of the grid
            outside_left_bound = (next_state[0] < 0 or next_state[1] < 0)
            outside_right_bound = (next_state[0] > 4 or next_state[1] > 4)

            if outside_left_bound or outside_right_bound:
                next_state = state
                reward = -1.0
            else:
                reward = 0.0
        return next_state, reward

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        next_state, reward = self.transitions(state, action)
        # TODO compute the expected return
        ret = None
        next_state_index = self.state_space.index(next_state)
        ret = (reward + V[next_state_index]*gamma)
        return ret


class JacksCarRental:
    def __init__(self, modified: bool = True) -> None:
        """JacksCarRental

        Args:
           modified (bool): False = original problem Q6a, True = modified problem for Q6b

        State: tuple of (# cars at location A, # cars at location B)

        Action (int): -5 to +5
            Positive if moving cars from location A to B
            Negative if moving cars from location B to A
        """
        self.modified = modified

        self.action_space = list(range(-5, 6))

        self.rent_reward = 10
        self.move_cost = 2

        # For modified problem
        self.overflow_cars = 10
        self.overflow_cost = 4

        # Rent and return Poisson process parameters
        # Save as an array for each location (Loc A, Loc B)
        self.rent = [poisson(3), poisson(4)]
        self.return_ = [poisson(3), poisson(2)]

        # Max number of cars at end of day
        self.max_cars_end = 20
        # Max number of cars at start of day
        self.max_cars_start = self.max_cars_end + max(self.action_space)

        self.state_space = [
            (x, y)
            for x in range(0, self.max_cars_end + 1)
            for y in range(0, self.max_cars_end + 1)
        ]

        # Store all possible transitions here as a multi-dimensional array (locA, locB, action, locA', locB')
        # This is the 3-argument transition function p(s'|s,a)
        self.t = np.zeros(
            (
                self.max_cars_end + 1,
                self.max_cars_end + 1,
                len(self.action_space),
                self.max_cars_end + 1,
                self.max_cars_end + 1,
            ),
        )

        # Store all possible rewards (locA, locB, action)
        # This is the reward function r(s,a)
        self.r = np.zeros(
            (self.max_cars_end + 1, self.max_cars_end + 1, len(self.action_space))
        )

    def _open_to_close(self, loc_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the probability of ending the day with s_end \in [0,20] cars given that the location started
         with s_start \in [0, 20+5] cars.

        Args:
            loc_idx (int): the location index. 0 is for A and 1 is for B. All other values are invalid
        Returns:
            probs (np.ndarray): list of probabilities for all possible combination of s_start and s_end
            rewards (np.ndarray): average rewards for all possible s_start
        """
        probs = np.zeros((self.max_cars_start + 1, self.max_cars_end + 1))
        rewards = np.zeros(self.max_cars_start + 1)
        for start in range(probs.shape[0]):
            # TODO Calculate average rewards.
            # For all possible s_start, calculate the probability of renting k cars.
            # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
            avg_rent = 0.0
            for rent in range(start + 1):
                # prob_rent is the probability of renting exactly rent cars
                prob_rent = self.rent[loc_idx].pmf(rent)
                # Calculate expected reward for renting rent cars
                # If more cars are rented than are available,
                # the expected reward is the reward for renting the available number of cars
                expected_reward = min(rent, start) * self.rent_reward

                avg_rent += prob_rent * expected_reward
            # Add expected reward for lost business
            avg_rent += (1 - self.rent[loc_idx].cdf(start)) * start * self.rent_reward
            rewards[start] = avg_rent

            # TODO Calculate probabilities
            # Loop over every possible s_end
            for end in range(probs.shape[1]):
                prob = 0.0
                # Since s_start and s_end are specified,
                # you must rent a minimum of max(0, start-end)
                min_rent = max(0, start - end)

                # TODO Loop over all possible rent scenarios and compute probabilities
                # Be sure to consider the case where business is lost (i.e. renting k > s_start cars)
                for i in range(min_rent, start + 1):
                    # prob_rent is the probability of renting exactly rent cars
                    prob_rent = self.rent[loc_idx].pmf(i)

                    # prob_return is the probability of returning end - start + rent cars
                    # no. of cars that will be at the location at the end of the day
                    prob_return = self.return_[loc_idx].pmf(end - start + rent)
                    prob += prob_rent * prob_return
                probs[start, end] = prob

        return probs, rewards

    def _calculate_cost(self, state: Tuple[int, int], action: int) -> float:
        """A helper function to compute the cost of moving cars for a given (state, action)

        Note that you should compute costs differently if this is the modified problem.

        Args:
            state (Tuple[int,int]): state
            action (int): action
        """
        cost = 0.0
        if self.modified:
            #cost of moving cars is complex.
            #cost = 0.0
            if action > 0:
                cost += abs(action - 1) * self.move_cost
            else:
                cost += abs(action) * self.move_cost
            # number of cars at the A location after the move is calculated.
            cars_at_A = state[0] - action
            # number of cars at the B location after the move is calculated.
            cars_at_B = state[1] + action

            # number of cars at the first location after the move is greater than 10, increase by the overflow cost.
            if cars_at_A > 10:
                cost += self.overflow_cost
            # number of cars at the first location after the move is greater than 10, increase by the overflow cost.
            if cars_at_B > 10:
                cost += self.overflow_cost
        else:
            # when modified is false
            cost = abs(action) * self.move_cost

        return cost

    def _valid_action(self, state: Tuple[int, int], action: int) -> bool:
        """Helper function to check if this action is valid for the given state

        Args:
            state:
            action:
        """
        if state[0] < action or state[1] < -action:
            return False
        else:
            return True

    def precompute_transitions(self) -> None:
        """Function to precompute the transitions and rewards.

        This function should have been run at least once before calling expected_return().
        You can call this function in __init__() or separately.

        """
        # Calculate open_to_close for each location
        day_probs_A, day_rewards_A = self._open_to_close(0)
        day_probs_B, day_rewards_B = self._open_to_close(1)

        # Perform action first then calculate daytime probabilities
        for locA in range(self.max_cars_end + 1):
            for locB in range(self.max_cars_end + 1):
                for ia, action in enumerate(self.action_space):
                    # Check boundary conditions
                    if not self._valid_action((locA, locB), action):
                        self.t[locA, locB, ia, :, :] = 0
                        self.r[locA, locB, ia] = 0
                    else:
                        # TODO Calculate day rewards from renting
                        day_reward_A = day_rewards_A[locA - action]
                        day_reward_B = day_rewards_B[locB + action]

                        cost = self._calculate_cost((locA, locB), action)
                        # Use day_rewards_A and day_rewards_B and _calculate_cost()
                        self.r[locA, locB, ia] = day_reward_A + day_reward_B - cost

                        # Loop over all combinations of locA_ and locB_
                        for locA_ in range(self.max_cars_end + 1):
                            for locB_ in range(self.max_cars_end + 1):

                                # TODO Calculate transition probabilities
                                # Use the probabilities computed from open_to_close

                                self.t[locA, locB, ia, locA_, locB_] = day_probs_A[locA-action,locA_] * day_probs_B[locB+action,locB_]

    def expected_return(
        self, V, state: Tuple[int, int], action: Action, gamma: float
    ) -> float:
        """Compute the expected_return for all transitions from the (s,a) pair, i.e. do a 1-step Bellman backup.

        Args:
            V (np.ndarray): list of state values (length = number of states)
            state (Tuple[int, int]): state
            action (Action): action
            gamma (float): discount factor

        Returns:
            ret (float): the expected return
        """

        # TODO compute the expected return
        ret = 0.0
        probs = self.transitions(state, action)
        reward = self.rewards(state, action)

        for state in self.state_space:
            ret += probs[state] * gamma * V[state]
        ret = ret + reward
        return ret

    def transitions(self, state: Tuple, action: Action) -> np.ndarray:
        """Get transition probabilities for given (state, action) pair.

        Note that this is the 3-argument transition version p(s'|s,a).
        This particular environment has stochastic transitions

        Args:
            state (Tuple): state
            action (Action): action

        Returns:
            probs (np.ndarray): return probabilities for next states. Since transition function is of shape
            (locA, locB, action, locA', locB'), probs should be of shape (locA', locB')
        """
        # TODO
        probs = None
        # (locA', locB') --> no of cras at A and B in the next state.
        action_index = self.action_space.index(action)
        probs = self.t[state[0], state[1], action_index, :, :]
        return probs

    def rewards(self, state, action) -> float:
        """Reward function r(s,a)

        Args:
            state (Tuple): state
            action (Action): action
        Returns:
            reward: float
        """
        # TODO
        # positive action --> from A to B
        # negative action --> from B to A
        # tensor r stores all rewards.
        action_index = self.action_space.index(action)
        reward = self.r[state[0], state[1], action_index]
        return reward