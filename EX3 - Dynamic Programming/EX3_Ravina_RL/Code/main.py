"""
Author - Ravina Lad
EX3 - Dynammic Programming
main.py - To run Q 5 and Q 6 -> the gridworld and car rental problem
"""
from algorithm import GridWorld, CarRental
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from scipy.stats import poisson


def gridworld():
    gridworld_obj = GridWorld()
    # value_func = gridworld_obj.iterative_policy_eval()
    # print("Value Function for iterative policy evaluation\n", format(np.round(value_func, 1)))

    # policy, optimal_val = gridworld_obj.value_iteration()
    # print("Optimal Value function for Value Iteration \n", format(np.round(optimal_val,1)))
    # print("Optimal policy for Value Iteration \n", format(policy))
    #
    optimal_policy, optimal_val = gridworld_obj.policy_iteration()
    print("Optimal Value function for Policy Iteration \n", format(np.round(optimal_val, 1)))
    print("Optimal policy function for Policy Iteration \n", format(optimal_policy))

def carRental():
    carRental_obj = CarRental()
    optimal_policy, optimal_val = carRental_obj.policy_iteration()
    print("Optimal Value function for Policy Iteration \n", format(np.round(optimal_val, 1)))
    print("Optimal policy function for Policy Iteration \n", format(optimal_policy))

    plt.imshow(optimal_policy, origin='lower')
    plt.colorbar()
    plt.plot()
    plt.savefig("C:/Users/ravin/Desktop/Fall'23/RL/ex3/policy_original.png")

    (x, y) = np.meshgrid(np.arange(optimal_val.shape[0]), np.arange(optimal_val.shape[1]))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Generate Poisson distribution with lambda = optimal_val
    (x, y) = np.meshgrid(np.arange(optimal_val.shape[0]), np.arange(optimal_val.shape[1]))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(x, y, (np.round(optimal_val, 1)), cmap=plt.cm.coolwarm)
    fig.colorbar(surf)
    fig.show()
    plt.savefig("C:/Users/ravin/Desktop/Fall'23/RL/ex3/value_original.png")

def main():
    # Uncomment to run Q5
    #gridworld()
    # Uncomment to run Q6 , To modify the policy make chnages in env.py modified = True
    # Unfortunately i couldn't do it in the main function
    #carRental()

if __name__ == "__main__":
    main()