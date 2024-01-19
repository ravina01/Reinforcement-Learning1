import numpy as np
import matplotlib.pyplot as plt

# Define the MDP
class MDP:
    def __init__(self):
        self.states = ["s"]
        self.actions = ["left", "right"]
        self.transitions = {
            "s": {
                "left": [("s", 0.9), ("terminal", 0.1)],
                "right": [("terminal", 1.0)],
            },
        }
        self.rewards = {
            ("s", "left", "terminal"): 1.0,
            ("s", "right", "terminal"): 0.0,
        }

    def transition(self, state, action):
        next_states, rewards = np.array(self.transitions[state][action]).T
        next_state, reward = np.random.choice(next_states), np.random.choice(rewards)
        return next_state, reward


# Define the behavior policy
def behavior_policy(state):
    return np.random.choice(["left", "right"])

# Define the target policy
def target_policy(state):
    return "left"

# Estimate the value of state "s" using first-visit MC
def first_visit_mc(mdp, behavior_policy, target_policy, episodes):
    V = np.zeros(len(mdp.states))
    print("V = ", V)

    for episode in episodes:
        state = mdp.states[0]
        G = 0.0

        while state != "terminal":
            action = behavior_policy(state)
            next_state, reward = mdp.transition(state, action)
            G += float(reward)
            state = next_state

        V[0] += (G - V[0])
    V[0] += (G - V[0])
    return V

# Estimate the value of state "s" using every-visit MC
def every_visit_mc(mdp, behavior_policy, target_policy, episodes):
    V = np.zeros(len(mdp.states))
    N = np.zeros(len(mdp.states))

    for episode in episodes:
        state = mdp.states[0]
        G = 0.0

        while state != "terminal":
            action = behavior_policy(state)
            next_state, reward = mdp.transition(state, action)
            G += float(reward)
            state = next_state

            N[0] += 1
            V[0] += (G - V[0]) / N[0]

    return V

# Generate episodes
mdp = MDP()
episodes = []

for i in range(10000):
    state = mdp.states[0]
    episode = []

    while state != "terminal":
        action = behavior_policy(state)
        next_state, reward = mdp.transition(state, action)
        episode.append((state, action, reward))
        state = next_state

    episodes.append(episode)

# Estimate the value of state "s" using first-visit MC
V_first_visit = first_visit_mc(mdp, behavior_policy, target_policy, episodes)

# Estimate the value of state "s" using every-visit MC
V_every_visit = every_visit_mc(mdp, behavior_policy, target_policy, episodes)

# Define the x and y arrays
x = np.arange(10000)
y = np.zeros(1)

# Reshape the y array
y_reshaped = y.reshape(10000,)

# Plot the results
plt.plot(x, y_reshaped, label="First-visit MC")
plt.plot(x, y_reshaped, label="Every-visit MC")
plt.xlabel("Episodes")
plt.ylabel("V(s)")
plt.legend()
plt.show()