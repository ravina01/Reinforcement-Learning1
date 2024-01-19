# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# print(gpu_info)

# import gym
# env = gym.make("RocketLander-v0")
# env.reset()
# a = env.action_space
# print(a)                    #prints Discrete(3)
# print(a.n)


# import gym
# import gym.spaces
# # import rocket_lander_gym env = gym.envs.registration.register(id='RocketLander-v0',
# # entry_point='rocketlander.rocket_lander:RocketLander', max_episode_steps=1000, reward_threshold=0, )
# env = gym.make('RocketLander-v0')
# # env.reset()
# a = env.action_space
# print(a)
# PRINT_DEBUG_MSG = True
#
#
# while True:
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#
#     if PRINT_DEBUG_MSG:
#         print("Action Taken  ",action)
#         print("Observation   ",observation)
#         print("Reward Gained ",reward)
#         print("Info          ",info,end='\n\n')
#
#     if done:
#         print("Simulation done.")
#         break
# env.close()

import numpy as np
import Box2D
