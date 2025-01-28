from time import time
import AgentREINFORCE
from AgentREINFORCE import AgentREINFORCE
from Maze import Maze
from Utility import Utility

from NNSUF import NNSUF
from KOSUF import KOSUF

import gymnasium as gym


class Experiment:
    
    def __init__(self, env, state_update_function, agent):
        self.env = env
        self.state_update_function = state_update_function
        self.agent = agent

    def get_MCSV(self, n: int, discount_factor=0.99) -> float:
        
        sum = 0

        for i in range(n):
            total_return, timesteps = self.run_episode(discount_factor=discount_factor, train=False)
            sum += total_return
        
        return sum / n

    def run_episode(self, discount_factor=0.99, train=False) -> float:

        total_return: float = 0
        start_obs, dummy = env.reset()
        self.state_update_function.episode_end_reset()
        self.agent.episode_end_reset()

        state = self.state_update_function.get_start_state(start_obs)

        self.agent.current_state = state

        probs, d1, d2 = self.agent.policy_function_predict(state)
        action = AgentREINFORCE.select_action(probs)

        done = False

        timestep: int = 0

        while not done:
            obs, reward, done, d1, d2 = self.env.step(action)

            total_return += (discount_factor**timestep) * reward

            state, reward, done = self.state_update_function.step(obs, action, reward, done)

            action = self.agent.step(reward, state, done)

            timestep += 1

        if train:
            self.agent.REINFORCE_update()

        self.state_update_function.episode_end_reset()
        self.agent.episode_end_reset()

        return total_return, timestep


    def train_mdp_agent(self, timestep_limit: int, discount_factor: float=0.99):
        total_timesteps = 0

        while total_timesteps < timestep_limit:
            total_return, timesteps = self.run_episode(discount_factor=discount_factor, train=True)

            total_timesteps += timesteps

    def run_experiment(self, num_iterations: int, timesteps_per_iteration: int, discount_factor: float, epsilon: float, mcsv_samples: int, mcsv_threshold: float):

        for iteration in range(num_iterations):
            start_time = time()
            self.train_mdp_agent(timesteps_per_iteration, discount_factor)
            end_time = time()

            mcsv: float = self.get_MCSV(mcsv_samples)
            print("Iteration " + str(iteration) + ": MCSV: " + str(mcsv) + " Time elapsed: " + str(end_time - start_time))

            if mcsv > mcsv_threshold:
                break
            
            if self.state_update_function is Converter:
                self.state_update_function.set_current_param_performance(mcsv)
                self.state_update_function.create_new_params(epsilon)

    def run_experiment_hill_climbing(self, num_iterations: int, epsilon: float, discount_factor: float, mcsv_samples: int, mcsv_threshold: float):

        best_agent_params = Utility.clone_list(self.agent.policy_function_params)
        best_update_function_params = Utility.clone_list(self.state_update_function.update_function_params)
        best_mcsv: float = float("-inf")

        for iteration in range(num_iterations):

            mcsv: float = self.get_MCSV(mcsv_samples)

            print("Iteration " + str(iteration) + ": MCSV: " + str(mcsv))

            if mcsv > best_mcsv:
                best_mcsv = mcsv
                best_agent_params = Utility.clone_list(self.agent.policy_function_params)
                best_update_function_params = Utility.clone_list(self.state_update_function.update_function_params)

            if mcsv > mcsv_threshold:
                return
            
            new_agent_params = Utility.create_new_params(current_params=best_agent_params, epsilon=epsilon)
            new_update_function_params = Utility.create_new_params(current_params=best_update_function_params, epsilon=epsilon)

            self.agent.policy_function_params = Utility.clone_list(new_agent_params)
            self.state_update_function.update_function_params = Utility.clone_list(new_update_function_params)




#Experiment parameters
num_converter_iterations: int = 50 #50 for nn-suf, 15 for kth order methods

#NN-SUF hyperparameters
state_size: int = 50 #{1, 2, 3, 4, 5}, 3 for epsilon experiment
interval: int = 10 * 1000 * 1000 #10_000_000
epsilon: float = 0 #{1e-1, 1e-2, 1e-3, 1e-4, 1e-5}, 1e-3 for state vector size experiment
converter_hidden_layer_size: int = 50 #30

#Agent Hyperparameters
agent_hidden_layer_size: int = 24 #24
discount: float = .99 #.99
policy_lr: float = 1e-4 #1e-4

#Experiment Hyperparameters
num_samples_start_end_score: int = 10_000 #10_000
num_episodes_logging: int = 100 #100
start_value_threshold: float = 0.8 #0.8

#For kth order methods
k: int = 3

use_nn_suf: bool = True

env = Maze()
#env = Breakout()


action_count = env.action_space.n
dummy_obs, dummy = env.reset()
obs_size = len(dummy_obs)

if use_nn_suf:
    conv: NNSUF = NNSUF(state_size=state_size, obs_size=obs_size, action_count=action_count, epsilon=epsilon, hidden_layer_size=converter_hidden_layer_size)
else:
    conv = KOSUF(k=k, obs_size=obs_size, action_count=action_count)
    state_size = conv.state_size

agent: AgentREINFORCE = AgentREINFORCE(state_size=state_size, action_count=action_count, hidden_layer_size=agent_hidden_layer_size, discount_factor=discount, policy_lr=policy_lr)


e = Experiment(env, conv, agent)


print("State size: ", state_size)
print("Action count: ", action_count)
print("Obs size: ", obs_size)
print("Compression ratio: ", (state_size + 1 + agent.action_count + conv.obs_size)/state_size)


start_mcsv = e.get_MCSV(num_samples_start_end_score)
print("Start MCSV: " + str(start_mcsv))



start_time = time()

e.run_experiment(num_iterations=num_converter_iterations, timesteps_per_iteration=interval, discount_factor=discount, epsilon=epsilon, mcsv_samples=num_episodes_logging, mcsv_threshold=start_value_threshold)
end_time = time()

if use_nn_suf:
    e.state_update_function.update_function_params = e.state_update_function.best_converter_params

final_mcsv = e.get_MCSV(num_samples_start_end_score)

print("Final MCSV: " + str(final_mcsv))
print("Change in mean start value:", str(final_mcsv - start_mcsv))

le: float = (final_mcsv - start_mcsv) / (end_time - start_time)

print("Learning efficiency: ", le)
