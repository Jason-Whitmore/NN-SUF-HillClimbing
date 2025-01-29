from time import time
from AgentREINFORCE import AgentREINFORCE
from Maze import Maze

from NNSUF import NNSUF
from KOSUF import KOSUF

class Experiment:
    """
    Defines the Experiment class. Contains RL components and functions to train and evaluate an agent/SUF
    """
    
    def __init__(self, env, state_update_function, agent):
        """
        Initializing the Experiment class.
        """
        self.env = env
        self.state_update_function = state_update_function
        self.agent = agent

    def get_MCSV(self, n: int, discount_factor=0.99) -> float:
        """
        Returns the Monte Carlo Start Value

        n: The number of episodes of data to collect

        discount_factor: The hyperparameter that controls how future rewards are weighted. This value is in [0, 1).
        """
        sum = 0

        for i in range(n):
            total_return, timesteps = self.run_episode(discount_factor=discount_factor, train=False)
            sum += total_return
        
        return sum / n

    def run_episode(self, discount_factor=0.99, train=False) -> float:
        """
        Returns the discounted return and number of timesteps from an episode run under the current policy.

        discount_factor: The hyperparameter that controls how future rewards are weighted. This value is in [0, 1).

        train: If true, agent will adjust parameters to improve performance. If false, no agent parameters are changed.
        """

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
        """
        Trains the agent for potentially many episodes

        timestep_limit: The approximate number of timesteps that the agent will train over. The actual number of timesteps trained
        is likely higher than this amount.

        discount_factor: The hyperparameter that controls how future rewards are weighted. This value is in [0, 1).
        """
        total_timesteps = 0

        while total_timesteps < timestep_limit:
            total_return, timesteps = self.run_episode(discount_factor=discount_factor, train=True)

            total_timesteps += timesteps

    def run_experiment(self, num_iterations: int, timesteps_per_iteration: int, discount_factor: float, epsilon: float, mcsv_samples: int, mcsv_threshold: float):
        """
        Runs the experiment and outputs training progress to standard out

        num_iterations: The number of times the NN-SUF is updated in the hill climbing algorithm

        timesteps_per_iteration: The approximate number of timesteps the agent will train per hill climbing algorithm iteration.
        The actual number of timesteps trained is likely higher than this amount.

        discount_factor: The hyperparameter that controls how future rewards are weighted. This value is in [0, 1).

        epsilon: Controls the bounds of the uniform random distribution that adjusts the NN-SUF parameters. 
        Parameter changes are in range [-epsilon, epsilon]

        mcsv_samples: The number of episodes collected when calculating MCSV. This number is used to optimize the NN-SUF's parameters.

        mcsv_threshold: If the MCSV reaches this threshold, the experiment is stopped early.
        """
        for iteration in range(num_iterations):
            start_time = time()
            self.train_mdp_agent(timesteps_per_iteration, discount_factor)
            end_time = time()

            mcsv: float = self.get_MCSV(mcsv_samples)
            print("Iteration " + str(iteration) + ": MCSV: " + str(mcsv) + " Time elapsed: " + str(end_time - start_time))

            if mcsv > mcsv_threshold:
                break
            
            if self.state_update_function is NNSUF:
                self.state_update_function.set_current_param_performance(mcsv)
                self.state_update_function.create_new_params(epsilon)




#Experiment parameters
num_converter_iterations: int = 15 #50 for nn-suf, 15 for kth order methods

#NN-SUF hyperparameters
state_size: int = 3 #{1, 2, 3, 4, 5}, 3 for epsilon experiment
interval: int = 10 * 1000 * 1000 #10_000_000
epsilon: float = 0 #{1e-1, 1e-2, 1e-3, 1e-4, 1e-5}, 1e-3 for state vector size experiment
nn_suf_hidden_layer_size: int = 30 #30

#Agent Hyperparameters
agent_hidden_layer_size: int = 24 #24
discount_factor: float = .99 #.99
policy_lr: float = 1e-4 #1e-4

#Experiment Hyperparameters
mcsv_num_samples_start_end: int = 10_000 #10_000
mcsv_num_samples_interval: int = 100 #100
MCSV_threshold: float = 0.8 #0.8

#For kth order methods
k: int = 1


#Use this variable to switch between NN-SUF and KO-SUF
use_nn_suf: bool = False

env = Maze()


action_count = env.action_space.n
dummy_obs, dummy = env.reset()
obs_size = len(dummy_obs)

if use_nn_suf:
    conv: NNSUF = NNSUF(state_size=state_size, obs_size=obs_size, action_count=action_count, epsilon=epsilon, hidden_layer_size=nn_suf_hidden_layer_size)
else:
    conv = KOSUF(k=k, obs_size=obs_size, action_count=action_count)
    state_size = conv.state_size

agent: AgentREINFORCE = AgentREINFORCE(state_size=state_size, action_count=action_count, hidden_layer_size=agent_hidden_layer_size, discount_factor=discount_factor, policy_lr=policy_lr)


e = Experiment(env, conv, agent)


print("State size: ", state_size)
print("Action count: ", action_count)
print("Obs size: ", obs_size)
print("Compression ratio: ", (state_size + 1 + agent.action_count + conv.obs_size)/state_size)


start_mcsv = e.get_MCSV(mcsv_num_samples_start_end)
print("Start MCSV: " + str(start_mcsv))



start_time = time()

e.run_experiment(num_iterations=num_converter_iterations, timesteps_per_iteration=interval, discount_factor=discount_factor, epsilon=epsilon, mcsv_samples=mcsv_num_samples_interval, mcsv_threshold=MCSV_threshold)
end_time = time()

if use_nn_suf:
    e.state_update_function.update_function_params = e.state_update_function.best_converter_params

final_mcsv = e.get_MCSV(mcsv_num_samples_start_end)

print("Final MCSV: " + str(final_mcsv))
print("Change in mean start value:", str(final_mcsv - start_mcsv))

le: float = (final_mcsv - start_mcsv) / (end_time - start_time)

print("Learning efficiency: ", le)
