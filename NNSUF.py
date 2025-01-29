import numpy as np
from Utility import Utility


class NNSUF:
    """
    Defines the Neural Network State Update Function
    """

    def __init__(self, state_size: int, obs_size: int, action_count: int, epsilon: float, hidden_layer_size: int):
        """
        Initializes the NN-SUF
        """
        
        self.state_size: int = state_size
        self.epsilon: float = epsilon

        
        self.hidden_layer_size = hidden_layer_size

        #Env variables
        self.action_count: int = action_count
        self.obs_size: int = obs_size

        

        #Interaction variables
        self.prev_state = None
        self.prev_action = None

        self.update_function_params: "list[ndarray]" = self.create_update_function_params()

        #Best parameters
        self.best_converter_params: "list[ndarray]" = self.update_function_params
        self.best_performance: float = float("-inf")

    def create_update_function_params(self) -> list["ndarray"]:
        """
        Creates the NN-SUF parameters.
        """
        params = []

        #First layer
        params.append(np.zeros((self.hidden_layer_size, self.state_size + self.obs_size + self.action_count + 1)))
        #Utility.init_weights_xavier_uniform(params[-1])

        params.append(np.zeros((self.hidden_layer_size)))

        #Second layer
        params.append(np.zeros((self.hidden_layer_size, self.hidden_layer_size)))
        #Utility.init_weights_xavier_uniform(params[-1])

        params.append(np.zeros((self.hidden_layer_size)))

        #Output layer
        params.append(np.zeros((self.state_size, self.hidden_layer_size)))
        #Utility.init_weights_xavier_uniform(params[-1])

        params.append(np.zeros((self.state_size)))

        #Randomize params
        Utility.init_params_xavier_uniform(params)


        return params

    
    def step(self, observation: "ndarray", action: int, reward: float, done: bool):
        """
        Performs the SUFs part of the interaction loop.

        obs: The new observation
        action: The previous action
        reward: The previous reward
        done: The signal that indicates if the episode is complete.

        Returns the new state, reward, and the done signal
        """
        
        state = self.update_function_predict(observation, reward, self.prev_state, self.prev_action)

        self.prev_state = state

        if done:
            self.episode_end_reset()

        return state, reward, done
        

    def set_current_param_performance(self, current_performance: float):
        """
        Compares the current performance of the NN-SUF with the previous best performance.
        If better, the current performance and NN-SUF parameters become the new best performance and parameters.

        current_performance: Any selected objective function (example: MCSV)
        """

        #Conduct test
        current_better: bool = current_performance > self.best_performance

        if current_better:
            #Current params are best. Set new best parameters
            self.best_converter_params = Utility.clone_list(self.update_function_params)

            #self.best_mean_reward = current_mean_reward
            self.best_performance = current_performance


    
    def create_new_params(self, epsilon: float):
        """
        Creates new parameters by cloning the existing parameters and adding small random values to each component.

        epsilon: The bounds for the random values. These random values are selected from a random uniform
        distribution in [-epsilon, epsilon]
        """
        new_params: "list[ndarray]" = Utility.clone_list(self.best_converter_params)


        for i in range(len(new_params)):
            if new_params[i].ndim == 2:
                for j in range(len(new_params[i])):
                    for k in range(len(new_params[i][j])):
                        new_params[i][j][k] += np.random.uniform(-epsilon, epsilon)
            else:
                for j in range(len(new_params[i])):
                    new_params[i][j] += np.random.uniform(-epsilon, epsilon)

        self.update_function_params = new_params

    def episode_end_reset(self):
        """
        Performs end of episode actions. This resets the previous state and action values to dummy values
        """
        self.prev_state = np.zeros(self.state_size, dtype="float32") - 1
        self.prev_action = -1


    def update_function_predict(self, next_obs: "ndarray", reward: float, prev_state: "ndarray", prev_action: int) -> "ndarray":
        """
        Performs the feedforward pass of the NN-SUF.

        next_obs: The next observation
        
        reward: The previous reward

        prev_state: The previous state

        prev_action: The previous action

        Returns the new state
        """
        
        x: "ndarray" = self.create_update_input_array(next_obs, reward, prev_state, prev_action)
        y: "ndarray" = None

        for i in range(0, len(self.update_function_params) - 1, 2):
            w: "ndarray" = self.update_function_params[i]
            b: "ndarray" = self.update_function_params[i + 1]

            y = np.matmul(w, x) + b
            y = Utility.tanh(y)


            x = np.array(y)

        return np.array(y)

    def get_start_state(self, start_obs: "ndarray") -> "ndarray":
        """
        Returns the start state

        start_obs: The starting observation
        """
        return self.update_function_predict(start_obs, 0, self.prev_state, self.prev_action)
    
    def create_update_input_array(self, next_obs: "ndarray", reward: float, prev_state: "ndarray", prev_action: int) -> "ndarray":
        """
        Prepares and returns in input array for the NN-SUF

        next_obs: The next observation

        reward: The next reward

        prev_state: The previous state

        prev_action: The previous action
        """

        input_size: int = self.obs_size + self.action_count + self.state_size + 1

        #Concatenate next observation and prev state into one array
        r: "ndarray" = np.zeros(input_size, dtype="float32")

        index: int = 0
        for i in range(len(next_obs)):
            r[index] = next_obs[i]
            index += 1
        
        r[index] = reward
        index += 1

        for i in range(len(prev_state)):
            r[index] = prev_state[i]
            index += 1
        
        #If there is no previous action (at start state), then don't populate any more of the return vector
        if prev_action == -1:
            return r

        r[index + prev_action] = 1

        return r