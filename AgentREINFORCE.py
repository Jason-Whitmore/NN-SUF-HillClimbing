import numpy as np
from Utility import Utility

class AgentREINFORCE:
    """
    Defines a RL agent that uses the REINFORCE algorithm to learn an optimal policy.
    """

    def __init__(self, state_size: int, action_count: int, hidden_layer_size: int, discount_factor: float, policy_lr: float):
        """
        Initializes the agent.

        state_size: The size of the state vector

        action_count: The number of actions in the action set

        hidden_layer_size: The size of the hidden layers in the policy neural network

        discount_factor: The hyperparameter that controls how future rewards are weighted. This value is in [0, 1).

        policy_lr: The learning rate for the policy
        """
        
        #Basic hyperparameters
        self.hidden_layer_size: int = hidden_layer_size
        self.policy_lr: float = policy_lr

        #MDP variables
        self.state_size: int = state_size
        self.action_count: int = action_count
        self.current_state: "ndarray" = None
        self.current_timestep: int = 0
        self.discount_factor: float = discount_factor

        #REINFORCE variables
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_probs = []
        self.episode_layer_inputs = []
        self.episode_pre_activation_vectors = []

        self.dtype = "float64"

        #Functions
        self.policy_function_params: list["ndarray"] = self.create_policy_function_params()
    
    def episode_end_reset(self):
        """
        Clears algorithm data that was collected from an old episode in anticipation of a new episode.
        """
        self.current_timestep = 0

        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_probs = []
        self.episode_layer_inputs = []
        self.episode_pre_activation_vectors = []


    def create_policy_function_params(self) -> list["ndarray"]:
        """
        Creates the policy neural network parameters
        """
        params: list["ndarray"] = []

        #First layer
        params.append(np.zeros((self.hidden_layer_size, self.state_size), dtype=self.dtype))

        params.append(np.zeros((self.hidden_layer_size), dtype=self.dtype))

        #Second layer
        params.append(np.zeros((self.hidden_layer_size, self.hidden_layer_size), dtype=self.dtype))

        params.append(np.zeros((self.hidden_layer_size), dtype=self.dtype))

        #Output layer
        params.append(np.zeros((self.action_count, self.hidden_layer_size), dtype=self.dtype))
        params.append(np.zeros((self.action_count), dtype=self.dtype))

        #Initialize parameters
        Utility.init_params_xavier_uniform(params)
        params[-2] *= 0.01
        params[-1] *= 0

        return params

    def set_current_state(self, current_state: "ndarray"):
        """
        Sets the current state
        """
        self.current_state = current_state


    def REINFORCE_update(self):
        """
        Performs the REINFORCE parameter update. This is the code which helps the agent learn.
        """

        eps: float = 1e-4

        policy_grad = Utility.clone_list(self.policy_function_params)
        policy_grad = Utility.clear_list(policy_grad)

        g: float = 0
        for t in range(len(self.episode_states) - 1, -1, -1):
            
            g = self.episode_rewards[t] + (self.discount_factor * g)

            probs = self.episode_probs[t]
            layer_inputs = self.episode_layer_inputs[t]
            pre_activation_vectors = self.episode_pre_activation_vectors[t]
            state = self.episode_states[t]
            actions = self.episode_actions[t]

            prob = probs[actions]

            scalar: float = self.policy_lr * g * (self.discount_factor**t) * (1.0/(prob + eps))

            policy_grad_t = self.policy_gradient(state, actions, layer_inputs, pre_activation_vectors)

            policy_grad_scaled_t = Utility.scale_list(policy_grad_t, scalar)

            policy_grad = Utility.add_list(policy_grad, policy_grad_scaled_t)

        old_params = Utility.clone_list(self.policy_function_params)
        new_params = Utility.add_list(old_params, policy_grad)

        self.policy_function_params = new_params
        
        self.episode_end_reset()

    def policy_function_predict(self, state: "ndarray") -> "ndarray":
        """
        Gets the output of the policy function as well as other important information.

        state: The input state vector

        Returns: The action probability distribution, the input vectors for each layer,
        and the output vectors for each layer before the activation function is applied.
        """

        layer_inputs = []
        pre_activation_vectors = []

        x: "ndarray" = state
        y: "ndarray" = None

        for i in range(0, len(self.policy_function_params) - 1, 2):
            layer_inputs.append(np.array(x, dtype=self.dtype))

            w: "ndarray" = self.policy_function_params[i]
            b: "ndarray" = self.policy_function_params[i + 1]

            y = np.matmul(w, x) + b

            pre_activation_vectors.append(np.array(y, dtype=self.dtype))

            if i != len(self.policy_function_params) - 2:
                y = Utility.hidden_activation(y)
            else:
                y = Utility.softmax(y)

            x = y

        return y, layer_inputs, pre_activation_vectors


    def policy_gradient(self, state: "ndarray", action: int, input_vectors, pre_activation_vectors) -> "list[ndarray]":
        """
        Gets the policy gradient of the action with respect to the parameters.

        state: The input state

        action: The input action. The gradient is the direction which increases the probability of this action the most

        input_vectors: List of input vectors for each layer

        pre_activation_vectors: List of output vectors for each layer before the activation functions are applied.

        Returns: The gradient
        """
        params: list["ndarray"] = self.policy_function_params
        gradient: list["ndarray"] = [None for i in range(len(params))]
        dObj_dY: list["ndarray"] = [None for i in range(int(len(params) / 2))]
        

        #Set objective
        dObj_dY[-1] = np.zeros(self.action_count, dtype=self.dtype)

        output = Utility.softmax(pre_activation_vectors[-1])

        for i in range(self.action_count):
            if i != action:
                dObj_dY[-1][i] = -1 * output[i] * output[action]
            else:
                dObj_dY[-1][i] = output[i] * (1 - output[i])

        num_layers = int(len(params) / 2)
        
        for l in range(num_layers - 1, -1, -1):
            
            #Determine dObj_dY[l]
            activation_prime = None

            #If not on last layer
            if l != num_layers - 1:
                temp: "ndarray" = np.zeros(len(params[(2 * l) + 1]))
                activation_prime = Utility.hidden_activation_prime(pre_activation_vectors[l])
                
                w = params[2 * (l + 1)]

                w_t = np.transpose(w)
                temp = np.matmul(w_t, dObj_dY[l + 1])

                temp = np.multiply(temp, activation_prime)

                dObj_dY[l] = np.array(temp)

            if l == num_layers - 1:
                activation_prime = np.zeros((pre_activation_vectors[l].shape)) + 1
            else:
                activation_prime = Utility.hidden_activation_prime(pre_activation_vectors[l])

            
            
            #Determine gradients for weights
            w_grad: "ndarray" = np.zeros(params[l * 2].shape, dtype=self.dtype)

            dot_product = np.zeros(len(dObj_dY[l]), dtype=self.dtype)
            dot_product = np.multiply(dObj_dY[l], activation_prime)
            
            #I iterates over rows
            for r in range(len(w_grad)):
                w_grad[r] = np.multiply(dot_product[r], input_vectors[l])


            #Determine gradients for biases
            b_grad: "ndarray" = np.zeros(params[(l * 2) + 1].shape, dtype=self.dtype)

            b_grad = np.array(dot_product)
            
            #Append gradients to the return list
            gradient[2 * l] = (w_grad)
            gradient[(2 * l) + 1] = (b_grad)

        return gradient

    def policy_gradient_simple(self, state: "ndarray", action: int) -> "list[ndarray]":
        """
        Gets the policy gradient of the action with respect to the parameters.

        state: The input state

        action: The input action. The gradient is the direction which increases the probability of this action the most

        Returns: The gradient
        """

        params = self.policy_function_params

        pre_activation_vectors = [None for i in range(int(len(params) / 2))]

        input_vectors = [None for i in range(int(len(params) / 2))]

        #Do forward pass
        x = state
        
        for i in range(0, len(params) - 1, 2):
            input_vectors[int(i / 2)] = np.array(x)

            w: "ndarray" = params[i]
            b: "ndarray" = params[i + 1]

            y = np.matmul(w, x) + b

            pre_activation_vectors[int(i / 2)] = np.array(y)

            if i != len(params) - 2:
                y = Utility.hidden_activation(y)
            else:
                y = Utility.softmax(y)

            x = y

        return self.policy_gradient(state, action, input_vectors, pre_activation_vectors)


    @staticmethod
    def select_action(prob_distribution: "ndarray") -> int:
        """
        Selects the action based on the probability distribution

        prob_distribution: The distribution to select an action from.

        Returns: The selected action
        """
        n = np.random.random()

        index = 0
        for i in range(len(prob_distribution)):
            if n < prob_distribution[i]:
                index = i
                break
            else:
                n -= prob_distribution[i]

        return index

    def step(self, reward: float, next_state: "ndarray", done: bool) -> int:
        """
        Performs the agent's part of the interaction loop. This involves selecting an action and recording the agent inputs
        into agent memory.

        reward: The reward the agent experiences

        next_state: The next state in the trajectory

        done: The signal that the episode has ended.

        Returns: The action selected by the agent
        """
        prob_distribution, p_input_vectors, p_pre_activation_vectors = self.policy_function_predict(self.current_state)

        self.episode_probs.append(prob_distribution)
        self.episode_layer_inputs.append(p_input_vectors)
        self.episode_pre_activation_vectors.append(p_pre_activation_vectors)
        self.episode_rewards.append(reward)
        self.episode_states.append(self.current_state)

        action: int = AgentREINFORCE.select_action(prob_distribution)

        
        self.episode_actions.append(action)

        self.current_timestep += 1
        

        if done:
            pass
        else:
            self.current_state = next_state

        return action
