import numpy as np

class KOSUF:
    """
    Defines Kth Order State Update Functions
    """

    def __init__(self, k: int, obs_size: int, action_count: int):
        """
        Initializes the KO-SUF
        """
        
        self.k = k
        self.obs_size = obs_size
        self.action_count = action_count
        self.frame_size = obs_size + action_count + 1
        self.queue = []
        self.state_size = self.frame_size * self.k

        self.dtype = "float64"

        self.episode_end_reset()
        
    def update_function_predict(self, obs, action: int, reward: float) -> "ndarray":
        """
        Implementation of the State Update Function.

        Returns the new state
        """

        action_one_hot = np.zeros(self.action_count, dtype=self.dtype)
        action_one_hot[action] = 1
        reward_array = np.array([reward], dtype=self.dtype)


        self.queue.append(np.concatenate((obs, action_one_hot, reward_array)))

        self.trim_queue()

        return np.concatenate(self.queue, dtype=self.dtype)

    
    def get_start_state(self, obs: "ndarray") -> "ndarray":
        """
        Gets the starting state. Places dummy values for previous observation, reward, and action

        Returns the starting state
        """

        self.fill_queue()

        action_one_hot = np.zeros(self.action_count, dtype=self.dtype) - 1
        reward = np.array([-1], dtype=self.dtype)

        self.queue.append(np.concatenate((obs, action_one_hot, reward), dtype=self.dtype))

        self.trim_queue()

        return np.concatenate(self.queue, dtype=self.dtype)
    
    def trim_queue(self):
        """
        Removes elements in the queue such that the size stays equal to k
        """

        while len(self.queue) > self.k:
            self.queue.pop(0)

    def fill_queue(self):
        """
        Add dummy values to the queue such that the size stays equal to k.
        """
        
        while len(self.queue) < self.k:
            self.queue.append(np.zeros(self.frame_size, dtype=self.dtype) - 1)
    
    def episode_end_reset(self):
        """
        Performs end of episode actions. In this case, clears the queue and fills it with dummy values
        """
        self.queue.clear()
        self.fill_queue()

    def step(self, observation: "ndarray", action: int, reward: float, done: bool):
        """
        Performs the SUFs part of the interaction loop.

        obs: The new observation
        action: The previous action
        reward: The previous reward
        done: The signal that indicates if the episode is complete.

        Returns the new state, reward, and the done signal
        """
        next_state = self.update_function_predict(observation, action, reward)

        if done:
            self.episode_end_reset()

        return next_state, reward, done