import numpy as np

class KOSUF:

    def __init__(self, k: int, obs_size: int, action_count: int):
        self.k = k
        self.obs_size = obs_size
        self.action_count = action_count
        self.frame_size = obs_size + action_count + 1
        self.queue = []
        self.state_size = self.frame_size * self.k

        self.dtype = "float64"

        self.episode_end_reset()
        
    def update_function_predict(self, obs, action: int, reward: float) -> "ndarray":

        action_one_hot = np.zeros(self.action_count, dtype=self.dtype)
        action_one_hot[action] = 1
        reward_array = np.array([reward], dtype=self.dtype)


        self.queue.append(np.concatenate((obs, action_one_hot, reward_array)))

        self.trim_queue()

        return np.concatenate(self.queue, dtype=self.dtype)

    
    def get_start_state(self, obs: "ndarray") -> "ndarray":
        self.fill_queue()

        action_one_hot = np.zeros(self.action_count, dtype=self.dtype) - 1
        reward = np.array([-1], dtype=self.dtype)

        self.queue.append(np.concatenate((obs, action_one_hot, reward), dtype=self.dtype))

        self.trim_queue()

        return np.concatenate(self.queue, dtype=self.dtype)
    
    def trim_queue(self):

        while len(self.queue) > self.k:
            self.queue.pop(0)

    def fill_queue(self):
        
        while len(self.queue) < self.k:
            self.queue.append(np.zeros(self.frame_size, dtype=self.dtype) - 1)
    
    def episode_end_reset(self):
        self.fill_queue()

    def step(self, observation: "ndarray", action: int, reward: float, done: bool):
        next_state = self.update_function_predict(observation, action, reward)

        if done:
            self.episode_end_reset()

        return next_state, reward, done