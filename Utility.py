import numpy as np
import math


dtype="float64"

class Utility:

    @staticmethod
    def clone_list(l: "list[ndarray]") -> "list[ndarray]":
        r = []

        for i in range(len(l)):
            r.append(np.copy(l[i]))

        return r
    
    @staticmethod
    def copy_list(a: "list", b: "list"):
        for i in range(len(a)):
            b[i] = np.copy(a[i])


    @staticmethod
    def scale_list(l: "list[ndarray]", scalar: float) -> "list[ndarray]":

        for i in range(len(l)):
            l[i] *= scalar

        return l

    @staticmethod
    def add_list(a: "list[ndarray]", b: "list[ndarray]") -> "list[ndarray]":
        for i in range(len(b)):
            b[i] += a[i]

        return b

    @staticmethod
    def clear_list(l: "list[ndarray]") -> "list[ndarray]":

        for i in range(len(l)):
            l[i] = np.zeros(l[i].shape, dtype=dtype)

        return l
    
    @staticmethod
    def softmax(x: "ndarray"):

        sum: float = 0

        for i in range(len(x)):
            sum += math.exp(x[i])

        y: "ndarray" = np.zeros(len(x), dtype=dtype)

        for i in range(len(x)):
            y[i] = math.exp(x[i]) / sum

        return y
    
    @staticmethod
    def tanh(x: "ndarray") -> "ndarray":
        y: "ndarray" = np.zeros(len(x), dtype=dtype)

        for i in range(len(y)):
            y[i] = math.tanh(x[i])

        return y
    
    @staticmethod
    def tanh_prime(x: "ndarray") -> "ndarray":
        y: "ndarray" = np.zeros(len(x), dtype=dtype)

        for i in range(len(y)):
            t: float = math.tanh(x[i])
            y[i] = 1 - (t * t)

        return y
    
    @staticmethod
    def hidden_activation(x: "ndarray") -> "ndarray":
        return Utility.tanh(x)

    @staticmethod
    def hidden_activation_prime(x: "ndarray") -> "ndarray":
        return Utility.tanh_prime(x)
    
    
    @staticmethod
    def max_abs_value(x: "list[ndarray]") -> float:

        ret: float = 0

        for i in range(len(x)):
            matrix: "ndarray" = x[i]

            if matrix.ndim == 2:
                for j in range(len(matrix)):
                    for k in range(len(matrix[j])):
                        ret = max(ret, matrix[j][k])

            else:
                for j in range(len(matrix)):
                    ret = max(ret, matrix[j])

        return ret
    
    @staticmethod
    def get_vector_length(x: "list[ndarray]") -> float:

        ret: float = 0

        for i in range(len(x)):
            matrix: "ndarray" = x[i]

            if matrix.ndim == 2:
                for j in range(len(matrix)):
                    for k in range(len(matrix[j])):
                        ret += matrix[j][k]**2

            else:
                for j in range(len(matrix)):
                    ret += matrix[j]**2

        return math.sqrt(ret)
    
    @staticmethod
    def get_delta_vector_length(a: "list[ndarray]", b: "list[ndarray]") -> float:
        negative_b = Utility.scale_list(b, -1)

        delta = Utility.add_list(a, negative_b)

        return Utility.get_vector_length(delta)
    
    @staticmethod
    def init_weights_xavier_uniform(w: "ndarray"):

        num_rows: int = len(w)

        num_cols: int = 1

        if w.ndim == 2:
            num_cols = len(w[0])
            

        bounds: float = math.sqrt(6.0 / (num_cols + num_rows))

        for i in range(len(w)):
            if w.ndim == 2:
                for j in range(len(w[i])):
                    w[i][j] = np.random.uniform(-bounds, bounds)

            else:
                w[i] = np.random.uniform(-bounds, bounds)

    @staticmethod
    def init_params_xavier_uniform(l: "list[ndarray]"):

        for i in range(len(l)):
            Utility.init_weights_xavier_uniform(l[i])

    @staticmethod
    def size_of_list_np_array(l: "list[ndarray]", unit_size: int) -> int:

        s = 0

        for i in range(len(l)):
            s += np.prod(l[i].shape) * unit_size

        return int(s)
    
    @staticmethod
    def get_list_shapes(l: "list[ndarray]") -> "list[Tuple]":
        r = []

        for i in range(len(l)):

            r.append(l[i].shape)

        return r

    @staticmethod
    def add_to_queue(queue: list, item: float, max_len=3):
        queue.append(item)

        while len(queue) > max_len:
            queue.pop(0)

    @staticmethod
    def create_new_params(current_params, epsilon: float):
        new_params: "list[ndarray]" = Utility.clone_list(current_params)


        for i in range(len(new_params)):
            if new_params[i].ndim == 2:
                for j in range(len(new_params[i])):
                    for k in range(len(new_params[i][j])):
                        new_params[i][j][k] += np.random.uniform(-epsilon, epsilon)
            else:
                for j in range(len(new_params[i])):
                    new_params[i][j] += np.random.uniform(-epsilon, epsilon)

        return new_params