import numpy as np
import math


dtype="float64"

class Utility:
    """
    Defines important functions that are called statically in the rest of the program.
    """

    @staticmethod
    def clone_list(l: "list[ndarray]") -> "list[ndarray]":
        """
        Clones the input list
        """
        r = []

        for i in range(len(l)):
            r.append(np.copy(l[i]))

        return r
    
    @staticmethod
    def clone_list(a: "list", b: "list"):
        """
        Clones list a and places result in b
        """
        for i in range(len(a)):
            b[i] = np.copy(a[i])


    @staticmethod
    def scale_list(l: "list[ndarray]", scalar: float) -> "list[ndarray]":
        """
        Scales input list elements by the input scalar

        Returns the scaled list
        """

        for i in range(len(l)):
            l[i] *= scalar

        return l

    @staticmethod
    def add_list(a: "list[ndarray]", b: "list[ndarray]") -> "list[ndarray]":
        """
        Adds the contents of list a to list b and returns b
        """
        for i in range(len(b)):
            b[i] += a[i]

        return b

    @staticmethod
    def clear_list(l: "list[ndarray]") -> "list[ndarray]":
        """
        Sets each list element to a zero array of the same shape and returns it
        """

        for i in range(len(l)):
            l[i] = np.zeros(l[i].shape, dtype=dtype)

        return l
    
    @staticmethod
    def softmax(x: "ndarray"):
        """
        Performs the softmax activation function on the input array.

        Returns the probability distribution
        """

        sum: float = 0

        for i in range(len(x)):
            sum += math.exp(x[i])

        y: "ndarray" = np.zeros(len(x), dtype=dtype)

        for i in range(len(x)):
            y[i] = math.exp(x[i]) / sum

        return y
    
    @staticmethod
    def tanh(x: "ndarray") -> "ndarray":
        """
        Performs the tanh activation function on the input array
        """
        y: "ndarray" = np.zeros(len(x), dtype=dtype)

        for i in range(len(y)):
            y[i] = math.tanh(x[i])

        return y
    
    @staticmethod
    def tanh_prime(x: "ndarray") -> "ndarray":
        """
        Performs the tanh activation function derivative on the input array
        """
        y: "ndarray" = np.zeros(len(x), dtype=dtype)

        for i in range(len(y)):
            t: float = math.tanh(x[i])
            y[i] = 1 - (t * t)

        return y
    
    @staticmethod
    def hidden_activation(x: "ndarray") -> "ndarray":
        """
        Performs the hidden layer activation function
        """
        return Utility.tanh(x)

    @staticmethod
    def hidden_activation_prime(x: "ndarray") -> "ndarray":
        """
        Performs the hidden layer activation function derivative
        """
        return Utility.tanh_prime(x)
    
    
    @staticmethod
    def max_abs_value(x: "list[ndarray]") -> float:
        """
        Gets the largest absolute value of all components in the input array
        """

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
        """
        Gets the euclidean length of the input vector (not the number of components)
        """

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
        """
        Gets the vector length of the difference between the a and b vectors
        """

        negative_b = Utility.scale_list(b, -1)

        delta = Utility.add_list(a, negative_b)

        return Utility.get_vector_length(delta)
    
    @staticmethod
    def init_weights_xavier_uniform(w: "ndarray"):
        """
        Initializes the input weight matrix with uniform Xavier initialization.
        """

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
        """
        Initializes the input list with uniform Xavier initialization.
        """

        for i in range(len(l)):
            Utility.init_weights_xavier_uniform(l[i])
    
    @staticmethod
    def get_list_shapes(l: "list[ndarray]") -> "list[Tuple]":
        """
        Returns the list of each element in the input array
        """
        r = []

        for i in range(len(l)):

            r.append(l[i].shape)

        return r

    @staticmethod
    def create_new_params(current_params, epsilon: float):
        """
        Returns new parameters that have small differences with the input parameters

        current_params: The parameters that will be added to to create the new params. This input is not modified.

        epsilon: Defines the boundaries for the changes to current params. Values uniformily distributed in [-epsilon, epsilon]
        will be added to current_params and the new array is returned.
        """
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