import numpy as np


def sigmoid_activation(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def linear_activation(x: np.ndarray) -> np.ndarray:
    return x

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def mse(predictions, targets):
    return (np.square(predictions - targets)).mean()

class NeuralNetwork:
    def __init__(self, arch: "list[int]"):
        self.biases = [np.array([0])] + [np.random.normal(0, 1, arch[i])/5 for i in range(1, len(arch))]
        self.weights = [np.array([0])] + [np.random.normal(0, 1, (arch[i], arch[i+1]))/2 for i in range(len(arch)-1)]

        # self.weights[1][0][0] = -3
        # self.weights[1][0][1] = 6
        # self.weights[1][1][0] = 1
        # self.weights[1][1][1] = -2

        # self.weights[2][0][0] = 8
        # self.weights[2][1][0] = 4

        self.y = [np.array([0])] + [np.zeros(arch[i], dtype=np.float32) for i in range(1, len(arch))]
        self.error = [np.array([0])] + [np.zeros(arch[i], dtype=np.float32) for i in range(len(arch)-1)]

        self.gradient_b = [np.array([0])] + [np.full(arch[i], 0, dtype=np.float32) for i in range(1, len(arch))]
        self.gradient_w = [np.array([0])] + [np.full((arch[i], arch[i+1]), 1, dtype=np.float32) for i in range(len(arch)-1)]

    def feed_forward(self, input: np.ndarray) -> np.ndarray:
        if len(input.shape) == 1 and input.shape[0] != self.weights[1].shape[0]:
            print("ERROR: input doesn't have the correct shape wanted:",
                  self.weights[1].shape[0])

        # the first layer is the input layer so it's output is just the input unchanged
        self.y[0] = input

        for i in range(1, len(self.weights)):
            self.y[i] = self.y[i-1]@self.weights[i] + self.biases[i]
            self.y[i] = sigmoid_activation(self.y[i])
        
        return self.y[-1]

    def back_propagate(self, expected: np.ndarray, learning_rate = 0.5):
        # last layer index
        L = len(self.error) - 1

        if len(expected.shape) == 1 and expected.shape[0] != self.y[L].shape[0]:
            print("ERROR: input doesn't have the correct shape wanted:",
                  self.weights[0].shape[0])

        self.error[L] = (self.y[L]-expected)

        np.outer(self.y[L-1], self.error[L], out=self.gradient_w[L])
        self.gradient_b[L] = self.error[L]

        for l in range(L-1, 0, -1):
            self.error[l] = self.y[l]*(1-self.y[l])*(self.error[l+1]@self.weights[l+1].T)
            
            np.outer(self.y[l-1], self.error[l], out=self.gradient_w[l])
            self.gradient_b[l] = self.error[l]

        # update the parameters
        for i in range(1, len(self.gradient_w)):
           np.add(self.weights[i], -learning_rate*self.gradient_w[i], out=self.weights[i])
           np.add(self.biases[i], -learning_rate*self.gradient_b[i], out=self.biases[i])

    def learn(self, inputs: "list[np.ndarray]", expected: "list[np.ndarray]", learning_rate = 0.001, epochs = 500,
              batch_size = 500):
        # cost_print_interval = 1

        for epoch in range(epochs):
            average_cost = 0.0
            for i in np.random.choice(len(inputs), batch_size):
                X = inputs[i]
                predicted = self.feed_forward(X)
                self.back_propagate(expected[i], learning_rate)

            #     if epoch % cost_print_interval == 0:
            #         average_cost += cross_entropy(predicted, expected[i])
            
            # if epoch % cost_print_interval == 0:    
            #     print(f"Cost {average_cost/batch_size:.5f} Epoch {epoch}")
        




    def print_biases(self, values=False, shapes=True):
        print("Biases")
        for i in range(1, len(self.biases)):
            if values:
                print(self.biases[i], end="")

            if shapes:
                print(self.biases[i].shape, end="")
            print("")
        print("")

    def print_weights(self, values=False, shapes=True):
        print("Weights")
        for i in range(1, len(self.weights)):
            if values:
                print(self.weights[i], end="")

            if shapes:
                print(self.weights[i].shape, end="")
            print("")
        print("")

    def print_y(self, values=True, shapes=False):
        print("Y")
        for i in range(len(self.y)):
            if values:
                print(self.y[i], end="")

            if shapes:
                print(self.y[i].shape, end="")
            print("")
        print("")

    def print_errors(self):
        print("Errors")
        for i in range(1, len(self.error)):
            print(self.error[i])
        print("\n")

    def print_grad_w(self):
        print("Grad_Weights")
        for i in range(1, len(self.gradient_w)):
            print(self.gradient_w[i])
        print("\n")

if __name__ == "__main__":
    xorNet = NeuralNetwork([2, 2, 1])
    xorNet.print_biases()
    xorNet.print_weights()

    # xorNet.feed_forward(np.array([1, 1], dtype=np.float32))
    # xorNet.print_y()

    xorNet.feed_forward(np.array([2, 6], dtype=np.float32))
    xorNet.print_y()

    xorNet.back_propagate(np.array([0], dtype=np.float32))
    xorNet.print_errors()
    xorNet.print_weights(values=True)
    xorNet.print_grad_w()
