import numpy as np
from neural_network import NeuralNetwork

if __name__ == "__main__":
    xor_nn = NeuralNetwork([2, 2, 1])

    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    expected = np.array([0, 1, 1, 0])

    xor_nn.learn(inputs, expected, learning_rate=1, epochs=200, batch_size=700)
    print(xor_nn.feed_forward(inputs[0]))
    print(xor_nn.feed_forward(inputs[1]))
    print(xor_nn.feed_forward(inputs[2]))
    print(xor_nn.feed_forward(inputs[3]))