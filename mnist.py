import gzip
import numpy as np
import pickle
import random

from neural_network import NeuralNetwork

if __name__ == "__main__":
    mnist_nn = NeuralNetwork([28*28, 30, 10])

    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    training_inputs = np.array(train_set[0][:])

    c = list(zip(training_inputs, train_set[1]))
    training_inputs, training_answers = zip(*c)
    training_answers = np.array(training_answers)
    training_vec = np.zeros((training_answers.size, training_answers.max() + 1))
    training_vec[np.arange(training_answers.size), training_answers] = 1


    mnist_nn.learn(training_inputs, training_vec, learning_rate=0.2, batch_size=600)

