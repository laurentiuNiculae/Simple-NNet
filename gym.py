import pygame
import numpy as np
from neural_network import NeuralNetwork
from neural_network import cross_entropy
from neural_network import mse
import gzip
import pickle
import ctypes

ctypes.windll.user32.SetProcessDPIAware()

def gray(im):
    im = 255 * (im / im.max())
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

pygame.init()

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((900,600))
    pygame.display.set_caption('Runner')
    clock = pygame.time.Clock()

    graph_width = 400
    graph_height = 500
    cost_graph = pygame.Surface((graph_width, graph_height))
    cost_graph_rect = cost_graph.get_rect(midleft = (50, screen.get_height()//2))
    pygame.draw.rect(cost_graph, (255, 255, 255), (0, 0, graph_width, graph_height), width = 1)

    # Init neural network XOR
    # xor_nn = NeuralNetwork([2, 10, 10, 1])
    # train_input = np.array([
    #     [0, 0],
    #     [1, 0],
    #     [0, 1],
    #     [1, 1],
    # ])
    # train_answer = np.array([
    #     [1],
    #     [0],
    #     [0],
    #     [0.5],
    # ])

    # -------------------------------------
    # Init Neural Network MNIST
    # mnist_nn = NeuralNetwork([28*28, 36, 10])

    # f = gzip.open('mnist.pkl.gz', 'rb')
    # train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    # train_input = np.array(train_set[0][:])

    # c = list(zip(train_input, train_set[1]))
    # train_input, training_numer_unswers = zip(*c)

    # training_numer_unswers = np.array(training_numer_unswers)

    # train_answer = np.zeros((training_numer_unswers.size, training_numer_unswers.max() + 1))
    # train_answer[np.arange(training_numer_unswers.size), training_numer_unswers] = 1

    # train_input = train_input[:3000]
    # train_answer = train_answer[:3000]

    # -------------------------------------
    interesting_nn = NeuralNetwork([2, 10, 1])

    img = pygame.image.load("nine.jpg")
    img_buff = np.max(pygame.surfarray.array3d(img), axis=2)

    img_rect = img.get_rect(midleft = (50 + graph_width + 50 , screen.get_height()//2))
    screen.blit(img, img_rect)

    x = np.arange(0, 28)
    y = np.arange(0, 28)

    train_input = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    train_answer = img_buff[train_input[:, 0], train_input[:, 1]]

    # normalize
    train_input = train_input / 28
    train_answer = train_answer / 255
    
    # Parameters
    epoch = 0
    epochs= 10000
    learning_rate = 0.02
    cost_print_interval = 10
    cost_plot_points = []
    cost_plot_min = None
    cost_plot_max = None


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        average_cost = 0.0

        #for i in np.random.choice(len(train_input), batch_size):
        for i in range(0, len(train_input), 2):
            X = train_input[i]
            predicted = interesting_nn.feed_forward(X)
            interesting_nn.back_propagate(train_answer[i], learning_rate)

            if epoch % 1 == 0:
                average_cost += cross_entropy(predicted, train_answer[i])
        
        if epoch % 10 == 0:
            cost = average_cost/len(train_input)
            cost_plot_points.append(cost)

        if cost_plot_min is None or cost_plot_min < cost:
            cost_plot_min = cost 

        if cost_plot_max is None or cost_plot_max > cost:
            cost_plot_max = cost 
        
        # Print the cost points
        cost_graph.fill('Black')
        last_point = None
        
        for i, cost in enumerate(cost_plot_points, start=1):
            x = cost_graph.get_width() * (i/len(cost_plot_points))
            y = (cost - cost_plot_min)/(cost_plot_max - cost_plot_min)*cost_graph.get_height()
            if last_point is not None:
                pygame.draw.line(cost_graph, (255,255,255), last_point, (x,y))
                last_point = (x, y)
            else:
                last_point = (x, y)

        out_rez = 50
        output_img_buff = np.zeros((out_rez, out_rez, 3))

        # if epoch % cost_print_interval == -1:
        #     for i in range(out_rez):
        #         for j in range(out_rez):
        #             r = interesting_nn.feed_forward(np.array([i/out_rez, j/out_rez]))
        #             output_img_buff[i, j, 0] = output_img_buff[i, j, 1] = output_img_buff[i, j, 2] = r[0]

        #     output_img_buff = (output_img_buff*255)

        #     output_img = pygame.surfarray.make_surface(output_img_buff)
        #     output_img = pygame.transform.scale(output_img, (250, 250))

        #     out_img_rect = output_img.get_rect(midleft = (50 + graph_width + 50 , screen.get_height()//2))
        #     pygame.draw.rect(output_img, (255, 255, 255), (0 , 0, *output_img.get_size()), width=1)

        #     screen.blit(output_img, out_img_rect)

        screen.blit(cost_graph, cost_graph_rect)

        pygame.display.update()
        epoch += 1
