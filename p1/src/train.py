import numpy as np
from sklearn.datasets import fetch_openml

from network import ConvNet
from solver import Solver
from network1 import ConvNet1
from network2 import ConvNet2
import matplotlib.pyplot as plt


def load_fashion_mnist(flatten=False):
    data = {}
    mnist = fetch_openml(name="Fashion-MNIST")
    X = mnist.data
    y = mnist.target.astype(np.uint8)
    print(X.shape)
    print(y.shape)
    # print(X)

    #######################################################################
    # Optional: you're free to preprocess images here                     #
    #######################################################################
    # pass
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    if not flatten:
        X = X.reshape(X.shape[0], 28, 28)
        X = X[:, np.newaxis, :, :]

    #######################################################################
    # Optional: you're free to adjust the training and val split.         #
    # However, the last 10000 images must be the test set.                #
    #######################################################################
    data['X_train'] = X[:50000]
    data['y_train'] = y[:50000]
    data['X_val'] = X[50000:60000]
    data['y_val'] = y[50000:60000]
    data['X_test'] = X[60000:]
    data['y_test'] = y[60000:]
    return data


def load_toy(flatten=False):
    data = {}
    X = np.load('../data/x.npy')
    y = np.load('../data/y.npy')
    y = y.reshape(y.shape[0],1)
    print('yshape', y.shape)
    # print(X.shape)
    # print(y.shape)

   
    data['X_train'] = X[:700]
    data['y_train'] = y[:700]
    data['X_val'] = X[700:800]
    data['y_val'] = y[700:800]
    data['X_test'] = X[800:]
    data['y_test'] = y[800:]

    return data

def plot_performance(iterations, num_epochs, losses, train_accuracies, val_accuracies):

    plt.xlabel('epoch number')
    plt.ylabel('accuracy')

    plt.plot(num_epochs, train_accuracies, color='blue', linestyle='dashed', label='training_acc')
    plt.plot(num_epochs, val_accuracies, color='yellow', linestyle='dashed', label='val_acc')
    plt.legend(loc='upper left')
    plt.savefig('visualization.png')
    plt.show()

    # plt.ylabel('Purity')
    # plt.xlabel('Number of Clusters, k')

    # plt.plot(k_vals, k_means_scores, color='blue', linestyle='dashed', label='k-means_rand')
    # plt.plot(k_vals, kpp_scores, color='orange', linestyle='dashed', label='k_means_pp')
    # plt.plot(k_vals, spec_scores, color='green', linestyle='dashed', label='spectral')
    # plt.legend(loc='upper left')
    # plt.savefig('problem3.png')


def train():
    # load data
    toy_data = False
    print('loading data')

    if toy_data is False:
        data = load_fashion_mnist()
        print('initializing model')
        model = ConvNet2()

    else:
        data = load_toy()
        print('initializing model')
        model = ConvNet1(input_dim=(1, 1, 100), num_classes=1)

    # intialize solver
    print('initializing solver')
    solver = Solver(model, data, update_rule='sgd',
                    optim_config={'learning_rate': 1e-2,},
                    lr_decay=1.0, num_epochs=10,
                    batch_size=16, print_every=1)

    # start training
    print('starting training')
    iterations, num_epochs, losses, train_accuracies, val_accuracies = solver.train()

    # report test accuracy
    acc = solver.check_accuracy(data['X_test'], data['y_test'])
    print("Test accuracy: {}".format(acc))

    plot_performance(iterations, num_epochs, losses, train_accuracies, val_accuracies)


if __name__=="__main__":
    train()
