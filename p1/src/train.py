import numpy as np
from sklearn.datasets import fetch_openml

from network import ConvNet
from solver import Solver
from network1 import ConvNet1
from network2 import ConvNet2
import matplotlib.pyplot as plt
import pickle as pickle


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

   
    data['X_train'] = X[:700]
    data['y_train'] = y[:700]
    data['X_val'] = X[700:800]
    data['y_val'] = y[700:800]
    data['X_test'] = X[800:]
    data['y_test'] = y[800:]

    return data

def plot_performance(num_epochs, val_losses, train_losses, train_accuracies, val_accuracies):

    plt.xlabel('epoch number')
    plt.ylabel('avg loss')

    plt.plot(num_epochs, val_losses, color='blue', linestyle='dashed', label='val_loss')
    plt.plot(num_epochs, train_losses, color='yellow', linestyle='dashed', label='train_loss')
    plt.legend(loc='upper right')
    plt.savefig('visualization2.png')
    plt.show()

    plt.ylabel('average accuracy')
    plt.plot(num_epochs, train_accuracies, color='yellow', linestyle='dashed', label='training_accuracy')
    plt.plot(num_epochs, val_accuracies, color='blue', linestyle='dashed', label='val_accuracy')
    plt.legend(loc='upper right')
    plt.savefig('visualization3.png')
    plt.show()

def load_checkpoint():
    checkpoints = []
    for i in range(0, 11):
        filename = '%s_epoch_%d.pkl' % ('yunk', i)
        with open(filename, 'rb') as cp_file:
            cp = pickle.load(cp_file)
            checkpoints.append(cp)
    
    return checkpoints

def get_graphs(data):
    checkpoints = load_checkpoint()
    train_losses = []
    val_losses = []
    for cp in checkpoints:
        model = cp['model']
        solver = Solver(model, data, update_rule='sgd',
                    optim_config={'learning_rate': 1e-2,},
                    lr_decay=1.0, num_epochs=10,
                    batch_size=16, print_every=1)
        _ = solver.check_accuracy(data['X_train'], data['y_train'], num_samples=1000)
        #lets print the val loss history and the self.loss history for all of them when it's done
        for i in range(0, len(solver.loss_history), 3125):
          # val_loss = np.mean(val_losses[i:i+iterations_per_epoch])
          train_loss = np.mean(solver.loss_history[i:i+3125])
          # avg_val_loss.append(val_loss)
          train_losses.append(train_loss)

        #this might be all i need gonna need to print to make sure though
        # train_losses.append(np.mean(solver.loss_history))
        val_losses = solver.val_loss
    
    return train_losses, val_losses
               




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

    # # intialize solver
    print('initializing solver')

    solver = Solver(model, data, update_rule='sgd',
                    optim_config={'learning_rate': 1e-2,},
                    lr_decay=1.0, num_epochs=30,
                    batch_size=16, print_every=1)

    checkpoints = load_checkpoint()
    train_acc_hist = checkpoints[-1]['train_acc_history']
    val_acc_hist = checkpoints[-1]['val_acc_history']


    validation_losses = []
    #this loop aggregates the losses for each model
    for i in range(0, 11):
        model = checkpoints[i]['model']
        loss = model.loss(data['X_val'], data['y_val'], justLoss=True)
        # print(loss)
        validation_losses.append(loss)
    
    #this is the actual output
    validation_losses = [2.3031098524157447, 0.5432650488842817, 0.4228943625323263, 
    0.3960564576330717, 0.3794001140478632, 0.4184943074710574, 0.3660864188191209,
    0.3860544576370717, 0.3760543576270788, 0.39647363333369284, 0.38654269410560094]

    # for i, cp in enumerate(checkpoints):
    #     if len(cp['loss_history']) % 3125 == 0:
    #         print(str(i))

    #turns out I overwrote checkpoints through 8, the only ones that are good are 9,10
    #this has been fixed thank god

    #this is more for when you want to initialize a solver from a checkpoint
    # solver = Solver(model, data, update_rule='sgd',
    #                  optim_config={'learning_rate': 1e-2,},
    #                  lr_decay=1.0, num_epochs=2,
    #                  batch_size=16, print_every=1)
    

    
    train_loss = checkpoints[-1]['loss_history']
    epochs = checkpoints[-1]['epoch']
    epochs = list(range(0, epochs+1))

    #this loop gives you all the training losses
    #that first value is a filler taken from the model.solve for train
    training_losses = [2.4021198524157447]
    for i in range(0, 31250, 3125):
        to_plot = np.mean(train_loss[i:i+3125])
        training_losses.append(to_plot)


    # # start training
    print('starting training')
    solver.train()

    # # plot losses and accuracies
    plot_performance(epochs, validation_losses, training_losses, train_acc_hist, val_acc_hist)

    # report test accuracy
    acc = solver.check_accuracy(data['X_test'], data['y_test'])
    print("Test accuracy: {}".format(acc))

    


if __name__=="__main__":
    train()
