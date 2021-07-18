# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from src.mnist import load_mnist, make_batch
from src.three_layer_net import ThreeLayerNet
from src.CNN import CNN
from src.optimizer import *


# Hyper-parameters
EPOCHS = 30
BATCHSIZE = 100
LR = 0.01

# 0:Load MNIST data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_batch = make_batch(x_train, t_train, BATCHSIZE)
test_batch = make_batch(x_test, t_test, BATCHSIZE)

network = CNN(D = 32, hidden_size=300, output_size=10)
optimizer = Adam(lr=LR)

train_loss_list = []
train_acc_list = []
test_acc_list = []

def test(network, data_batch):
    print("testtt")
    correct_total = 0
    total = 0
    for x_batch, t_batch in data_batch:
        x_batch = x_batch.reshape(-1, 1, 28, 28)   
        correct = network.accuracy(x_batch, t_batch)
        total += x_batch.shape[0]
        correct_total += correct

    acc = correct_total/total
    return acc

for i in range(EPOCHS):
    step =0
    for x_batch, t_batch in train_batch:
        if step%100==0:
            print(step)
        step += 1
        x_batch = x_batch.reshape(-1, 1, 28, 28)   
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

    train_acc = test(network, train_batch)
    test_acc = test(network, test_batch)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print("epoch:" + str(i) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))




# '''# Plot the dataset
# n_data = 10
# row = 2
# col = 5
# fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(8, 6))

# fig.suptitle("MNIST data-set", fontsize=24, color='white')
# for i, img in enumerate(x_train[:n_data]):
#     _r = i // col
#     _c = i % col
#     ax[_r,_c].set_title(t_train[i], fontsize=16, color='black')
#     ax[_r,_c].axes.xaxis.set_visible(False)
#     ax[_r,_c].axes.yaxis.set_visible(False)
#     ax[_r,_c].imshow(img.reshape(28, 28), cmap='Greys')

# plt.show()
# plt.close()
# '''
# plotting

markers = {'train': 'o', 'test': 's'}
x = np.arange(EPOCHS)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

'''
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
from three_layer_net import ThreeLayerNet
from optimizer import *


# 0:Load MNIST data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# 1:Settings
network = ThreeLayerNet(input_size=784, hidden_size=100, output_size=10)
optimizer = SGD(lr=0.01)
max_epochs = 301
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(600*50):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# plotting
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
'''