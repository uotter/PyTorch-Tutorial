"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape=(100, 1)
x_left = torch.unsqueeze(torch.linspace(-1, 0, 500), dim=1)
x_right = torch.unsqueeze(torch.linspace(0, 1, 500), dim=1)
y = x.pow(2) + (0.2 * torch.rand(x.size()) - 0.5)  # noisy y data (tensor), shape=(100, 1)
y_right = x_right.pow(2) + (0.2 * torch.rand(x_right.size()) - 0.5)
y_left = x_left.pow(2) + (0.2 * torch.rand(x_left.size()) - 0.5)

x2 = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y2 = 4 * x2.pow(2) + (0.2 * torch.rand(x2.size()) - 0.5)  # noisy y data (tensor), shape=(100, 1)

x3 = torch.unsqueeze(torch.linspace(-2, 0, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y3 = 2 * x3.pow(2) + 4 * x3 + (0.2 * torch.rand(x3.size()) - 0.5)  # noisy y data (tensor), shape=(100, 1)


# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, name):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.fix_params_num = 0
        for p in self.parameters():
            self.fix_params_num += 1
            if name == "net3" or name == "net2":
                p.requires_grad = False
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)  # linear output
        return x


# x_train1, y_train1 = x_right, y_right
# x_train2, y_train2 = x_left, y_left
# x_test, y_test = x, y

x_train1, y_train1 = x, y
x_train2, y_train2 = x3, y3
x_test, y_test = x3, y3

# x_train1, y_train1 = x, y
# x_train2, y_train2 = x2, y2
# x_test, y_test = x2, y2
total_count = 10
test_name = "center_move"
loss1_total_list = []
loss2_total_list = []
loss3_total_list = []
LR = 0.01
for j in range(total_count):

    net1 = Net(n_feature=1, n_hidden=10, n_output=1, name="net1")  # define the network
    print(net1)  # net architecture

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net1.parameters()), lr=LR, betas=(0.9, 0.99))
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    loss1_list = []
    for t in range(1000):
        prediction = net1(x_train1)  # input x and predict based on x

        loss = loss_func(prediction, y_train1)  # must be (1. nn output, 2. target)
        loss1_list.append(loss.data.numpy())
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
    prediction = net1(x_test)
    plt.figure(figsize=(12, 12), dpi=100)
    plt.subplot(221)
    sc1 = plt.scatter(x_train1.data.numpy(), y_train1.data.numpy(), c="blue")
    sc2 = plt.scatter(x_train2.data.numpy(), y_train2.data.numpy(), c="green")
    line1, = plt.plot(x_test.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
    # plt.legend(handles=[line1], labels=['net1 - original network'], loc='best')
    plt.legend(handles=[line1, sc1, sc2], labels=['test', 'trained', 'not trained'], loc='best')
    plt.title("net1 - original network")

    torch.save(net1, "../models/net1.pkl")
    fix_params_num = net1.fix_params_num
    net2 = Net(n_feature=1, n_hidden=10, n_output=1, name="net2")
    dict_trained = torch.load("../models/net1.pkl").state_dict().copy()
    dict_new = net2.state_dict().copy()
    new_list = list(net2.state_dict().keys())
    trained_list = list(dict_trained.keys())
    print(new_list)
    print(trained_list)
    for i in range(fix_params_num):
        dict_new[new_list[i]] = dict_trained[trained_list[i]]
    net2.load_state_dict(dict_new)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net2.parameters()), lr=LR, betas=(0.9, 0.99))
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    loss2_list = []
    for t in range(1000):
        prediction = net2(x_train2)  # input x and predict based on x

        loss = loss_func(prediction, y_train2)  # must be (1. nn output, 2. target)
        loss2_list.append(loss.data.numpy())
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
    prediction = net2(x_test)
    plt.subplot(222)
    sc1 = plt.scatter(x_train1.data.numpy(), y_train1.data.numpy(), c="blue")
    sc2 = plt.scatter(x_train2.data.numpy(), y_train2.data.numpy(), c="green")
    line2, = plt.plot(x_test.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
    # plt.legend(handles=[line2], labels=['net2 - load net1 and fix the first two layes'], loc='best')
    plt.legend(handles=[line2, sc1, sc2], labels=['test', 'not trained', 'trained'], loc='best')
    plt.title("net2 - load net1 and fix the first two layes")

    net3 = Net(n_feature=1, n_hidden=10, n_output=1, name="net3")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net3.parameters()), lr=LR, betas=(0.9, 0.99))
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    loss3_list = []
    for t in range(1000):
        prediction = net3(x_train2)  # input x and predict based on x

        loss = loss_func(prediction, y_train2)  # must be (1. nn output, 2. target)
        loss3_list.append(loss.data.numpy())
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
    prediction = net3(x_test)
    plt.subplot(223)
    sc1 = plt.scatter(x_train1.data.numpy(), y_train1.data.numpy(), c="blue")
    sc2 = plt.scatter(x_train2.data.numpy(), y_train2.data.numpy(), c="green")
    line3, = plt.plot(x_test.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
    # plt.legend(handles=[line3], labels=['net3 - only fix the first two layes'], loc='best')
    plt.legend(handles=[line3, sc1, sc2], labels=['test', 'not trained', 'trained'], loc='best')
    plt.title("net3 - fix the first two layes without loading pretrained models")

    plt.subplot(224)
    line4, = plt.plot(np.array(range(1000)), loss1_list, 'r-', lw=2)
    line5, = plt.plot(np.array(range(1000)), loss2_list, 'b-', lw=2)
    line6, = plt.plot(np.array(range(1000)), loss3_list, 'g-', lw=2)
    loss1_total_list.append(loss1_list)
    loss2_total_list.append(loss2_list)
    loss3_total_list.append(loss3_list)
    plt.legend(handles=[line4, line5, line6], labels=['net1', 'net2', 'net3'], loc='best')
    plt.title("loss compare")
    plt.savefig('../images/{}_{}'.format(test_name, j) + '.png')
    plt.show()

plt.figure(figsize=(8, 8), dpi=100)
loss1_mean_list = np.mean(np.array(loss1_total_list), axis=0)
loss2_mean_list = np.mean(np.array(loss2_total_list), axis=0)
loss3_mean_list = np.mean(np.array(loss3_total_list), axis=0)
line4, = plt.plot(np.array(range(1000)), loss1_mean_list, 'r-', lw=2)
line5, = plt.plot(np.array(range(1000)), loss2_mean_list, 'b-', lw=2)
line6, = plt.plot(np.array(range(1000)), loss3_mean_list, 'g-', lw=2)
plt.legend(handles=[line4, line5, line6], labels=['net1', 'net2', 'net3'], loc='best')
plt.title("mean loss compare")
plt.savefig('../images/{}_mean'.format(test_name) + '.png')
plt.show()
