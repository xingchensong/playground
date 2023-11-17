#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-17] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


input_size = 784  # img_size = (28,28) ---> 28*28=784 in total
hidden_size = 512  # number of nodes at hidden layer
num_classes = 10  # number of output classes discrete range [0,9]
num_epochs = 20  # number of epochs
batch_size = 100  # the size of input data took for one iteration
lr = 1e-3  # size of step


train_data = dsets.MNIST(root='./assets/data', train=True,
                         transform=transforms.ToTensor(), download=True)

test_data = dsets.MNIST(root='./assets/data', train=False,
                        transform=transforms.ToTensor())


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


train_gen = torch.utils.data.DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True)

test_gen = torch.utils.data.DataLoader(dataset=test_data,
                                       batch_size=batch_size,
                                       shuffle=False)

net = Net(input_size, hidden_size, num_classes)
if torch.cuda.is_available():
    net.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_gen):
        images = Variable(images.view(-1, 28*28)).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                    epoch+1, num_epochs, i + 1,
                    len(train_data) // batch_size, loss.data)
            )

correct = 0
total = 0
for images, labels in test_gen:
    images = Variable(images.view(-1, 28*28)).cuda()
    labels = labels.cuda()

    output = net(images)
    _, predicted = torch.max(output, 1)
    correct += (predicted == labels).sum()
    total += labels.size(0)

print('Accuracy of the model: %.3f %%' % ((100*correct) / (total+1)))

print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

torch.save(net.state_dict(), "./assets/mnist_model.state_dict")
