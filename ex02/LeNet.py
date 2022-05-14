import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from   torch.autograd import Variable
import matplotlib
from matplotlib import pyplot as plt 

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
plt.rc('font',family='Times New Roman')
'''设计LeNet_5'''
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        '''第一层卷积，卷积核大小为5*5，步距为1，输入通道为3，输出通道为6'''
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=1)

        '''第一层池化层，卷积核为2*2，步距为2，相当于特征图缩小了一半'''
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        '''第二层卷积，卷积核大小为5*5，步距为1，输入通道为6，输出通道为16'''
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        '''第二层池化层，卷积核为2*2，步距为2，相当于特征图缩小了一半'''
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        '''第一层全连接层，维度由16*5*5=>120'''
        self.linear1 = nn.Linear(16 * 4 * 4, 120)

        '''第二层全连接层，维度由120=>84'''
        self.linear2 = nn.Linear(120, 84)

        '''第三层全连接层，维度由84=>10'''
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        '''将数据送入第一个卷积层'''
        out = torch.sigmoid(self.conv1(x))

        '''将数据送入第一个池化层'''
        out = self.pool1(out)

        '''将数据送入第二个卷积层'''
        out = torch.sigmoid(self.conv2(out))

        '''将数据送入第二个池化层'''
        out = self.pool2(out)

        '''将池化层后的数据进行Flatten，使数据变成能够被FC层接受的Vector'''
        out = out.view(-1, 16 * 4 * 4)

        '''将数据送入第一个全连接层'''
        out = torch.sigmoid(self.linear1(out))

        '''将数据送入第二个全连接层'''
        out = torch.sigmoid(self.linear2(out))

        '''将数据送入第三个全连接层得到输出'''
        out = self.linear3(out)

        return out

def main():

    def plot(losses):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)

        ax.plot(range(1,len(losses[0]) + 1), losses[0], label = "5 iters")
        ax.plot(range(1,len(losses[1]) + 1), losses[1], label = "10 iters")

        ax.set_xlabel("iter times",fontsize = 10)
        ax.set_ylabel('loss',fontsize = 10)

        plt.legend()
        plt.savefig(r"C:\Users\lwx\Dropbox\SWOT_simulator\LearnAndTest\ex02\test.png", dpi=300, pad_inches = 0)

    '''定义参数'''
    batch_size = 64
    lr = 0.001
    num_classes = 10

    '''获取数据集'''
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=False)

    losses = []
    for iterTime in [5,10]:
        model = LeNet(num_classes)
        losss = []
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        total_step = len(train_loader)
        for epoch in range(iterTime):
            for i, (images, labels) in enumerate(train_loader):

                images = Variable(images)
                labels = Variable(labels)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, 10, i + 1, total_step, loss.item()))
                losss.append(loss.item())
        losses.append(losss)
    plot(losses)
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('准确率为: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    main()