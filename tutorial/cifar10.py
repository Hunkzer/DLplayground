import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 20, 3, padding=1)
        self.conv4 = nn.Conv2d(20, 20, 3, padding=1)
        self.conv5 = nn.Conv2d(20, 20, 3, padding=1)
        self.conv6 = nn.Conv2d(20, 20, 3, padding=1)
        # Pooling Layers
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected Layers
        self.fc1 = nn.Linear(20 * 4 * 4, 10)

    def forward(self, x):
        # 32*32*3
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        # 16*16*16
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        # 8*8*20
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        # 4*4*20
        # Fully connected layer
        x = x.view(-1, 20 * 4 * 4)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(device, net, trainloader, lossfn, optimizer, scheduler):
    for epoch in range(20):  # loop over the dataset multiple times
        # Print Learning Rate
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = lossfn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every x mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i * trainloader.batch_size + 1 * trainloader.batch_size, running_loss / 1000))
                running_loss = 0.0
        # Decay Learning Rate
        scheduler.step()
    print('Finished Training')


def test(testloader, net, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    class_accuracies = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        class_accuracies.append(100 * class_correct[i] / class_total[i])
        print('Accuracy of %5s : %2d %%' % (classes[i], class_accuracies[len(class_accuracies)-1]))
    print("Mean Accuracy over 10 classes: {}".format(np.mean(class_accuracies)))


def main():
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                              shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=6)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("len(trainset) == {}".format(len(trainset)))
    print("len(testset) == {}".format(len(testset)))
    net = Net()
    # net = models.vgg16(pretrained=True)
    net.to(device)
    print(net)
    lossfn = nn.CrossEntropyLoss()
    # Adam with beta1 = 0.9,
    # beta2 = 0.999, and learning_rate = 1e-3 or 5e-4
    # is a great starting point for many models!
    # Adam = Momentum + bias correction + adagrad/RMSProp
    # Adam = momentum + preventing overshooting
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    train(device, net, trainloader, lossfn, optimizer, scheduler)
    test(testloader, net, classes, device)


if __name__ == '__main__':
    main()
    # 72.0 20 epochs
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    # https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
