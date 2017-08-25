# from __future__ import print_function

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def mpi_list_sum(x,y):
    return [a+b for a,b in zip(x,y)]

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default:ls 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(comm_rank)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.batch_size % comm_size != 0:
    msg = "Error: Batch Size must can be divide exactly by worker number."
    raise Exception(msg)


kwargs = {'num_workers': 0, 'pin_memory': False} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
#All reduce the initial model
if comm_size > 1:
    datas = [param.data.numpy() for param in model.parameters()]
    avg_datas = [sum_data / comm_size for sum_data in comm.allreduce(datas, op=mpi_list_sum)]
    for param, avg_data in zip(model.parameters(), avg_datas):
        # param.grad.data = torch.from_numpy(avg_grad).cuda()
        param.data = torch.from_numpy(avg_data)

if args.cuda:
    # model.cuda()
    model.cuda(comm_rank)

optimizer = optim.SGD(model.parameters(), lr=args.lr * comm_size, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #Split data into multiple workers
        if comm_size > 1:
            real_entire_batch_size = data.size()[0]
            real_minibatch_size = int(real_entire_batch_size / comm_size)
            if comm_rank < comm_size - 1:
                data = data[comm_rank * real_minibatch_size : comm_rank * real_minibatch_size + real_minibatch_size]
                target = target[comm_rank * real_minibatch_size : comm_rank * real_minibatch_size + real_minibatch_size]
            else:
                data = data[comm_rank * real_minibatch_size :]
                target = target[comm_rank * real_minibatch_size :]

        if args.cuda:
            # data, target = data.cuda(), target.cuda()
            data, target = data.cuda(comm_rank), target.cuda(comm_rank)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if comm_size > 1:
            grads = [param.grad.data.cpu().numpy() for param in model.parameters()]
            avg_grads = [sum_grad / comm_size for sum_grad in comm.allreduce(grads, op=mpi_list_sum)]
            for param, avg_grad in zip(model.parameters(), avg_grads):
                # param.grad.data = torch.from_numpy(avg_grad).cuda()
                param.grad.data = torch.from_numpy(avg_grad).cuda(comm_rank)
        # return 0
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            # data, target = data.cuda(), target.cuda()
            data, target = data.cuda(comm_rank), target.cuda(comm_rank)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
