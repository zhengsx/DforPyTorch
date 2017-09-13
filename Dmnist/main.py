# from __future__ import print_function

import torch

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def mpi_list_sum(x,y):
    return [a+b for a,b in zip(x,y)]

import argparse
# import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np



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
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--optim', type=str, default='ma', choices=['ma', 'dpsgd','dpsgd2', 'super', 'eds', 'try'],
                    help='Parallel Optimization methods: ')
parser.add_argument('--rho', type=float, default=0.1,
                    help='lambda (default: 0.1)')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'],
                    help='Mnist or Fashion Mnist')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(comm_rank)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.batch_size % comm_size != 0:
    msg = "Error: Batch Size must can be divide exactly by worker number."
    raise Exception(msg)

if args.dataset == 'fashion':
    from fashion import fashion

kwargs = {'num_workers': 0, 'pin_memory': False} if args.cuda else {}

if args.dataset == 'fashion':
    train_dataset = fashion(root = '../data/fashion',train=True,download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = fashion(root = '../data/fashion',train=False,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))]))
elif args.dataset == 'mnist':
    train_dataset = datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST('../data/mnist', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size, shuffle=False, **kwargs)

from torch.optim import Optimizer
class mockSGD(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(mockSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(mockSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def step1(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        v=[]
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                v.append(d_p.cpu().numpy())

                # p.data.add_(-group['lr'], d_p)

        return v

    def step2(self, v, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        idx = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # if weight_decay != 0:
                #     d_p.add_(weight_decay, p.data)
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = d_p.clone()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - dampening, d_p)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf

                p.data.add_(-group['lr'], torch.from_numpy(v[idx]).cuda(comm_rank))
                idx = idx+1

        return loss
    


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
        param.data = torch.from_numpy(avg_data)

if args.cuda:
    model.cuda(comm_rank)

optimizer = mockSGD(model.parameters(), lr=args.lr * comm_size if args.optim == 'ma' else args.lr, momentum=args.momentum)
# optimizer = optim.SGD(model.parameters(), lr=args.lr * comm_size if args.optim == 'ma' else args.lr, momentum=args.momentum)

if args.optim == 'eds':
    psis_pre = [param.data.cpu().numpy() for param in model.parameters()]

if args.optim == 'try':
    # gradleft_pre = [np.zeros_like(param.data.cpu().numpy()) for param in model.parameters()]
    # gradright_pre = [np.zeros_like(param.data.cpu().numpy()) for param in model.parameters()]
    paramsleft_pre = [param.data.cpu().numpy() for param in model.parameters()]
    paramsright_pre = [param.data.cpu().numpy() for param in model.parameters()]
    moms_left = [np.zeros_like(param.data.cpu().numpy()) for param in model.parameters()]
    moms_right = [np.zeros_like(param.data.cpu().numpy()) for param in model.parameters()]
    traincnt = 0




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
            data, target = data.cuda(comm_rank), target.cuda(comm_rank)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if comm_size > 1:
            if args.optim == 'ma':
                grads = [param.grad.data.cpu().numpy() for param in model.parameters()]
                avg_grads = [sum_grad / comm_size for sum_grad in comm.allreduce(grads, op=mpi_list_sum)]
                for param, avg_grad in zip(model.parameters(), avg_grads):
                    param.grad.data = torch.from_numpy(avg_grad).cuda(comm_rank)
                optimizer.step()

            elif args.optim == 'dpsgd':
                optimizer.step()
                params = [param.data.cpu().numpy() for param in model.parameters()]
                req_sendleft = comm.isend(params, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                req_sendright = comm.isend(params, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                params_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                params_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                req_sendleft.wait()
                req_sendright.wait()

                avg_params = [(param + param_left + param_right)/3 for param, param_left, param_right in zip(params, params_left, params_right)]

                for param, avg_param in zip(model.parameters(), avg_params):
                    param.data = torch.from_numpy(avg_param).cuda(comm_rank)

            elif args.optim == 'dpsgd2':
                params = [param.data.cpu().numpy() for param in model.parameters()]
                req_sendleft = comm.isend(params, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                req_sendright = comm.isend(params, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                params_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                params_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                req_sendleft.wait()
                req_sendright.wait()

                avg_params = [(param + param_left + param_right)/3 for param, param_left, param_right in zip(params, params_left, params_right)]

                for param, avg_param in zip(model.parameters(), avg_params):
                    param.data = torch.from_numpy(avg_param).cuda(comm_rank)

                optimizer.step()                

            elif args.optim == 'super':
                grads = [param.grad.data.cpu().numpy() for param in model.parameters()]
                req_sendleft = comm.isend(grads, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                req_sendright = comm.isend(grads, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                grads_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                grads_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                req_sendleft.wait()
                req_sendright.wait()

                avg_grads = [(grad + grad_left + grad_right) for grad, grad_left, grad_right in zip(grads, grads_left, grads_right)]

                for param, avg_grad in zip(model.parameters(), avg_grads):
                    param.grad.data = torch.from_numpy(avg_grad).cuda(comm_rank)
                optimizer.step()

            elif args.optim == 'eds':
                global psis_pre
                params_k = [param.data.cpu().numpy() for param in model.parameters()]
                optimizer.step()
                phis_k = [param.data.cpu().numpy() for param in model.parameters()]
                psis_k = [phi_k - param_k + psis_k for phi_k, param_k, psis_k in zip(phis_k, params_k, psis_pre)]
                psis_pre = psis_k

                req_sendleft = comm.isend(psis_k, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                req_sendright = comm.isend(psis_k, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                params_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                params_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                req_sendleft.wait()
                req_sendright.wait()

                avg_params = [(param + param_left + param_right)/3 for param, param_left, param_right in zip(psis_k, params_left, params_right)]

                for param, avg_param in zip(model.parameters(), avg_params):
                    param.data = torch.from_numpy(avg_param).cuda(comm_rank)

            elif args.optim == 'try':
                #--------------------------------momentum term decentralized average together-----------------------------------------------
                vs=optimizer.step1()
                # params = [param.data.cpu().numpy() for param in model.parameters()]
                req_sendleft = comm.isend(vs, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                req_sendright = comm.isend(vs, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                vs_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                vs_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                req_sendleft.wait()
                req_sendright.wait()

                avg_v = [(v + v_left + v_right)/3 for v, v_left, v_right in zip(vs, vs_left, vs_right)]

                # for param, avg_param in zip(model.parameters(), avg_params):
                    # param.data = torch.from_numpy(avg_param).cuda(comm_rank)
                optimizer.step2(avg_v)




                #--------------------------------model average, and add a now-previous model's vector.-----------------------------------------------
                # global paramsleft_pre, paramsright_pre, traincnt, moms_left, moms_right
                # optimizer.step()
                # params = [param.data.cpu().numpy() for param in model.parameters()]

                # req_sendleft = comm.isend(params, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                # req_sendright = comm.isend(params, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                # params_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                # params_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                # req_sendleft.wait()
                # req_sendright.wait()

                # if traincnt == 0:
                #     traincnt = traincnt + 1
                #     avg_params = [(param + param_left + param_right)/3 for param, param_left, param_right in zip(params, params_left, params_right)]
                # else:
                #     avg_params = [(param + param_left + param_right)/3 for param, param_left, param_right in zip(params, params_left, params_right)]
                #     moms_left = [0.9 * mom_left + 0.1 * (param_left - paramleft_pre) for mom_left, param_left, paramleft_pre in zip(moms_left, params_left, paramsleft_pre)]
                #     moms_right = [0.9 * mom_right + 0.1 * (param_right - paramright_pre) for mom_right, param_right, paramright_pre in zip(moms_right, params_right, paramsright_pre)]
                #     # moms_left = [(param_left - paramleft_pre) for param_left, paramleft_pre in zip(params_left, paramsleft_pre)]
                #     # moms_right = [(param_right - paramright_pre) for param_right, paramright_pre in zip(params_right, paramsright_pre)]
                #     avg_params = [avg_param + args.rho * mom_left + args.rho * mom_right for avg_param, mom_left, mom_right in zip(avg_params, moms_left, moms_right)]
                #     paramsleft_pre = params_left
                #     paramsright_pre = params_right

                # for param, avg_param in zip(model.parameters(), avg_params):
                #     param.data = torch.from_numpy(avg_param).cuda(comm_rank)

                #--------------------------------send and add previous gradient-----------------------------------------------
                # global gradleft_pre, gradright_pre
                # grads = [param.grad.data.cpu().numpy() for param in model.parameters()]
                # req_sendleft = comm.isend(grads + gradright_pre, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                # req_sendright = comm.isend(grads + gradleft_pre, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                # grads_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                # grads_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)


                # req_sendleft.wait()
                # req_sendright.wait()

                # avg_grads = [(grad + grad_left + grad_right) for grad, grad_left, grad_right in zip(grads, grads_left, grads_right)]
                # gradleft_pre = grads_left
                # gradright_pre = grads_right
                # for param, avg_grad in zip(model.parameters(), avg_grads):
                #     param.grad.data = torch.from_numpy(avg_grad).cuda(comm_rank)
                # optimizer.step()

                #--------------------------------model geometry average-----------------------------------------------
                # optimizer.step()
                # params = [param.data.cpu().numpy() for param in model.parameters()]
                # req_sendleft = comm.isend(params, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                # req_sendright = comm.isend(params, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                # params_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                # params_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                # req_sendleft.wait()
                # req_sendright.wait()

                # avg_params = [np.cbrt(param*param_left,param_right) for param, param_left, param_right in zip(params, params_left, params_right)]

                # for param, avg_param in zip(model.parameters(), avg_params):
                #     param.data = torch.from_numpy(avg_param).cuda(comm_rank)



                #--------------------------------dcasgd-----------------------------------------------
                # grads = [param.grad.data.cpu().numpy() for param in model.parameters()]
                # req_sendleft = comm.isend(grads, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                # req_sendright = comm.isend(grads, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                # grads_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                # grads_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                # req_sendleft.wait()
                # req_sendright.wait()

                # params = [param.data.cpu().numpy() for param in model.parameters()]
                # req_sendleft = comm.isend(params, dest = (comm_rank - 1) % comm_size, tag = batch_idx)
                # req_sendright = comm.isend(params, dest = (comm_rank + 1) % comm_size, tag = batch_idx)
                # params_left = comm.recv(source = (comm_rank - 1) % comm_size, tag = batch_idx)
                # params_right = comm.recv(source = (comm_rank + 1) % comm_size, tag = batch_idx)

                # req_sendleft.wait()
                # req_sendright.wait()

                # # avg_grads = [(grad + grad_left + grad_right) for grad, grad_left, grad_right in zip(grads, grads_left, grads_right)]
                # avg_grads = [(grad + grad_left + 
                #     args.rho * (grad_left * grad_left) * (param - param_left) + 
                #     grad_right + args.rho * (grad_right * grad_right) * (param - param_right)) 
                # for grad, grad_left, grad_right, param, param_left, param_right in zip(grads, grads_left, grads_right, params, params_left, params_right)]

                # for param, avg_grad in zip(model.parameters(), avg_grads):
                #     param.grad.data = torch.from_numpy(avg_grad).cuda(comm_rank)
                # optimizer.step()

        else:
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    # -------------------additional model average ----------------------------
    # params = [param.data.cpu().numpy() for param in model.parameters()]
    # avg_params = [sum_param / comm_size for sum_param in comm.allreduce(params, op=mpi_list_sum)]
    # for param, avg_param in zip(model.parameters(), avg_params):
    #     param.data = torch.from_numpy(avg_param).cuda(comm_rank)
    

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(comm_rank), target.cuda(comm_rank)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest Epoch:{} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
