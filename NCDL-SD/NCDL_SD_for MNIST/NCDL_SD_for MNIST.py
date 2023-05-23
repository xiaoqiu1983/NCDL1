'''
torch gpu版本 1.9.0
torchvision 0.10.0
'''

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torchvision import utils as vutils

from torch.quantization import convert, prepare_qat, quantize_dynamic, get_default_qat_qconfig
from torch.quantization import QuantStub, DeQuantStub

mse_loss = nn.MSELoss()

# 通过给模型实例增加一个名为"qconfig"的成员变量实现量化方案的指定
# backend目前支持fbgemm和qnnpack
BACKEND = "fbgemm"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # float -> int
        self.quant = QuantStub()
        # int -> float
        self.dequant = DeQuantStub()
        
        ## 连接层都是全连接来做
        self.hidden2 = nn.Sequential(nn.Linear(in_features=392,out_features=64,bias=True),nn.ReLU())
        self.hidden3 = nn.Sequential(nn.Linear(in_features=392,out_features=64,bias=True),nn.ReLU())
        self.hidden4 = nn.Sequential(nn.Linear(in_features=64,out_features=32,bias=True),nn.ReLU())
        self.hidden5 = nn.Sequential(nn.Linear(in_features=32,out_features=64,bias=True),nn.ReLU())
        self.hidden6 = nn.Sequential(nn.Linear(in_features=64,out_features=784,bias=True),nn.ReLU())
        self.hidden7 = nn.Sequential(nn.Linear(in_features=64,out_features=784,bias=True),nn.ReLU())

    def forward(self, x):
        ## x_[in, out][number]的命名方式针对提供文档的标号，对每个标号中的输入输出都是直接命名化
        ## 如 6 的输入是 x_in6 = torch.cat((x_in6_1, x_in6_2), 1), 为两个输入的连接

        x = self.quant(x)

        x_in3 = x[:, :392]
        x_in2 = x[:, 392:]
        x_out3 = self.hidden3(x_in3)
        x_out2 = self.hidden2(x_in2)
        x_in7_1 = x_out3[:, :32] ## 3的输出分为两个输出
        x_in4_1 = x_out3[:, 32:]
        x_in6_2 = x_out2[:, 32:]
        x_in4_2 = x_out2[:, :32]
        x_in4 = torch.cat((x_in4_1, x_in4_2), 1) 
        x_out4 = self.hidden4(x_in4)
        x_in5 = x_out4
        x_out5 = self.hidden5(x_in5)
        x_in7_2 = x_out5[:, :32]
        x_in6_1 = x_out5[:, 32:]
        x_in6 = torch.cat((x_in6_1, x_in6_2), 1) 
        x_in7 = torch.cat((x_in7_1, x_in7_2), 1) 
        x_out6 = self.hidden6(x_in6)
        x_out7 = self.hidden6(x_in7)
        
        x_out6 = self.dequant(x_out6)
        x_out7 = self.dequant(x_out7)

        return x_out6, x_out7


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        # 根据batchsize reshape为 (1, 784)
        # data = data.reshape(64, 1,-1)
        shape1,shape2,shape3,shape4 = data.shape
        data = data.reshape(shape1, -1)
        # target为点对点的loss
        data, target = data.to(device), data.to(device)

        optimizer.zero_grad()
        output1, output2 = model(data)
        #loss = F.nll_loss(output, target)
        loss1 = mse_loss(output1, target)
        loss2 = mse_loss(output2, target)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            # output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            shape1,shape2,shape3,shape4 = data.shape
            data = data.reshape(shape1, -1)
            data, target = data.to(device), data.to(device)
            output1, output2 = model(data)
            loss1 = mse_loss(output1, target).item()
            loss2 = mse_loss(output2, target).item()
            test_loss += loss1
            test_loss += loss2

            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)
    # 输出的test_loss是所有数据的loss
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor()])
        #transforms.Normalize((0.1307,), (0.3081,))
        #])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    model.qconfig = get_default_qat_qconfig(BACKEND)

    # 插入伪量化模块
    prepare_qat(model, inplace=True)
    print("model with observers:")
    print(model)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn_qt.pt")

def predict():
    device = torch.device("cpu")
    model = Net().to(device)
    model.qconfig = get_default_qat_qconfig(BACKEND)

    # 插入伪量化模块
    prepare_qat(model, inplace=True)
    model.load_state_dict(torch.load('./mnist_cnn_qt.pt'))

    model.eval()

    model = convert(model)
    print("quantized model:")
    print(model)

    torch.backends.quantized.engine = BACKEND

    ## 可以查看相关权重
    '''
    import pdb
    pdb.set_trace()
    # 量化后的权重
    model.hidden2[0].weight().int_repr()
    '''

    transform=transforms.Compose([
        transforms.ToTensor()])

    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    test_kwargs = {'batch_size': 64}
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    
    for data, target in test_loader:
        shape1,shape2,shape3,shape4 = data.shape
        data = data.reshape(shape1, -1)
        data, target = data.to(device), target.to(device)

        output1, output2 = model(data)

        ## ouput1和output2的输出shape为 【batchsize, 728】
        ## target[0] 为7，保存target[0]的测试样例看看
        ## 原始 7 样本
        vutils.save_image(data[0].reshape(28,28), "ori_{:}.jpg".format(target[0]))
        ## output1 7 样本
        vutils.save_image(output1[0].reshape(28,28), "out1_{:}.jpg".format(target[0]))
        ## output2 7 样本
        vutils.save_image(output2[0].reshape(28,28), "out2_{:}.jpg".format(target[0]))
        break
        
    

if __name__ == '__main__':
    ## 训练& 保存模型
    main()
    ## 预测& 保存示例
    # predict()
