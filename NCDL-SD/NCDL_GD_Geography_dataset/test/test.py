'''
torch gpu版本 1.9.0
torchvision 0.10.0
改成单通道的tiny数据集，看看效果。
'''
# 网络结构换成cnn或者transformer
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import random
from torchvision import utils as vutils
import matplotlib.pyplot as plt
# import torch.utils.data.Dataset as DataSet
from torch.utils.data import Dataset
import torchvision
import numpy as np
from PIL import Image

trans = transforms.ToTensor()

mse_loss = nn.MSELoss()

### Constants ###
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SHAPE = (64, 64)

NB_EPOCHS = 1000
BATCH_SIZE = 32



# a = plt.imread('/home/dazhi/network_coding_复现请教/data/train/n01443537/images/n01443537_0.JPEG')

def load_dataset_small(num_images_per_class_train=10):
    """Loads training and test datasets, from Tiny ImageNet Visual Recogition Challenge.

    Arguments:
        num_images_per_class_train: number of images per class to load into training dataset.
        num_images_test: total number of images to load into training dataset.
    """

    X_train = []
    X_test = []

    # Create training set.
    for c in os.listdir("./data/train"):
        c_dir = os.path.join("./data/train", c, 'images')
        c_imgs = os.listdir(c_dir)
        random.shuffle(c_imgs)
        for img_name_i in c_imgs[0:num_images_per_class_train]:
        # for img_name_i in c_imgs:
        #     img_i = image.load_img(os.path.join(c_dir, img_name_i))
        #     img_i = plt.imread(os.path.join(c_dir, img_name_i))
            # img_i = plt.open(os.path.join(c_dir, img_name_i))
            img_i = Image.open(os.path.join(c_dir, img_name_i))
            img_i = img_i.convert('RGB')
            # img_i = trans(img_i)
            img_i = trans(transforms.Resize(IMG_SHAPE)(img_i))
            # img_i = plt.imread(os.path.join(c_dir, img_name_i)).astype(np.float32)# image是keras下面的
            # if img_i.shape == (64, 64):
            #    print (os.path.join(c_dir, img_name_i))
            #    os.remove(os.path.join(c_dir, img_name_i))
               # continue
            # x = image.img_to_array(img_i)
            # x = np.array(img_i)
            x = img_i
            # x = x[:10, :100, :3]  # 截取一部分像素
            X_train.append(x)
    random.shuffle(X_train)


    # Return train and test data as numpy arrays.
    # return np.array(X_train), np.array(X_test)
    return np.array(X_train)
    # return X_train


#写一个自己的数据类
class MyDataSet(Dataset): #继承DataSet类
    def __init__(self,num_images_per_class_train):
        super(MyDataSet, self).__init__()
        self.num_images_per_class_train = num_images_per_class_train
        data = load_dataset_small(num_images_per_class_train)

        # data = data[:,:,:, 0]  # 变成单通道的
        # data = data.reshape(data.shape[0], 512, 512, 1)
        # data = data / 255 # 格式默认uint8，做数据归一化（预处理阶段）会导致精度缺失，https://blog.csdn.net/qq_34914551/article/details/88943807
        # data = data.astype(np.float32)  #https://www.cnblogs.com/yibeimingyue/p/13935821.html
        self.len = data.shape[0]
        self.x_data = data
        self.y_data = data
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    	#注意:想返回几个参数自己定，比如我可以return data, self.x_data[index],self.y_data[index]供三个参数，如果你含有颜色，曲率、对应关系等数据，都可以在这里返回
    def __len__(self):
        return self.len



# #针对的是28*28的数据集
# class Net(nn.Module):   #  model(data) Net(data)
#     def __init__(self):
#         super(Net, self).__init__()
#
#         ## 连接层都是全连接来做，激活函数用ReLU，严格按照论文的方式
#         self.hidden2 = nn.Sequential(nn.Linear(in_features=392,out_features=64,bias=True),nn.ReLU())
#         self.hidden3 = nn.Sequential(nn.Linear(in_features=392,out_features=64,bias=True),nn.ReLU())
#         self.hidden4 = nn.Sequential(nn.Linear(in_features=64,out_features=32,bias=True),nn.ReLU())
#         self.hidden5 = nn.Sequential(nn.Linear(in_features=32,out_features=64,bias=True),nn.ReLU())
#         self.hidden6 = nn.Sequential(nn.Linear(in_features=64,out_features=784,bias=True),nn.ReLU())
#         self.hidden7 = nn.Sequential(nn.Linear(in_features=64,out_features=784,bias=True),nn.ReLU())
#
#     def forward(self, x):
#         ## x_[in, out][number]的命名方式针对提供文档的标号，对每个标号中的输入输出都是直接命名化
#         ## 如 6 的输入是 x_in6 = torch.cat((x_in6_1, x_in6_2), 1), 为两个输入的连接
#         x_in3 = x[:, :392]
#         x_in2 = x[:, 392:]
#         x_out3 = self.hidden3(x_in3)
#         x_out2 = self.hidden2(x_in2)
#         x_in7_1 = x_out3[:, :32] ## 3的输出分为两个输出
#         x_in4_1 = x_out3[:, 32:]
#         x_in6_2 = x_out2[:, 32:]
#         x_in4_2 = x_out2[:, :32]
#         x_in4 = torch.cat((x_in4_1, x_in4_2), 1)
#         x_out4 = self.hidden4(x_in4)
#         x_in5 = x_out4
#         x_out5 = self.hidden5(x_in5)
#         x_in7_2 = x_out5[:, :32]
#         x_in6_1 = x_out5[:, 32:]
#         x_in6 = torch.cat((x_in6_1, x_in6_2), 1)
#         x_in7 = torch.cat((x_in7_1, x_in7_2), 1)
#         x_out6 = self.hidden6(x_in6)
#         x_out7 = self.hidden6(x_in7)
#
#         return x_out6, x_out7

# #针对64*64的tiny数据集更改
# class Net(nn.Module):  # model(data) Net(data)
#     def __init__(self):
#         super(Net, self).__init__()
#
#         ## 连接层都是全连接来做，激活函数用ReLU，严格按照论文的方式
#         self.hidden2 = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), nn.ReLU())
#         self.hidden3 = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), nn.ReLU())
#         self.hidden4 = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True), nn.ReLU())
#         self.hidden5 = nn.Sequential(nn.Linear(in_features=256, out_features=512, bias=True), nn.ReLU())
#         self.hidden6 = nn.Sequential(nn.Linear(in_features=512, out_features=4096, bias=True), nn.ReLU())
#         self.hidden7 = nn.Sequential(nn.Linear(in_features=512, out_features=4096, bias=True), nn.ReLU())
#         self.hidden = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True), nn.ReLU())
#
#     def forward(self, x):
#         ## x_[in, out][number]的命名方式针对提供文档的标号，对每个标号中的输入输出都是直接命名化
#         ## 如 6 的输入是 x_in6 = torch.cat((x_in6_1, x_in6_2), 1), 为两个输入的连接
#         x_in3 = x[:, :2048]
#         x_in2 = x[:, 2048:]
#         x_out3 = self.hidden3(x_in3)
#         x_out2 = self.hidden2(x_in2)
#
#         x_out3 = self.hidden(x_out3)
#         x_out2 = self.hidden(x_out2)
#
#
#
#         x_in7_1 = x_out3[:, :256]  ## 3的输出分为两个输出
#         x_in4_1 = x_out3[:, 256:]
#         x_in6_2 = x_out2[:, 256:]
#         x_in4_2 = x_out2[:, :256]
#         x_in4 = torch.cat((x_in4_1, x_in4_2), 1)
#
#         x_in4 = self.hidden(x_in4)
#         x_in4 = self.hidden(x_in4)
#
#
#         x_out4 = self.hidden4(x_in4)
#         x_in5 = x_out4
#         x_out5 = self.hidden5(x_in5)
#         x_in7_2 = x_out5[:, :256]
#         x_in6_1 = x_out5[:, 256:]
#         x_in6 = torch.cat((x_in6_1, x_in6_2), 1)
#         x_in7 = torch.cat((x_in7_1, x_in7_2), 1)
#
#
#
#         x_in6 = self.hidden(x_in6)
#         x_in7 = self.hidden(x_in7)
#
#         # x_in6 = self.hidden(x_in6)
#         # x_in7 = self.hidden(x_in7)
#         #
#         # x_in6 = self.hidden(x_in6)
#         # x_in7 = self.hidden(x_in7)
#         #
#         # x_in6 = self.hidden(x_in6)
#         # x_in7 = self.hidden(x_in7)
#
#
#
#
#         x_out6 = self.hidden6(x_in6)
#         x_out7 = self.hidden6(x_in7)
#
#
#         return x_out6, x_out7
#




# #再来一个尝试
# #针对64*64的tiny数据集更改
# class Net(nn.Module):  # model(data) Net(data)
#     def __init__(self):
#         super(Net, self).__init__()
#
#         ## 连接层都是全连接来做，激活函数用ReLU，严格按照论文的方式
#         self.hidden2 = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), nn.ReLU())
#         self.hidden3 = nn.Sequential(nn.Linear(in_features=2048, out_features=512, bias=True), nn.ReLU())
#         self.hidden4 = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True), nn.ReLU())
#         self.hidden5 = nn.Sequential(nn.Linear(in_features=256, out_features=512, bias=True), nn.ReLU())
#         self.hidden6 = nn.Sequential(nn.Linear(in_features=512, out_features=4096, bias=True), nn.ReLU())
#         self.hidden7 = nn.Sequential(nn.Linear(in_features=512, out_features=4096, bias=True), nn.ReLU())
#         self.hidden = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True), nn.ReLU())
#
#     def forward(self, x):
#         ## x_[in, out][number]的命名方式针对提供文档的标号，对每个标号中的输入输出都是直接命名化
#         ## 如 6 的输入是 x_in6 = torch.cat((x_in6_1, x_in6_2), 1), 为两个输入的连接
#         x_in3 = x[:, :2048]
#         x_in2 = x[:, 2048:]
#         x_out3 = self.hidden3(x_in3)
#         x_out2 = self.hidden2(x_in2)
#         #
#         # x_out3 = self.hidden(x_out3)
#         # x_out2 = self.hidden(x_out2)
#         #
#
#
#         x_in7_1 = x_out3[:, :256]  ## 3的输出分为两个输出
#         x_in4_1 = x_out3[:, 256:]
#         x_in6_2 = x_out2[:, 256:]
#         x_in4_2 = x_out2[:, :256]
#         x_in4 = torch.cat((x_in4_1, x_in4_2), 1)
#
#         # x_in4 = self.hidden(x_in4)
#         # x_in4 = self.hidden(x_in4)
#
#
#         x_out4 = self.hidden4(x_in4)
#         x_in5 = x_out4
#         x_out5 = self.hidden5(x_in5)
#         x_in7_2 = x_out5[:, :256]
#         x_in6_1 = x_out5[:, 256:]
#         x_in6 = torch.cat((x_in6_1, x_in6_2), 1)
#         x_in7 = torch.cat((x_in7_1, x_in7_2), 1)
#
#
#
#
#
#         # x_in6 = self.hidden(x_in6)
#         # x_in7 = self.hidden(x_in7)
#         #
#         # x_in6 = self.hidden(x_in6)
#         # x_in7 = self.hidden(x_in7)
#         #
#         # x_in6 = self.hidden(x_in6)
#         # x_in7 = self.hidden(x_in7)
#
#
#
#
#         x_out6 = self.hidden6(x_in6)
#         x_out7 = self.hidden6(x_in7)
#
#
#         return x_out6, x_out7



# #试试另外一种
# class Net(nn.Module):   #  model(data) Net(data)
#     def __init__(self):
#         super(Net, self).__init__()
#
#         ## 连接层都是全连接来做，激活函数用ReLU，严格按照论文的方式
#         self.hidden2 = nn.Sequential(nn.Linear(in_features=2048,out_features=64,bias=True),nn.ReLU())
#         self.hidden3 = nn.Sequential(nn.Linear(in_features=2048,out_features=64,bias=True),nn.ReLU())
#         self.hidden4 = nn.Sequential(nn.Linear(in_features=64,out_features=32,bias=True),nn.ReLU())
#         self.hidden5 = nn.Sequential(nn.Linear(in_features=32,out_features=64,bias=True),nn.ReLU())
#         self.hidden6 = nn.Sequential(nn.Linear(in_features=64,out_features=4096,bias=True),nn.ReLU())
#         self.hidden7 = nn.Sequential(nn.Linear(in_features=64,out_features=4096,bias=True),nn.ReLU())
#
#     def forward(self, x):
#         ## x_[in, out][number]的命名方式针对提供文档的标号，对每个标号中的输入输出都是直接命名化
#         ## 如 6 的输入是 x_in6 = torch.cat((x_in6_1, x_in6_2), 1), 为两个输入的连接
#         x_in3 = x[:, :2048]
#         x_in2 = x[:, 2048:]
#         x_out3 = self.hidden3(x_in3)
#         x_out2 = self.hidden2(x_in2)
#         x_in7_1 = x_out3[:, :32] ## 3的输出分为两个输出
#         x_in4_1 = x_out3[:, 32:]
#         x_in6_2 = x_out2[:, 32:]
#         x_in4_2 = x_out2[:, :32]
#         x_in4 = torch.cat((x_in4_1, x_in4_2), 1)
#         x_out4 = self.hidden4(x_in4)
#         x_in5 = x_out4
#         x_out5 = self.hidden5(x_in5)
#         x_in7_2 = x_out5[:, :32]
#         x_in6_1 = x_out5[:, 32:]
#         x_in6 = torch.cat((x_in6_1, x_in6_2), 1)
#         x_in7 = torch.cat((x_in7_1, x_in7_2), 1)
#         x_out6 = self.hidden6(x_in6)
#         x_out7 = self.hidden6(x_in7)
#
#         return x_out6, x_out7


#针对的是512*512的数据集
class Net(nn.Module):   #  model(data) Net(data)
    def __init__(self):
        super(Net, self).__init__()

        ## 连接层都是全连接来做，激活函数用ReLU，严格按照论文的方式
        self.hidden2 = nn.Sequential(nn.Linear(in_features=131072,out_features=64,bias=True),nn.ReLU())
        self.hidden3 = nn.Sequential(nn.Linear(in_features=131072,out_features=64,bias=True),nn.ReLU())
        self.hidden4 = nn.Sequential(nn.Linear(in_features=64,out_features=32,bias=True),nn.ReLU())
        self.hidden5 = nn.Sequential(nn.Linear(in_features=32,out_features=64,bias=True),nn.ReLU())
        self.hidden6 = nn.Sequential(nn.Linear(in_features=64,out_features=262144,bias=True),nn.ReLU())
        self.hidden7 = nn.Sequential(nn.Linear(in_features=64,out_features=262144,bias=True),nn.ReLU())

    def forward(self, x):
        ## x_[in, out][number]的命名方式针对提供文档的标号，对每个标号中的输入输出都是直接命名化
        ## 如 6 的输入是 x_in6 = torch.cat((x_in6_1, x_in6_2), 1), 为两个输入的连接
        x_in3 = x[:, :131072]
        x_in2 = x[:, 131072:]
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
        data, target = data.to(device), data.to(device)#不使用dataset里的label，重新规定label为数据本身，但是pytorch的dataset里必须给出label

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
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
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
    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)

    train_dataset = MyDataSet(400)
    # test_dataset = MyDataSet(400)
    test_dataset = train_dataset

    train_loader = torch.utils.data.DataLoader(train_dataset,  **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,  **test_kwargs)

    model = Net().to(device)#这个时候因为Net()的初始化没有其他参数，所以就不传，但是像“Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)”就得传入，
                            #因为init里有encoder, decoder, device，所以就得传入。但是这里没有，也自动将上面的train_loader， test_loader。传不传参数都有这个功能。
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)#这里的model是已经经过上面一句话train更改过权重系数的model
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "network_code_nnc4.pt")
# #针对64*64的
# def predict():
#     device = torch.device("cuda")
#     model = Net().to(device)
#     model.load_state_dict(torch.load('./mnist_cnn1.pt'))
#     transform=transforms.Compose([
#         transforms.ToTensor()])
#
#     predict_dataset = MyDataSet(400)
#     test_kwargs = {'batch_size': 64}
#     test_loader = torch.utils.data.DataLoader(predict_dataset, **test_kwargs)
#
#
#     model.eval()
#     for data, target in test_loader:
#         shape1,shape2,shape3,shape4 = data.shape
#         data = data.reshape(shape1, -1)
#         data, target = data.to(device), target.to(device)
#         output1, output2 = model(data)
#         ## ouput1和output2的输出shape为 【batchsize, 728】
#         ## target[0] 为7，保存target[0]的测试样例看看
#         ## 原始 7 样本
#         vutils.save_image(data[0].reshape(64,64), "原始{:}.jpg".format('tiny'))#https://m.php.cn/article/471817.html
#         ## output1 7 样本
#         vutils.save_image(output1[0].reshape(64,64), "out1_{:}.jpg".format('tiny'))
#         ## output2 7 样本
#         vutils.save_image(output2[0].reshape(64,64), "out2_{:}.jpg".format('tiny'))
        

#针对28*28
def predict():
    device = torch.device("cuda")
    model = Net().to(device)
    model.load_state_dict(torch.load('./network_code_nnc4.pt'))
    transform=transforms.Compose([
        transforms.ToTensor()])

    predict_dataset = MyDataSet(400)
    test_kwargs = {'batch_size': 64}
    test_loader = torch.utils.data.DataLoader(predict_dataset, **test_kwargs)


    model.eval()
    for data, target in test_loader:
        shape1,shape2,shape3,shape4 = data.shape
        data = data.reshape(shape1, -1)
        data, target = data.to(device), target.to(device)
        output1, output2 = model(data)
        ## ouput1和output2的输出shape为 【batchsize, 728】
        ## target[0] 为7，保存target[0]的测试样例看看
        ## 原始 7 样本
        vutils.save_image(data[0].reshape(512,512), "原始{:}.jpg".format('tiny'))#https://m.php.cn/article/471817.html
        ## output1 7 样本
        vutils.save_image(output1[0].reshape(512,512), "out1_{:}.jpg".format('tiny'))
        ## output2 7 样本
        vutils.save_image(output2[0].reshape(512,512), "out2_{:}.jpg".format('tiny'))



if __name__ == '__main__':
    ## 训练& 保存模型
    main()
    # 预测& 保存示例
    # predict()

