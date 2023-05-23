'''
torch gpu版本 1.9.0
torchvision 0.10.0
改成单通道的tiny数据集，看看效果。
'''

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

import numpy as np
# from Image import Image

from PIL import Image
trans = transforms.ToTensor()

mse_loss = nn.MSELoss()
# mse_loss = nn.CrossEntropyLoss()


### Constants ###
DATA_DIR = "./test/data/train/n01443537/"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SHAPE = (28, 28)###########################################32*32
input_size = int(IMG_SHAPE[0] * IMG_SHAPE[1] / 2)
output_size = IMG_SHAPE[0] * IMG_SHAPE[1]

# NB_EPOCHS = 1000
BATCH_SIZE = 100
data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize([IMG_SHAPE[0],IMG_SHAPE[1]]), transforms.ToTensor()])############################归化到28*28


def load_dataset_small(num_images_per_class_train=10):
    """Loads training and test datasets, from Tiny ImageNet Visual Recogition Challenge.

    Arguments:
        num_images_per_class_train: number of images per class to load into training dataset.
        num_images_test: total number of images to load into training dataset.
    """

    X_train = []
    X_test = []

    # Create training set.
    for c in os.listdir(TRAIN_DIR):
        c_img = os.path.join(TRAIN_DIR, c)####################
        img = Image.open(c_img)
        img = data_transform(img)
        X_train.append(img)
        '''
        c_dir = os.path.join(TRAIN_DIR, c)
        c_imgs = os.listdir(c_dir)
        random.shuffle(c_imgs)
        for img_name_i in c_imgs[0:num_images_per_class_train]:
            img_i = Image.open(os.path.join(c_dir, img_name_i))

            img_i = img_i.convert('RGB')
            # w, h = img_i.size

            img_i = trans(transforms.Resize(IMG_SHAPE)(img_i))

            x = img_i
            X_train.append(x)
        '''
    random.shuffle(X_train)


    # Return train and test data as numpy arrays.
    # return np.array(X_train), np.array(X_test)
    X_train = torch.tensor([item.cpu().detach().numpy() for item in X_train]).cuda()#https://blog.csdn.net/weixin_40740309/article/details/114700259
    X_train = X_train.cpu()
    # X_train = torch.Tensor(X_train)


    return np.array(X_train)
    # return X_train


#写一个自己的数据类     vutils.save_image(data.reshape(137, 93), "原始{:}.jpg".format('测试'))
class MyDataSet(Dataset): #继承DataSet类
    def __init__(self,num_images_per_class_train):
        super(MyDataSet, self).__init__()
        self.num_images_per_class_train = num_images_per_class_train
        data = load_dataset_small(num_images_per_class_train)

        #data = data / 255  # 格式默认uint8，做数据归一化（预处理阶段）会导致精度缺失，https://blog.csdn.net/qq_34914551/article/details/88943807
        self.len = data.shape[0]
        self.x_data = data
        self.y_data = data
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    	#注意:想返回几个参数自己定，比如我可以return data, self.x_data[index],self.y_data[index]供三个参数，如果你含有颜色，曲率、对应关系等数据，都可以在这里返回
    def __len__(self):
        return self.len



#为predict单独写一个dataset，主要是没有data = data / 255
class My_predict_DataSet(Dataset): #继承DataSet类
    def __init__(self,num_images_per_class_train):
        super(My_predict_DataSet, self).__init__()
        self.num_images_per_class_train = num_images_per_class_train
        data = load_dataset_small(num_images_per_class_train)

        # data = data / 255  # 格式默认uint8，做数据归一化（预处理阶段）会导致精度缺失，https://blog.csdn.net/qq_34914551/article/details/88943807
                             #是否用这条语句我不太确定。
                             #如果用，那相对的loss损失会很低，不过因为要测试需要恢复图片，无论是直接从nnrray数组格式恢复的vutils.save_image函数，还是把读取vutils.save_image函数产生的图片然后再次产生图片，
                             # 都需要不使用本条语句，使用了都度不出来，不使用，上述两种恢复图片的方法都可以。所以就不是用本条语句，不知道对整个模型的精度是否有影响，不使用，对本数据集训练loss大约是1100，测试loss是64
                             # 恢复结果都不好，看来还是模型有问题。怎么办呢？看来需要改变网络结构，搞成VIT的？不会搞。
        self.len = data.shape[0]
        self.x_data = data
        self.y_data = data
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    	#注意:想返回几个参数自己定，比如我可以return data, self.x_data[index],self.y_data[index]供三个参数，如果你含有颜色，曲率、对应关系等数据，都可以在这里返回
    def __len__(self):
        return self.len


#针对的是512*512的数据集

#input_size = 392
#output_size = 784 # 28*28
class Net(nn.Module):   #  model(data) Net(data)
    def __init__(self):
        super(Net, self).__init__()

        ## 连接层都是全连接来做，激活函数用ReLU，严格按照论文的方式
        self.hidden2 = nn.Sequential(nn.Linear(in_features=input_size,out_features=64,bias=True),nn.ReLU())
        self.hidden3 = nn.Sequential(nn.Linear(in_features=input_size,out_features=64,bias=True),nn.ReLU())
        self.hidden4 = nn.Sequential(nn.Linear(in_features=64,out_features=32,bias=True),nn.ReLU())
        self.hidden5 = nn.Sequential(nn.Linear(in_features=32,out_features=64,bias=True),nn.ReLU())
        self.hidden6 = nn.Sequential(nn.Linear(in_features=64,out_features=output_size,bias=True),nn.ReLU())
        self.hidden7 = nn.Sequential(nn.Linear(in_features=64,out_features=output_size,bias=True),nn.ReLU())

    def forward(self, x):
        ## x_[in, out][number]的命名方式针对提供文档的标号，对每个标号中的输入输出都是直接命名化
        ## 如 6 的输入是 x_in6 = torch.cat((x_in6_1, x_in6_2), 1), 为两个输入的连接
        x_in3 = x[:, :input_size]
        x_in2 = x[:, input_size:]
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
        data = data.to(torch.float64)###########################模型容易取捕捉前景
        data = torch.where(data<0.95, data, 0.)
        data = data.to(torch.float32)###############################
        shape1,shape2,shape3,shape4 = data.shape
        data = data.reshape(shape1, -1)
        # target为点对点的loss
        data, target = data.to(device), data.to(device)#不使用dataset里的label，重新规定label为数据本身，但是pytorch的dataset里必须给出label

        optimizer.zero_grad()
        output1, output2 = model(data)

        '''
        vutils.save_image(target[0].reshape(3, 64, 64), "target_train_{:}.jpg".format(batch_idx))#测试一下原始图片用这句话是否可以显示出图片，尺寸变形了。用'batch_idx'就会产生最终的一个图片，会变形，如果用batch_idx，会循环产生图片。
                                                                                                #不过能正确显示的前提是dataset里不能有data = data / 255这句话，但是没有这句话，训练的loss就会很大。
        vutils.save_image(output1[0].reshape(3, 64, 64), "out1_train_{:}.jpg".format('batch_idx'))#看看训练阶段的图片恢复质量，此时loss很低，但是存储模型之后不知道为什么loss很高，恢复质量不行。
        original = Image.open(os.path.join('./', 'out1_train_batch_idx.jpg'))
        original = transforms.Resize([93, 137])(original)#训练阶段读取的图片原始尺寸是不一样的，这里变形为第一张训练图片0.png的原始尺寸，根据之前对函数的测试，如何精度不损失，大致也可以显示，只是会长宽变形一点。
        original.save('out1_train_batch_idx_重新读取变形后.jpg')
        '''

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




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=40.0, metavar='LR',
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
                       'pin_memory': False,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor()])
        #transforms.Normalize((0.1307,), (0.3081,))
        #])
    train_dataset = MyDataSet(9923)#一共9922张图片
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
        #test(model, device, test_loader)#这里的model是已经经过上面一句话train更改过权重系数的model
        scheduler.step()

    #if args.save_model:
        if epoch==100:
            torch.save(model.state_dict(), "./models/100.pt")


#测试


def predict():
    device = torch.device("cuda")
    model = Net().to(device)
    model.load_state_dict(torch.load('./models/100.pt'))
    transform=transforms.Compose([
        transforms.ToTensor()])

    # predict_dataset = MyDataSet(400)
    # test_kwargs = {'batch_size': 64}
    predict_dataset = My_predict_DataSet(1)#测试时就测试0.png，根据它的实际尺寸恢复，把数据集改成就只有0.png即可。先看看它的实际尺寸


    test_kwargs = {'batch_size': 1}
    test_loader = torch.utils.data.DataLoader(predict_dataset, **test_kwargs)


    model.eval()
    for data, target in test_loader:
        data = data.to(torch.float64)
        data = torch.where(data<0.95, data, 0.)
        data = data.to(torch.float32)

        shape1,shape2,shape3,shape4 = data.shape

        data = data.reshape(shape1, -1)
        data, target = data.to(device), target.to(device)


        output1, output2 = model(data)

        ## 原始样本，意外发现，经过各种变化之后，不用变回原始图片尺寸，也可以显示原来图片，只要像素不损失。原来并不是(3,64,64)，也不是[93, 137]，[93, 137]是0.png样本，所以测试时，也可以是用大数据集，而不需要是用只有一张图片的数据集。
        vutils.save_image(data[0].reshape(1,IMG_SHAPE[0],IMG_SHAPE[1]), "ori_{:}.jpg".format('1'))#https://m.php.cn/article/471817.html
        original = Image.open(os.path.join('./', 'ori_1.jpg'))
        original = transforms.Resize([IMG_SHAPE[0], IMG_SHAPE[1]])(original)
        original.save('ori.jpg')
        ## output1  样本
        vutils.save_image(output1[0].reshape(1,IMG_SHAPE[0], IMG_SHAPE[1]), "out1_{:}.jpg".format('1'))
        loss_out1 = mse_loss(output1[0], data[0])
        print('Loss: {:.6f}'.format(loss_out1.item()))
        out1 = Image.open(os.path.join('./', 'out1_1.jpg'))
        out1 = transforms.Resize([IMG_SHAPE[0], IMG_SHAPE[1]])(out1)
        out1.save('out1.jpg')
        ## output2  样本
        vutils.save_image(output2[0].reshape(1,IMG_SHAPE[0],IMG_SHAPE[1]), "out2_{:}.jpg".format('1'))
        loss_out2 = mse_loss(output2[0], data[0])
        print('Loss: {:.6f}'.format(loss_out2.item()))
        out2 = Image.open(os.path.join('./', 'out2_1.jpg'))
        out2 = transforms.Resize([IMG_SHAPE[0], IMG_SHAPE[1]])(out2)
        out2.save('out2.jpg')
        break



if __name__ == '__main__':
    # 训练& 保存模型
    main()
    # 预测 保存示例
    # predict()





