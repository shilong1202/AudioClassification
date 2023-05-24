# -*- coding: utf-8 -*-
"""
Created on ${DATE} ${TIME}
@author:
"""
import torch
import torchvision

# prepare
model1_vgg16 = torchvision.models.vgg16(pretrained=True)

# 设置网络参数
# 读取输入特征的维度
num_fc = model1_vgg16.classifier[6].in_features
# 修改最后一层的输出维度，即分类数
model1_vgg16.classifier[6] = torch.nn.Linear(num_fc, 2)

# 固定权值参数
# 对于模型的每个权重，使其不进行反向传播，即固定参数
for param in model1_vgg16.parameters():
    param.requires_grad = False
# 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
for param in model1_vgg16.classifier[6].parameters():
    param.requires_grad = True

# 训练模型
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable

# 加载图片
##这种读取图片的方式用的是Torch自带的ImageFloder,读取的文件夹必须在一个大的子文件下，按类别归好类


# train data
train_data = torchvision.datasets.ImageFolder("../input/Spirals/testing", transform=transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))

train_loader = DataLoader(train_data, batch_size=20, shuffle=True)

# test data
test_data = torchvision.datasets.ImageFolder("../input/Spirals/testing", transform=transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]))

test_loader = DataLoader(test_data, batch_size=20, shuffle=True)

# 训练
## 定义自己的优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1_vgg16.parameters(), lr=0.001)

# 训练模型
EPOCH = 20
for epoch in range(EPOCH):
    train_loss = 0.
    train_acc = 0.
    for step, data in enumerate(train_loader):
        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        # batch_y not one hot
        # out is the probability of eatch class
        # such as one sample[-1.1009  0.1411  0.0320],need to calculate the max index
        # out shape is batch_size * class
        out = model1_vgg16(batch_x)
        loss = criterion(out, batch_y)
        train_loss += loss.item()
        # pred is the expect class
        # batch_y is the true label
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Epoch: ', epoch, 'Step', step,
                  'Train_loss: ', train_loss / ((step + 1) * 20), 'Train acc: ', train_acc / ((step + 1) * 20))

    print('Epoch: ', epoch, 'Train_loss: ', train_loss / len(train_data), 'Train acc: ', train_acc / len(train_data))

torch.save(model1_vgg16.state_dict(), '../output/vgg.pth')
# 测试
model1_vgg16.eval()
eval_loss = 0
eval_acc = 0
for step, data in enumerate(test_loader):
    batch_x, batch_y = data
    batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    out = model1_vgg16(batch_x)
    loss = criterion(out, batch_y)
    eval_loss += loss.item()
    # pred is the expect class
    # batch_y is the true label
    pred = torch.max(out, 1)[1]
    test_correct = (pred == batch_y).sum()
    eval_acc += test_correct.item()
print('Test_loss: ', eval_loss / len(test_data), 'Test acc: ', eval_acc / len(test_data))