import os

import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim
import torchvision
import torchvision.datasets as dsets
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F

batch_size = 8
learning_rate = 1e-5
epoches = 10

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 自定义数据集


# 数据集、验证集、检测集路径
trainpath = './corpedImages/HSV/train/'
valpath = './val/'
testpath = './corpedImages/HSV/test/'

# 处理训练集图片
trainTransform = transforms.Compose([
    transforms.RandomCrop([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

# 处理验证集图片
valTransform = transforms.Compose([
    transforms.RandomCrop([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
])

# 处理测试集图片
testTransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
])

trainData = dsets.ImageFolder(trainpath, transform=trainTransform)  # 读取训练集，标签就是train目录下的文件夹的名字，图像保存在格子标签下的文件夹里
testData = dsets.ImageFolder(testpath, transform=testTransform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=False)

# 获取训练集、验证集和测试集的图片数目
train_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(trainpath))])
print('训练集总图片数目：', train_sum)
val_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(valpath))])
print('验证集集总图片数目：', val_sum)
test_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(testpath))])
print('测试集总图片数目：', test_sum)


# 手搓CNN
class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


# cnn = models.resnet18(pretrained=True)
# cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
# cnn.fc = nn.Linear(512, 1)
cnn = RestNet18()
cnn.to(device)

# cnn = RestNet18().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
loss_func = nn.CrossEntropyLoss()

best_val_loss = float('inf')

# training
cnn.train()
for epoch in range(epoches):
    for step, (x, y) in enumerate(trainLoader):
        # 取出一个batch的图片
        # X, Y = next(iter(trainLoader))
        # plt.figure(figsize=(10, 5))
        # plt.imshow(torchvision.utils.make_grid(X, nrow=16).permute([1, 2, 0]))
        # plt.axis("off")
        # plt.show()

        b_x = x.to(device)
        b_y = y.to(device)
        b_y = torch.tensor(b_y, dtype=torch.float)

        train_output = cnn(b_x)
        # print(output)
        # output =output >=0.5
        # output[output >= 0.5] = 1
        # output[output < 0.5] = 0
        train_output = train_output.view(train_output.shape[0])
        # print(output)
        # print(b_y)
        # print(output)
        # import sys
        # sys.exit()
        loss = loss_func(train_output, b_y)
        optimizer.zero_grad()
        loss.backward()  # 在这一步训练清除梯度
        optimizer.step()

        # pred_y = torch.max(train_output, 1)[1].data.squeeze()
        pred_y = train_output
        pred_y[pred_y >= 0.5] = 1
        pred_y[pred_y < 0.5] = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(pred_y.shape[0]):
            if pred_y[i] == 1 and b_y[i] == 1:
                TP += 1
            elif pred_y[i] == 1 and b_y[i] == 0:
                FP += 1
            elif pred_y[i] == 0 and b_y[i] == 1:
                FN += 1
            elif pred_y[i] == 0 and b_y[i] == 0:
                TN += 1
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-20)
        APCER = FP / (TN + FP + 1e-20)
        BPCER = FN / (TP + FN + 1e-20)
        ACER = (APCER + BPCER) / 2
        print('Epoch: ', epoch, '|Step:', step, '| train loss: %.4f' % loss.data, '| train accuracy: %.2f' % accuracy,
              '| ACER: %.2f' % ACER)
    # 更新学习率
    # scheduler.step()
    # torch.save(cnn, './cnn.pth')

# testing
print('正在测试集上进行测试！')
cnn.eval()
TP_test = 0
FP_test = 0
FN_test = 0
TN_test = 0
for batch, (x, y) in enumerate(testLoader):
    # 取出一个batch的图片
    # X, Y = next(iter(testLoader))
    # plt.figure(figsize=(10, 5))
    # plt.imshow(torchvision.utils.make_grid(X, nrow=16).permute([1, 2, 0]))
    # plt.axis("off")
    # plt.show()

    t_x = x.to(device)
    t_y = y.to(device)
    # if step % 100 == 0:
    test_output = cnn(t_x)
    test_output = test_output.view(test_output.shape[0])
    pred_t = test_output
    pred_t[pred_t >= 0.5] = 1
    pred_t[pred_t < 0.5] = 0
    for i in range(pred_t.shape[0]):
        if pred_t[i] == 1 and t_y[i] == 1:
            TP_test += 1
        elif pred_t[i] == 1 and t_y[i] == 0:
            FP_test += 1
        elif pred_t[i] == 0 and t_y[i] == 1:
            FN_test += 1
        elif pred_t[i] == 0 and t_y[i] == 0:
            TN_test += 1
    print('batch:%.2f:' % step, '测试完成！')
accuracy_test = (TP_test + FN_test) / (TP_test + TN_test + FP_test + FN_test + 1e-20)
APCER_test = FP_test / (TN_test + FP_test + 1e-20)
BPCER_test = FN_test / (TP_test + FN_test + 1e-20)
ACER_test = (APCER_test + BPCER_test) / 2
print('test accuracy: %.2f' % accuracy_test, '| APCER_test: %.2f' % APCER_test, '| BPCER_test: %.2f' % BPCER_test,
      '| ACER: %.2f' % ACER_test)
