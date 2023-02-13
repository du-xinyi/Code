import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

std = (0.1307,)
mean = (0.3081,)

# 模型训练和参数优化
epoch = 5 # 迭代次数
batch_size = 64 # 训练批次大小

Loss = [] # 损失函数
Accuracy = [] # 准确率

# 数据变换
pipline_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), #随机旋转图片
    # transforms.Resize((32,32)),  #将图片尺寸resize到32x32
    transforms.ToTensor(), #将图片转化为Tensor格式
    transforms.Normalize(std,mean) #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
])
pipline_test = transforms.Compose([
    # transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(std,mean)
])

# 导入数据集
data_train = datasets.MNIST(root="./",
                            transform=pipline_train,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./",
                           transform=pipline_test,
                           train=False,
                           download=True)

# 数据装载
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=batch_size,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=batch_size,
                                               shuffle=True)

# 数据预览
# images, labels = next(iter(data_loader_train))
#
# img = torchvision.utils.make_grid(images)
# img = img.numpy().transpose(1, 2, 0)
#
# img = img * std + mean
# plt.imshow(img)
# # plt.pause(1) # PyCharm专业版中可以通过开启Python Scientific使程序在显示图像时继续运行，其它的需要将plt.show()改成plt.pause(1)
# plt.show()

# for i in range(batch_size):
#     print(format(labels[i]), " ", end='')
#     if (i + 1) % (batch_size) ** 0.5 == 0:
#         print("\n")

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 1) # 1*28*28 -> 12*28*28
        self.maxpool1 = nn.MaxPool2d(2) # 12*28*28 -> 12*14*14
        self.conv2 = nn.Conv2d(12, 16, 5) # 12*14*14 -> 16*10*10
        self.maxpool2 = nn.MaxPool2d(2) # 16*10*10 -> 16*5*5
        self.conv3 = nn.Conv2d(16, 120, 5) # 16*5*5 ->120*1*1

        self.fc1 = nn.Linear(120 * 1 * 1, 84)
        self.fc2 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = x.view(-1, 120 * 1 * 1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 5) # 1*28*28 -> 30*24*24
        self.maxpool1 = nn.MaxPool2d(2) # 30*24*24 -> 30*12*12
        self.conv2 = nn.Conv2d(30, 15, 3) # 30*12*12 -> 15*10*10
        self.maxpool2 = nn.MaxPool2d(2) # 15*10*10 -> 15*5*5

        self.fc1 = nn.Linear(15*5*5, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 10)

        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = F.dropout(x, p=0.4)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = F.dropout(x, p=0.4)
        x = self.flatten(x)
        x = x.view(-1, 15*5*5)
        x = torch.relu(self.fc1(x))
        x = F.dropout(x, p=0.4)
        x = torch.relu(self.fc2(x))
        x = F.dropout(x, p=0.4)
        x = self.fc3(x)
        output = torch.log_softmax(x, dim=1)
        return output

# 判断环境
if(torch.cuda.is_available()):
    device = torch.device("cuda")
    print("使用GPU训练：{}".format(torch.cuda.get_device_name()))
else:
    device = torch.device("cpu")
    print("使用CPU训练")


model = Model()
model = model.to(device) #模型转移

optimizer = optim.Adam(model.parameters(),lr=1e-2) # 定义优化器

# 查看搭建好的模型结构
# print(model)

# 模型训练
def train_runner(model, device, trainloader, optimizer, epoch):
    model.train()  # 训练模型, 启用 BatchNormalization 和 Dropout
    total = 0
    correct = 0.0

    for i, data in enumerate(trainloader, 0):  # enumerate迭代已加载的数据集,同时获取数据和数据下标
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 把模型部署到device上

        # 初始化梯度
        optimizer.zero_grad()

        # 保存训练结果
        outputs = model(inputs)

        # 计算损失和
        loss = F.cross_entropy(outputs, labels) # 多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题通常使用sigmod

        # 获取最大概率的预测结果
        predict = outputs.argmax(dim=1) # dim=1表示返回每一行的最大值对应的列下标
        total += labels.size(0)
        correct += (predict == labels).sum().item()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        if i % 1000 == 0:
            print(
                "Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".
                format(epoch, loss.item(), 100 * (correct / total))) # loss.item()表示当前loss的数值
            Loss.append(loss.item())
            Accuracy.append(correct / total)
    return loss.item(), correct / total

# 模型验证
def test_runner(model, device, testloader):
    model.eval()

    #统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0

    with torch.no_grad(): # torch.no_grad将不会计算梯度, 也不会进行反向传播
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, label).item()
            predict = output.argmax(dim=1)

            #计算正确数量
            total += label.size(0)
            correct += (predict == label).sum().item()

        #计算损失值
        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/total, 100*(correct/total)))

for epoch in range(1, epoch + 1):
    print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    loss, acc = train_runner(model, device, data_loader_train, optimizer, epoch)
    Loss.append(loss)
    Accuracy.append(acc)
    test_runner(model, device, data_loader_test)
    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')

print('Finished Training')
plt.subplot(2, 1, 1)
plt.plot(Loss)
plt.title('Loss')
plt.show()
plt.subplot(2, 1, 2)
plt.plot(Accuracy)
plt.title('Accuracy')
plt.show()

# 保存模型参数
torch.save(model.state_dict(), 'mnist.pt')
# 保存完整模型
# torch.save(model, 'mnist.pt')

# 验证模型的准确率
data, label = next(iter(data_loader_test))
data, label = data.to(device), label.to(device)
inputs = Variable(data)
pred = model(inputs)

_, pred = torch.max(pred, 1)

print('Predict Label is:')
for i in range(len(pred.data)):
    print(format(pred.data[i].cpu()), " ", end=' ') # 不转换的话打印时会出现device='cuda:0'
    if (i + 1) % (batch_size ** 0.5) == 0:
        print("\n")

print('Real Label is:')
for i in range(len(label)):
    print(format(label.data[i].cpu()), " ", end=' ')
    if (i + 1) % (batch_size ** 0.5) == 0:
        print("\n")


img = torchvision.utils.make_grid(data)
img = img.cpu().numpy().transpose(1, 2, 0) # cuda不能直接转换成numpy

img = img * std + mean
plt.imshow(img)
plt.show()

test_correct = 0
for i in range(len(pred)):
    if pred.data[i] == label.data[i]:
        test_correct += 1
print('test_correct:{:.4f}%'.format(100 * test_correct / len(pred)))