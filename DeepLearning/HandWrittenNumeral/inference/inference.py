from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from torchvision import transforms

# 定义模型
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化模型
model = Model()

# 初始化优化器
optimizer = optim.Adam(model.parameters(),lr=1e-2)

model.load_state_dict(torch.load("../train/mnist.pt"))
model.eval() #把模型转为test模式
model.to(device)
# print(model)

def where_num(frame):
    rois = []
    # 灰度处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 二次腐蚀处理
    gray2 = cv2.dilate(gray, element)
    # cv2.imshow("dilate", gray2)

    # 二次膨胀处理
    gray2 = cv2.erode(gray2, element)
    gray2 = cv2.erode(gray2, element)
    # cv2.imshow("erode", gray2)

    # 膨胀腐蚀做差
    edges = cv2.absdiff(gray, gray2)
    # cv2.imshow("absdiff", edges)
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    _, ddst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)

    # 轮廓查找
    contours, _ = cv2.findContours(
        ddst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 12 and h > 24:
            rois.append((x, y, w, h))
    return rois

def resize_image(image):
    GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化并翻转为黑底白字
    ret, thresh2 = cv2.threshold(GrayImage, 120, 255, cv2.THRESH_BINARY_INV)

    # 给数字增加一圈黑色方框
    constant = cv2.copyMakeBorder(
        thresh2, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)

    image = cv2.resize(constant, (28, 28))
    return image

trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

# 对OpenCV中的手写数字进行识别
def opencv_test():
    for i in range(10):
        flag = 0
        for j in range(500):
            name = r'Number/' + str(i) + '/' + str(j) + '.png'
            image = cv2.imread(name)
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            X = trans(image)
            X = X.to(device)
            with torch.no_grad():
                pred = model(X)
                # cv2.imshow("image", image)
                # cv2.waitKey(5)
                # print(format(pred[0].argmax(0).cpu())) # 预测的结果
                if int(format(pred[0].argmax(0).cpu())) == i:
                    flag += 1

        print("对{0}识别率为{1:.2%}".format(i, flag / 500))

# 对图片进行测试
def picture_test():
    frame = cv2.imread("number.png")
    rois = where_num(frame)
    # print(rois)
    if len(rois) > 0:
        for r in rois:
            x, y, w, h = r
            image = frame[y: y + h, x: x + w]
            image = resize_image(image)
            X = trans(image)
            X = X.to(device)
            with torch.no_grad():
                pred = model(X)
                result = format(pred[0].argmax(0).cpu())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(result), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.namedWindow("frame", 0)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)

# 部署
def arrange():
    cap = cv2.VideoCapture(0)
    while (1):
        ret, frame = cap.read()
        rois = where_num(frame)
        # print(rois)
        if len(rois) > 0:
            for r in rois:
                x, y, w, h = r
                image = frame[y: y + h, x: x + w]
                image = resize_image(image)
                X = trans(image)
                X = X.to(device)
                with torch.no_grad():
                    pred = model(X)
                    result = format(pred[0].argmax(0).cpu())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(result), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.namedWindow("frame", 0)
        cv2.imshow("frame", frame)
        if (cv2.waitKey(5) == 27):
            break

    cap.release()

if __name__ == "__main__":
    # opencv_test()
    # picture_test()
    arrange()
    cv2.destroyAllWindows()

