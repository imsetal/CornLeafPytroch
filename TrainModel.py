import matplotlib.pyplot as plt
import time
import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim

if __name__ == '__main__':

    # windows操作系统
    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

    # 忽略烦人的红色提示
    warnings.filterwarnings("ignore")

    # ## 获取计算硬件
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # ## 图像预处理
    from torchvision import transforms

    # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                        ])

    # ## 载入图像分类数据集
    # 数据集文件夹路径
    dataset_dir = 'data'
    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'val')
    print('训练集路径', train_path)
    print('测试集路径', test_path)

    # 载入训练集
    train_dataset = datasets.ImageFolder(train_path, train_transform)

    # 载入测试集
    test_dataset = datasets.ImageFolder(test_path, test_transform)

    print('训练集图像数量', len(train_dataset))
    print('类别个数', len(train_dataset.classes))
    print('各类别名称', train_dataset.classes)

    print('测试集图像数量', len(test_dataset))
    print('类别个数', len(test_dataset.classes))
    print('各类别名称', test_dataset.classes)

    # ## 类别和索引号 一一对应
    # 各类别名称
    class_names = train_dataset.classes
    n_class = len(class_names)
    class_names

    # 映射关系：类别 到 索引号
    train_dataset.class_to_idx

    # 映射关系：索引号 到 类别
    idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
    idx_to_labels

    # 保存为本地的 npy 文件
    np.save('idx_to_labels.npy', idx_to_labels)
    np.save('labels_to_idx.npy', train_dataset.class_to_idx)

    # ## 定义数据加载器DataLoader
    BATCH_SIZE = 32

    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4
                             )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4
                            )

    # ## 查看一个batch的图像和标注
    # DataLoader 是 python生成器，每次调用返回一个 batch 的数据
    images, labels = next(iter(train_loader))
    images.shape
    labels

    # ## 可视化一个batch的图像和标注
    # 将数据集中的Tensor张量转为numpy的array数据类型
    images = images.numpy()
    images[5].shape
    plt.hist(images[5].flatten(), bins=50)
    plt.show()

    # batch 中经过预处理的图像
    idx = 2
    plt.imshow(images[idx].transpose((1,2,0))) # 转为(224, 224, 3)
    plt.title('label:'+str(labels[idx].item()))
    label = labels[idx].item()
    label
    pred_classname = idx_to_labels[label]
    pred_classname

    # 原始图像
    idx = 2
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    plt.imshow(np.clip(images[idx].transpose((1,2,0)) * std + mean, 0, 1))
    plt.title('label:'+ pred_classname)
    plt.show()

    model = models.resnet18(pretrained=False) # 只载入模型结构，不载入预训练权重参数
    model.fc = nn.Linear(model.fc.in_features, n_class)
    optimizer = optim.Adam(model.parameters())

    # ## 训练配置
    model = model.to(device)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练轮次 Epoch
    EPOCHS = 3

    # ## 模拟一个batch的训练
    # 获得一个 batch 的数据和标注
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    # 输入模型，执行前向预测
    outputs = model(images)

    # 获得当前 batch 所有图像的预测类别 logit 分数
    outputs.shape

    # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
    loss = criterion(outputs, labels)

    # 反向传播“三部曲”
    optimizer.zero_grad() # 清除梯度
    loss.backward() # 反向传播
    optimizer.step() # 优化更新

    # 获得当前 batch 所有图像的预测类别
    _, preds = torch.max(outputs, 1)
    preds
    labels

    # ## 运行完整训练
    # 遍历每个 EPOCH
    for epoch in tqdm(range(EPOCHS)):

        model.train()

        for images, labels in train_loader:  # 获取训练集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)           # 前向预测，获得当前 batch 的预测结果
            loss = criterion(outputs, labels) # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数

            optimizer.zero_grad()
            loss.backward()                   # 损失函数对神经网络权重反向传播求梯度
            optimizer.step()                  # 优化更新神经网络权重

    # ## 在测试集上初步测试
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader): # 获取测试集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)              # 前向预测，获得当前 batch 的预测置信度
            _, preds = torch.max(outputs, 1)     # 获得最大置信度对应的类别，作为预测结果
            total += labels.size(0)
            correct += (preds == labels).sum()   # 预测正确样本个数

        print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))

    # ## 保存模型
    torch.save(model, 'model.pth')