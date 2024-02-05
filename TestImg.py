import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torch
from PIL import Image
from torchvision import transforms, models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 创建主窗口
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 打开文件对话框
file_path = filedialog.askopenfilename()
if file_path:
    print("已选择文件:", file_path)
else:
    print("没有选择文件")

# 定义测试集转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 读取图像,转换为tensor
img_path = file_path
img = Image.open(img_path).convert('RGB')
img = transform(img)
img = img.unsqueeze(0)  # 增加batch维度
img = img.to(device)

# 导入模型
model = torch.load('model.pth')
model = model.to(device)

# 预测
model.eval()
with torch.no_grad():
    output = model(img)
    pred = torch.max(output, 1)[1].item()

# 载入类别与索引映射
idx_to_label = np.load('idx_to_labels.npy', allow_pickle=True).item()

# 打印预测类别
print('预测结果:', idx_to_label[pred])