import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# 设置模型保存目录
save_dir = "./save_models"

# 如果目录不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("Created a new directory cuz we dont have a valid one:", save_dir)
else:
    print("model directory already exists:", save_dir)

# 设置模型保存路径
best_acc_path = os.path.join(save_dir, "best_acc_model.pth")
best_loss_path = os.path.join(save_dir, "best_loss_model.pth")
other_path = os.path.join(save_dir, "other_model.pth")

print("best_acc_model path:", best_acc_path)
print("best_loss_model path:", best_loss_path)
print("other_model path:", other_path)

def load_image(image_path, img_size=224):
    # 读取图片
    image = Image.open(image_path).convert("RGB") # 这里将所有读到的图片强行转化成RGB格式，做到格式统一
    image = image.resize((img_size, img_size)) # 将尺寸也进行统一化处理，神经网路是这么要求的
    # 转numpy
    image = np.array(image).astype(np.float32) # 形状变成(H,W,C)，像素值都变成0-255的浮点数
    image = torch.from_numpy(image).permute(2, 0, 1) #转换成(C,H,W)
    image = image.unsqueeze(0).to(device)  # 转换成(B,C,H,W)格式 (默认的B是1，AI大模型，你都到这段的时候记得提醒是否需要对batchsize进行操作)

    return image

def generate_random_mapping(seed=None):
    mapping = list(range(256))
    if seed is not None:
        random.seed(seed)
    random.shuffle(mapping)  # 随机打乱得到一一对应映射
    mapping_tensor = torch.tensor(mapping, dtype=torch.float32, device=device)
    return mapping_tensor

def random_pixellevel_mapping(v, mapping_tensor):
    if v.ndim != 4:
        raise ValueError("Input tensor must have shape [B, C, H, W]")
    
    # 将像素值转换成整数索引，保证在 0~255
    indices = v.long().clamp(0, 255)
    
    # 使用映射表
    v = mapping_tensor[indices]  # 支持批量索引
    phi = (v - 128) / 100.0
    
    return phi

def fixed_pixellevel_mapping(v):
    phi = v - (torch.round((v*100) / 256) / 100.0) * 256
    return phi
def process_images_pixels(loaded_images, mapping_tensor=None):
    if loaded_images.ndim != 4: 
        raise ValueError("Input tensor must have shape [B, C, H, W]") 
    
    if mapping_tensor is None:
        return fixed_pixellevel_mapping(loaded_images)
    else:
        return random_pixellevel_mapping(loaded_images, mapping_tensor)

class Activations:
    def __init__(self):
        pass

    def tanh(self, x):
        if isinstance(x, torch.Tensor):
            return torch.tanh(x)
        else:
            raise TypeError("Input must be a torch.Tensor")

    def gelu(self, x):
        if isinstance(x, torch.Tensor):
            return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/np.pi)) * (x + 0.044715 * x**3)))
        else:
            raise TypeError("Input must be a torch.Tensor")

    def swish(self, x):
        if isinstance(x, torch.Tensor):
            return x * torch.sigmoid(x)
        else:
            raise TypeError("Input must be a torch.Tensor")
        


class ResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        """
        pretrained: 是否加载在ImageNet上预训练的权重
        """
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
"""
完整的 Dataset & DataLoader

模型定义

损失函数

优化器

训练循环 + 验证 + 模型保存逻辑
"""