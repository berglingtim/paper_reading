import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Tuple


class FixedPixelMapping(nn.Module):
    """
    固定像素级映射模块
    
    对应论文公式(4):
    φ_f(ν) = ν - round(ν/256, 2) × 256
    
    功能: 将原始像素值(0-255)映射到新值，破坏像素值的单调排列，
          将低频信息转化为高频成分，同时保持计算高效。
    """
    
    def __init__(self, decimals: int = 2):
        """
        参数:
            decimals: round运算的小数位数，论文建议值为2
        """
        super().__init__()
        self.decimals = decimals
        
        # 预计算所有0-255像素值的映射表
        self._precompute_mapping_table()
        
    def _precompute_mapping_table(self):
        """预计算0-255所有像素值的映射结果"""
        v = torch.arange(0, 256, dtype=torch.float32)
        # φ_f(ν) = ν - round(ν/256, 2) × 256
        mapped = v - torch.round(v / 256.0, decimals=self.decimals) * 256.0
        self.register_buffer('mapping_table', mapped)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入图像张量，范围[0, 255]，形状 [B, C, H, W] 或 [B, H, W, C]
               数据类型应为整数或浮点数
        
        返回:
            映射后的图像张量，形状与输入相同，值域约[-1.28, 1.28]
        """
        # 确保输入值在合理范围
        # 使用clone()可以防止改变原值
        x_orig = x.clone()
        
        # 如果是浮点数，可能已经是[0,1]范围，需要转换回[0,255]
        if x_orig.is_floating_point():
            if x_orig.max() <= 1.0 and x_orig.min() >= 0.0:
                x_orig = x_orig * 255.0
        
        # 将像素值四舍五入到整数作为索引
        indices = torch.round(x_orig).clamp(0, 255).long()
        
        # 应用预计算的映射表
        if indices.shape[-1] == 3:  # [B, H, W, C] 格式
            mapped = torch.zeros_like(x_orig, dtype=torch.float32)
            for c in range(3):
                mapped[..., c] = self.mapping_table[indices[..., c]]
        else:  # [B, C, H, W] 格式
            mapped = torch.zeros_like(x_orig, dtype=torch.float32)
            for c in range(x_orig.shape[1]):
                mapped[:, c] = self.mapping_table[indices[:, c]]
                
        return mapped


class RandomPixelMapping(nn.Module):
    """
    随机像素级映射模块
    
    对应论文公式(5)(6):
    T_c ~ U(-1,1)^{256}, c ∈ {0,1,2}
    I'[x,y] = T_c[I_c[x,y]]
    
    功能: 为每个样本的每个通道生成独立的随机映射表，
          同样能有效增强相邻像素差异性。
    """
    
    def __init__(self, 
                 per_sample: bool = True,
                 per_channel: bool = True,
                 low: float = -1.0, 
                 high: float = 1.0):
        """
        参数:
            per_sample: 是否为每个样本生成不同的映射表
            per_channel: 是否为每个通道生成不同的映射表
            low: 均匀分布下限
            high: 均匀分布上限
        """
        super().__init__()
        self.per_sample = per_sample
        self.per_channel = per_channel
        self.low = low
        self.high = high
        
    def _generate_mapping_table(self, 
                               batch_size: int, 
                               num_channels: int,
                               device: torch.device) -> torch.Tensor:
        """
        生成随机映射表
        
        返回:
            形状: [B, C, 256] 或 [C, 256] 或 [256]
        """
        if self.per_sample and self.per_channel:
            # [B, C, 256]
            return torch.empty(batch_size, num_channels, 256, device=device)\
                     .uniform_(self.low, self.high)
        elif self.per_channel:
            # [C, 256]
            return torch.empty(num_channels, 256, device=device)\
                     .uniform_(self.low, self.high)
        else:
            # [256]
            return torch.empty(256, device=device)\
                     .uniform_(self.low, self.high)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入图像张量，范围[0, 255]，形状 [B, C, H, W] 或 [B, H, W, C]
        
        返回:
            随机映射后的图像张量
        """
        x_orig = x.clone()
        
        # 处理浮点输入
        if x_orig.is_floating_point():
            if x_orig.max() <= 1.0 and x_orig.min() >= 0.0:
                x_orig = x_orig * 255.0
        
        # 确定输入格式
        if x_orig.shape[-1] == 3:  # [B, H, W, C]
            x_orig = x_orig.permute(0, 3, 1, 2)  # [B, C, H, W]
            channel_dim = 1
            h_dim, w_dim = 2, 3
            need_permute_back = True
        else:  # [B, C, H, W]
            channel_dim = 1
            h_dim, w_dim = 2, 3
            need_permute_back = False
        
        batch_size, num_channels, h, w = x_orig.shape
        
        # 四舍五入到整数索引
        indices = torch.round(x_orig).clamp(0, 255).long()
        
        # 生成映射表
        mapping_table = self._generate_mapping_table(batch_size, num_channels, x_orig.device)
        
        # 应用映射
        mapped = torch.zeros_like(x_orig, dtype=torch.float32)
        
        if self.per_sample and self.per_channel:
            # [B, C, H, W] 每个样本每个通道独立映射
            for b in range(batch_size):
                for c in range(num_channels):
                    mapped[b, c] = mapping_table[b, c, indices[b, c]]
        elif self.per_channel:
            # 所有样本共享通道映射表
            for c in range(num_channels):
                mapped[:, c] = mapping_table[c, indices[:, c]]
        else:
            # 全局共享一个映射表
            mapped = mapping_table[indices]
        
        if need_permute_back:
            mapped = mapped.permute(0, 2, 3, 1)  # [B, H, W, C]
            
        return mapped


class PixelMappingModule(nn.Module):
    """
    像素级映射模块 - 封装固定和随机两种映射方式
    
    可作为独立的预处理模块，在分类前对图像进行变换。
    """
    
    def __init__(self, 
                 mode: str = 'fixed',
                 decimals: int = 2,
                 per_sample: bool = True,
                 per_channel: bool = True,
                 low: float = -1.0,
                 high: float = 1.0):
        """
        参数:
            mode: 'fixed' 或 'random'
            decimals: fixed模式下的round小数位数
            per_sample: random模式下是否为每个样本生成不同映射
            per_channel: random模式下是否为每个通道生成不同映射
            low, high: random模式下均匀分布的范围
        """
        super().__init__()
        self.mode = mode
        
        if mode == 'fixed':
            self.mapping = FixedPixelMapping(decimals=decimals)
        elif mode == 'random':
            self.mapping = RandomPixelMapping(
                per_sample=per_sample,
                per_channel=per_channel,
                low=low,
                high=high
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'fixed' or 'random'.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapping(x)


# ==================== 使用示例 ====================

def example_usage():
    """展示如何使用像素级映射模块"""
    
    # 创建示例图像: [B, C, H, W], 值域[0,255]
    dummy_image = torch.randint(0, 256, (4, 3, 128, 128), dtype=torch.float32)
    
    # 1. 固定映射
    fixed_mapper = PixelMappingModule(mode='fixed', decimals=2)
    mapped_fixed = fixed_mapper(dummy_image)
    print(f"固定映射 - 输入范围: [{dummy_image.min():.1f}, {dummy_image.max():.1f}]")
    print(f"固定映射 - 输出范围: [{mapped_fixed.min():.2f}, {mapped_fixed.max():.2f}]")
    print(f"固定映射 - 输出形状: {mapped_fixed.shape}")
    print()
    
    # 2. 随机映射 (每个样本、每个通道独立)
    random_mapper = PixelMappingModule(
        mode='random',
        per_sample=True,
        per_channel=True,
        low=-1.0,
        high=1.0
    )
    mapped_random = random_mapper(dummy_image)
    print(f"随机映射 - 输入范围: [{dummy_image.min():.1f}, {dummy_image.max():.1f}]")
    print(f"随机映射 - 输出范围: [{mapped_random.min():.2f}, {mapped_random.max():.2f}]")
    print(f"随机映射 - 输出形状: {mapped_random.shape}")
    print()
    
    # 3. 固定映射 + 分类器示例
    class PixelMappingClassifier(nn.Module):
        """集成像素级映射的检测器"""
        def __init__(self, backbone, mode='fixed'):
            super().__init__()
            self.pixel_mapping = PixelMappingModule(mode=mode)
            self.classifier = backbone  # 假设是预训练分类器
            
        def forward(self, x):
            # 确保输入是[0,255]范围的整数
            if x.is_floating_point() and x.max() <= 1.0:
                x = x * 255.0
            # 应用像素映射
            x = self.pixel_mapping(x)
            # 分类
            return self.classifier(x)
    
    # 使用ResNet-50作为backbone
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=False)
    # 修改最后一层为二分类
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)
    
    model = PixelMappingClassifier(backbone, mode='fixed')
    print(f"检测器模型构建完成")
    
    # 前向传播
    dummy_input = torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.float32)
    output = model(dummy_input)
    print(f"检测器输出形状: {output.shape}")


if __name__ == "__main__":
    example_usage()