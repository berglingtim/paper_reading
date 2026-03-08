## 复现SPAI有感
https://github.com/mever-team/spai
这是源代码仓库
**训练标识**: `train_bs32_ep100` (batch size 32, 100 epochs)

### 训练配置
| 参数 | 值 |
|------|-----|
| Batch Size | 32 |
| Epochs | 100 |
| 基础学习率 | 0.0005 |
| 优化器 | AdamW (weight_decay=0.05) |
| 学习率调度 | Cosine + 5 epochs warmup |
| 损失函数 | BCE (Binary Cross Entropy) |
| 训练数据 | 7,040 张图像 |
| 验证数据 | 1,760 张图像 |
| 图像尺寸 | 224×224 |
| 模型 | ViT-Base (PatchSize=16) |
| 预训练权重 | mfm_pretrain_vit_base.pth |

### 最佳性能指标 (Epoch 89)

| 指标 | 最佳值 | 对应Epoch |
|------|--------|-----------|
| **AP** | **96.6%** | 89 |
| **AUC** | **96.3%** | 89 |
| **ACC** | **90.0%** | 87 |
| **Loss** | **0.6932** | 12 (最低) |

**Epoch 89 具体数据**:
- 验证 Loss: 0.7845
- 验证 ACC: 89.7%
- 验证 AP: **96.6%**
- 验证 AUC: **96.3%**
- 训练 Loss: 0.0624

### 训练过程关键节点
```
Epoch 0:  ACC=50.0%, AP=60.8%, AUC=63.5%  (初始化)
Epoch 12: Min Loss=0.6932  (loss最低点)
Epoch 26: AP开始稳定 >90%
Epoch 87: Max ACC=90.0%
Epoch 89: Max AP=96.6%, Max AUC=96.3%  (最佳综合性能)
Epoch 90-99: 性能稳定维持高位
```


### 2. 关键成功因素
- **更大的batch size (32 vs 8)**: 梯度估计更稳定，收敛效果更好
- **充足的训练轮数 (100 epochs)**: 模型有充分时间学习特征
- **合适的预训练权重**: 基于mfm_pretrain_vit_base.pth微调，利用了预训练知识
- **有效的数据增强**: 包含随机裁剪、翻转、旋转、高斯模糊、JPEG压缩等

### 3. 收敛分析
- **快速上升期 (Epoch 0-20)**: AP从60%快速提升到90%+
- **稳定优化期 (Epoch 20-89)**: 缓慢提升至最佳性能
- **平稳期 (Epoch 89-99)**: 性能基本稳定，略有波动

---

## 复现论文代码结构

### MFM预训练（通过自监督学习，让模型学习真实图像的频谱分布）

```
┌─────────────────────────────────────────────────────────────┐
│  输入图像 (PIL Image)                                        │
│     ↓                                                        │
│  数据增强：RandomResizedCrop + RandomHorizontalFlip          │
│     ↓                                                        │
│  频域掩码生成：FreqMaskGenerator                              │
│     - 随机选择高通或低通滤波器                                │
│     - mask_radius1=16（默认）                                │
│     - 采样比例sample_ratio=0.5（随机选择高低频）               │
│     ↓                                                        │
│  频域变换（2D FFT）：                                         │
│     - 将图像转换到频域                                        │
│     - 应用圆形掩码（中心低频/边缘高频）                        │
│     - 2D IFFT恢复时域图像（被损坏的版本）                      │
│     ↓                                                        │
│  编码器（ViT/Swin/ResNet）处理被损坏的图像 → 潜在表示 z       │
│     ↓                                                        │
│  解码器（PixelShuffle上采样）→ 重建图像                       │
│     ↓                                                        │
│  损失函数：Frequency Loss（计算重建图像与原始图像的差异）        │
└─────────────────────────────────────────────────────────────┘
```

### 监督微调（在AI生成图像检测任务上进行微调）

```
┌─────────────────────────────────────────────────────────────┐
│  输入图像（真实图像 + AI生成图像）                            │
│     ↓                                                        │
│  频谱预处理（MFViT.forward）：                                │
│     - 原始图像 → 归一化 → ViT特征提取                         │
│     - 低频分量 → 归一化 → ViT特征提取                         │
│     - 高频分量 → 归一化 → ViT特征提取                         │
│     ↓                                                        │
│  频谱恢复估计器（FrequencyRestorationEstimator）：            │
│     - 对每个patch特征进行投影                                 │
│     - 计算余弦相似度：                                        │
│       • sim(原始, 低频)                                       │
│       • sim(原始, 高频)                                       │
│       • sim(低频, 高频)                                       │
│     - 计算均值和标准差 → 6*N维特征向量（N=中间层数）           │
│     - （可选）原始图像特征分支 → 加权投影                       │
│     ↓                                                        │
│  分类头（ClassificationHead）：                               │
│     - MLP (输入→输入×mlp_ratio→输入×mlp_ratio→num_classes)   │
│     ↓                                                        │
│  损失计算：BCE Loss / SupCon Loss / Triplet Loss              │
└─────────────────────────────────────────────────────────────┘
```
### 具体的调用如下(为了防止代码不见了,我特意放在了隔壁文件夹,作为我的第一份复现的代码,各种备份是必要的,假如有陌生人闯入了我的github仓库,注意隔壁哪个叫做spai的文件夹不是我的,原网址为https://github.com/mever-team/spai)

```
MFM预训练:
main_mfm.py:main()
    └── models/mfm.py:build_mfm()
        ├── VisionTransformerForMFM (encoder)
        ├── VisionTransformerDecoderForMFM (decoder, optional)
        └── MFM (wrapper)
    └── data/data_mfm.py:build_loader_mfm()
        └── MFMTransform + FreqMaskGenerator

监督微调:
__main__.py:train()
    └── models/build.py:build_cls_model()
        └── models/sid.py:PatchBasedMFViT / MFViT
            ├── MFViT (频谱预处理)
            │   ├── filters.py:filter_image_frequencies()
            │   └── vision_transformer.py:VisionTransformer
            ├── FrequencyRestorationEstimator (频谱恢复估计)
            │   ├── Projector (patch投影)
            │   └── FeatureImportanceProjector (原始特征分支)
            └── ClassificationHead (分类头)
    └── data/data_finetune.py:build_loader_finetune()
        └── CSVDataset + build_transform()
```

## 因为第一次复现出了论文，本身有点激动，而且也有了些想法,查了些资料后,有了写信的收获，故列举出下个周以及之后做的小实验的方向
```
1. 颜色丰富度+皮肤平滑度+饱和度+噪声+面部对称性是否过于完美+面部光影
2. 将图分成RGB图，三个通道分别进行不同的处理，分别学习判断，采用3局两胜的方式进行判断
3. 观察RRDataset(真假各取一百张)的RGB, error level analysis(ELA),Noise Pattern, Edge Detection, Frequency Domain(FFT),local binary pattern, color histogram,(若有人脸,特别对人脸进行框取进行分析!!!), R-G distribution,R-B distribution, G-B distribution, luminance histogram,saturation histogram
```