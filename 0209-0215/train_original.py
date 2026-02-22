import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from tqdm import tqdm  # 进度条显示
import numpy as np
import os
import time

# =====================
# 1. 设备
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 创建保存模型的目录
os.makedirs("saved_models", exist_ok=True)

# =====================
# 2. 配置参数
# =====================
class Config:
    # 训练控制
    RESUME_TRAINING = True  # 是否从上次训练继续
    RESUME_MODEL_PATH = "saved_models/best_acc_model.pth"  # 要加载的模型路径
    START_EPOCH = 0  # 起始epoch（自动更新）
    
    # 训练超参数
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    
    # 数据增强
    USE_DATA_AUGMENTATION = True
    
    # 早停
    PATIENCE = 6
    MIN_DELTA = 0.001

config = Config()

# =====================
# 3. 加载 Hugging Face 数据集
# =====================
dataset = load_dataset("RohanRamesh/ff-images-dataset")
train_hf = dataset["train"]
val_hf = dataset["validation"]

# =====================
# 4. 定义 Dataset 封装
# =====================
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]      # PIL.Image
        label = item["label"]    # 0 = real, 1 = fake

        if self.transform:
            img = self.transform(img)

        return img, label

# =====================
# 5. 图像预处理和增强
# =====================
if config.USE_DATA_AUGMENTATION:
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), value='random')
    ])
else:
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# 验证集：基本预处理
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = HFDataset(train_hf, train_transform)
val_dataset = HFDataset(val_hf, val_transform)

# =====================
# 6. 数据集分析和处理
# =====================
train_labels = [item["label"] for item in train_hf]
val_labels = [item["label"] for item in val_hf]
print(f"训练集: Real={train_labels.count(0)}, Fake={train_labels.count(1)}")
print(f"验证集: Real={val_labels.count(0)}, Fake={val_labels.count(1)}")

# 计算类别权重
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"类别权重: {class_weights}")

from torch.utils.data import WeightedRandomSampler
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_dataset, 
    batch_size=config.BATCH_SIZE,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config.BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

# =====================
# 7. 构建模型
# =====================
def build_model():
    """构建或加载模型"""
    model = models.resnet18(pretrained=True)
    
    # 冻结部分层
    for name, param in model.named_parameters():
        if 'layer1' in name or 'layer2' in name:
            param.requires_grad = False

    # 替换全连接层
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    
    return model

# =====================
# 8. 加载预训练模型（如果存在）
# =====================
def load_pretrained_model(model, model_path, optimizer=None, scheduler=None):
    """加载之前训练的模型"""
    if os.path.exists(model_path):
        print(f"加载预训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果需要继续训练，加载优化器和学习率调度器
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("优化器状态已加载")
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("学习率调度器状态已加载")
        
        # 获取之前的epoch和最佳指标
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"从epoch {start_epoch}开始继续训练")
        print(f"之前的最佳验证准确率: {best_val_acc:.4f}")
        print(f"之前的最佳验证损失: {best_val_loss:.4f}")
        
        return model, optimizer, scheduler, start_epoch, best_val_acc, best_val_loss
    
    else:
        print(f"未找到预训练模型: {model_path}，从头开始训练")
        return model, optimizer, scheduler, 0, 0.0, float('inf')

# =====================
# 9. 初始化模型、损失函数和优化器
# =====================
model = build_model().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(
    model.parameters(), 
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=config.NUM_EPOCHS,
    eta_min=1e-6
)

# 加载预训练模型（如果配置了继续训练）
if config.RESUME_TRAINING:
    model, optimizer, scheduler, config.START_EPOCH, best_val_acc, best_val_loss = load_pretrained_model(
        model, config.RESUME_MODEL_PATH, optimizer, scheduler
    )
else:
    best_val_acc = 0.0
    best_val_loss = float('inf')

# 打印可训练参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数数量: {trainable_params:,}")

# =====================
# 10. 早停机制
# =====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

early_stopping = EarlyStopping(
    patience=config.PATIENCE, 
    min_delta=config.MIN_DELTA
)

# =====================
# 11. 训练循环
# =====================
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

# 如果是从上次继续训练，可能需要加载历史记录
if config.RESUME_TRAINING and config.START_EPOCH > 0:
    # 尝试加载之前的历史记录
    history_path = "saved_models/training_history.pth"
    if os.path.exists(history_path):
        history_checkpoint = torch.load(history_path, map_location=device)
        history = history_checkpoint['history']
        print(f"加载了之前 {len(history['train_loss'])} 个epoch的训练历史")

print(f"\n开始训练，总epoch数: {config.NUM_EPOCHS}，起始epoch: {config.START_EPOCH}")
print("=" * 60)

for epoch in range(config.START_EPOCH, config.NUM_EPOCHS):
    start_time = time.time()
    
    # ---- train ----
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=False)
    for batch_idx, (images, labels) in enumerate(train_bar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        train_bar.set_postfix({
            "loss": f"{train_loss/total:.4f}",
            "acc": f"{correct/total:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    train_loss /= total
    train_acc = correct / total
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    # ---- validation ----
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]", leave=False)
    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
            val_bar.set_postfix({
                "val_loss": f"{val_loss/val_total:.4f}",
                "val_acc": f"{val_correct/val_total:.4f}"
            })

    val_loss /= val_total
    val_acc = val_correct / val_total
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 更新学习率
    scheduler.step()
    
    epoch_time = time.time() - start_time
    
    # ============= 保存模型部分 =============
    
    # 1. 保存最佳验证集准确率模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, "saved_models/best_acc_model.pth")
        print(f"✓ 新最佳准确率模型保存! Val Acc: {val_acc:.4f}")
    
    # 2. 保存最佳验证集损失模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, "saved_models/best_loss_model.pth")
        print(f"✓ 新最佳损失模型保存! Val Loss: {val_loss:.4f}")
    
    # 3. 定期保存检查点（包含历史记录）
    if (epoch + 1) % 3 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': history,  # 保存完整历史
        }, f"saved_models/checkpoint_epoch_{epoch+1}.pth")
        
        # 单独保存历史记录
        torch.save({
            'history': history,
            'epoch': epoch
        }, "saved_models/training_history.pth")
    
    # 打印epoch结果
    print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  最佳Val Acc: {best_val_acc:.4f} | 最佳Val Loss: {best_val_loss:.4f}")
    print("-" * 60)
    
    # 早停检查
    if early_stopping(val_loss):
        print(f"\n⚠️ 早停触发! 在epoch {epoch+1}停止训练")
        break

# =====================
# 12. 训练结束后
# =====================
# 保存最终模型
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_acc': val_acc,
    'history': history,
}, "saved_models/final_model.pth")

# 保存历史记录
torch.save({
    'history': history,
    'epoch': epoch,
    'best_val_acc': best_val_acc,
    'best_val_loss': best_val_loss
}, "saved_models/training_history.pth")

# =====================
# 13. 可视化训练历史
# =====================
import matplotlib.pyplot as plt

def plot_training_history(history):
    """绘制训练历史图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 训练/验证损失
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 训练/验证准确率
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s', markersize=3)
    axes[0, 1].axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.7, label=f'Best Val Acc: {best_val_acc:.4f}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 学习率变化
    axes[1, 0].plot(history['lr'], marker='o', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    
    # 准确率差值
    diff = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(diff, marker='o', markersize=3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train-Val Acc Diff')
    axes[1, 1].set_title('Overfitting Indicator')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("saved_models/training_history.png", dpi=150, bbox_inches='tight')
    plt.show()

plot_training_history(history)

# =====================
# 14. 模型评估
# =====================
def evaluate_model(model_path, model_name="模型"):
    """评估模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建新模型实例并加载权重
    eval_model = build_model().to(device)
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    eval_model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = eval_model(images)
            
            # 获取预测概率
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算评估指标
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
    import seaborn as sns
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*60)
    print(f"评估{model_name}: {model_path}")
    print(f"准确率: {accuracy:.4f}")
    print(f"加权精确率: {precision:.4f}")
    print(f"加权召回率: {recall:.4f}")
    print(f"加权F1分数: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"saved_models/confusion_matrix_{os.path.basename(model_path).split('.')[0]}.png", dpi=150)
    plt.show()
    
    return accuracy, precision, recall, f1

# 评估最佳模型
print("\n评估最佳模型...")
evaluate_model("saved_models/best_acc_model.pth", "最佳准确率模型")

# =====================
# 15. 打印总结
# =====================
print("\n" + "="*60)
print("训练完成!")
print(f"总训练轮数: {epoch+1}/{config.NUM_EPOCHS}")
print(f"起始epoch: {config.START_EPOCH}")
print(f"最佳验证集准确率: {best_val_acc:.4f}")
print(f"最佳验证集损失: {best_val_loss:.4f}")
print(f"最终验证集准确率: {history['val_acc'][-1]:.4f}")
print("\n保存的模型文件:")
print(f"  - 最佳准确率模型: saved_models/best_acc_model.pth")
print(f"  - 最佳损失模型: saved_models/best_loss_model.pth")
print(f"  - 最终模型: saved_models/final_model.pth")
print(f"  - 检查点文件: saved_models/checkpoint_epoch_*.pth")
print(f"  - 训练历史: saved_models/training_history.pth")
print(f"  - 训练历史图: saved_models/training_history.png")
print("\n继续训练说明:")
print(f"  要从中断处继续训练，请设置: RESUME_TRAINING = True")
print(f"  要加载的模型: {config.RESUME_MODEL_PATH}")
print("="*60)

# =====================
# 16. 继续训练示例函数
# =====================
def continue_training_example():
    """如何继续训练的示例"""
    print("\n" + "="*60)
    print("如何继续训练:")
    print("="*60)
    print("""
方法1: 修改配置文件
在代码开头修改Config类:
    class Config:
        RESUME_TRAINING = True  # 设为True
        RESUME_MODEL_PATH = "saved_models/best_acc_model.pth"  # 要加载的模型
        START_EPOCH = 0  # 会自动更新
        NUM_EPOCHS = 50  # 可以增加总epoch数

方法2: 使用命令行参数（更推荐）
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_path', type=str, default='saved_models/best_acc_model.pth')
    args = parser.parse_args()
    
    config.RESUME_TRAINING = args.resume
    config.RESUME_MODEL_PATH = args.model_path

方法3: 单独的训练脚本
创建 train_continue.py:
    import torch
    from main import build_model, load_pretrained_model
    
    # 加载模型
    model = build_model()
    model, optimizer, scheduler, start_epoch, best_acc, best_loss = \\
        load_pretrained_model(model, 'saved_models/best_acc_model.pth')
    
    # 继续训练逻辑...
    """)

continue_training_example()