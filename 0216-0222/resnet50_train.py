import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ======================
# 1. Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
save_dir = "./resnet50_model"

# ======================
# 2. Model
# ======================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ======================
# 3. Transform
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# 4. Dataset
# ======================
train_dir = "../../temp_dir/RRDataset_original_train_val/train"
val_dir = "../../temp_dir/RRDataset_original_train_val/val"

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Class mapping:", train_dataset.class_to_idx)

# ======================
# 5. Loss & Optimizer
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,        # 学习率变为原来的一半
    patience=5,        # 5个epoch不变好就降
)

# ======================
# Resume Training from Best Model
# ======================
save_dir = "./resnet50_model"
os.makedirs(save_dir, exist_ok=True)

# 定义模型和检查点路径
best_acc_path = os.path.join(save_dir, "best_acc_model.pth")
best_loss_path = os.path.join(save_dir, "best_loss_model.pth")
checkpoint_path = os.path.join(save_dir, "last_checkpoint.pth")

start_epoch = 0
best_acc = 0.0
best_loss = float('inf')

# 从已保存的最佳模型文件中读取最佳值
if os.path.exists(best_acc_path):
    print("📊 Loading best accuracy from saved model...")
    # 先加载模型来获取信息，但这不是必须的
    # 我们可以通过文件名和文件修改时间来判断，但更准确的是从checkpoint读取
    # 实际上，最佳值应该从checkpoint中读取，而不是从模型文件
    
if os.path.exists(checkpoint_path):
    print("🔄 Loading training state from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载训练状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    best_loss = checkpoint['best_loss']
    
    print(f"Resumed from epoch {start_epoch}")
    print(f"Best metrics from checkpoint - Acc: {best_acc:.2f}%, Loss: {best_loss:.4f}")
    
    #  验证保存的最佳模型文件是否真的存在
    if os.path.exists(best_acc_path):
        print("✅ Best accuracy model file exists")
    else:
        print("⚠️ Best accuracy model file missing, will create at next save")
        
    if os.path.exists(best_loss_path):
        print("✅ Best loss model file exists")
    else:
        print("⚠️ Best loss model file missing, will create at next save")
        
else:
    print("🔄 No checkpoint found, starting from pretrained weights")
    # 如果没有checkpoint但存在最佳模型文件，尝试读取它们来设置初始最佳值
    try:
        # 这里我们无法直接从模型文件读取准确率和损失值
        # 所以保持默认值0和inf
        print("No checkpoint found, starting fresh training")
    except Exception as e:
        print(f"Error loading best model info: {e}")

# ======================
# 6. Training Loop
# ======================
num_epochs = 120

for epoch in range(start_epoch, num_epochs):

    # ========= Train =========
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for images, labels in train_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": optimizer.param_groups[0]["lr"]
        })

    train_loss = running_loss / len(train_loader)

    # ========= Validation =========
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    val_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    print(f"\nEpoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val Loss: {val_loss:.4f} "
          f"Val Acc: {val_acc:.2f}% "
          f"LR: {optimizer.param_groups[0]['lr']}")

    # 调用 scheduler
    scheduler.step(val_loss)

    # ========= Save Checkpoint =========
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,  # 保存当前记录的最佳值，而不是更新后的
        'best_loss': best_loss,
    }
    torch.save(checkpoint, checkpoint_path)

    # ========= Save Best Loss =========
    if val_loss < best_loss:
        best_loss = val_loss  # 修改4: 更新最佳损失记录
        torch.save(model.state_dict(), best_loss_path)
        print(f"✅ Saved new best loss model! (Loss: {best_loss:.4f})")
        
        # 更新checkpoint中的最佳损失
        checkpoint['best_loss'] = best_loss
        torch.save(checkpoint, checkpoint_path)

    # ========= Save Best Accuracy =========
    if val_acc > best_acc:
        best_acc = val_acc  # 修改5: 更新最佳准确率记录
        torch.save(model.state_dict(), best_acc_path)
        print(f"✅ Saved new best accuracy model! (Acc: {best_acc:.2f}%)")
        
        # 更新checkpoint中的最佳准确率
        checkpoint['best_acc'] = best_acc
        torch.save(checkpoint, checkpoint_path)
        
        # 保存一个专门的best_acc_checkpoint（可选）
        best_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'best_loss': best_loss,
        }
        torch.save(best_checkpoint, os.path.join(save_dir, "best_acc_checkpoint.pth"))

print("Training Finished.")
print(f"Best Accuracy: {best_acc:.2f}%")
print(f"Best Loss: {best_loss:.4f}")