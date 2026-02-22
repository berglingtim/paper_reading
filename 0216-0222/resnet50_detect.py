import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ======================
# 1. Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


"""
========== original Result ==========
Total samples: 17000
Overall Accuracy: 75.32%
Class 'ai' Accuracy: 77.00%
Class 'real' Accuracy: 73.64%

========== redigital Result ==========
Total samples: 16999
Overall Accuracy: 71.20%
Class 'ai' Accuracy: 77.22%
Class 'real' Accuracy: 65.17%

========== transfer Result ==========
Total samples: 17000
Overall Accuracy: 76.06%
Class 'ai' Accuracy: 77.56%
Class 'real' Accuracy: 74.56%
"""
# ======================
# 2. Path
# ======================
test_dir = "../../temp_dir/RRDataset_final/transfer"
model_path = "./resnet50_model/best_acc_model.pth"
# 如果你想加载完整checkpoint，可以改成：
# model_path = "./resnet50_model/best_acc_checkpoint.pth"

# ======================
# 3. Transform (预处理)
# ======================
# 标准 ImageNet 测试预处理
transform = transforms.Compose([
    transforms.Resize(256),             # 先统一最短边到256
    transforms.CenterCrop(224),         # 中心裁剪224x224
    transforms.ToTensor(),              # 转tensor [0,1]
    transforms.Normalize(               # ImageNet归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# 4. Dataset & DataLoader
# ======================
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Class mapping:", test_dataset.class_to_idx)

# ======================
# 5. Model
# ======================
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

checkpoint = torch.load(model_path, map_location=device)

# ===== 兼容两种保存方式 =====
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    print("Detected full checkpoint file.")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded from epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Best Acc recorded: {checkpoint.get('best_acc', 'Unknown')}")
else:
    print("Detected pure state_dict file.")
    model.load_state_dict(checkpoint)

model.eval()
from tqdm import tqdm 

# ======================
# 6. Evaluation with Progress Bar
# ======================
correct = 0
total = 0

class_correct = [0] * len(test_dataset.classes)
class_total = [0] * len(test_dataset.classes)

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

# ======================
# 7. Results
# ======================
accuracy = 100 * correct / total
print("\n========== Test Result ==========")
print(f"Total samples: {total}")
print(f"Overall Accuracy: {accuracy:.2f}%")

classes = test_dataset.classes
for i in range(len(classes)):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"Class '{classes[i]}' Accuracy: {acc:.2f}%")

print("\nEvaluation Finished.")