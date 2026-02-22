import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import ImageOps

# =========================
# 基本配置
# =========================
input_dir = "../../temp_dir/RRDataset_original_train_val/train/ai"
output_dir = "baseline_imgs/train/ai"
os.makedirs(output_dir, exist_ok=True)

model_id = "SG161222/Realistic_Vision_V1.4"
strength = 0.8
max_steps = 100  # 每张图片最大生成步数
min_epochs_before_detect = 50  # 至少训练 50 个 epoch 后才开始检测
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 加载 Stable Diffusion 模型
# =========================
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)
pipe.safety_checker = None

# =========================
# 加载检测模型 (ResNet50)
# =========================
det_model_path = "./resnet50_model/best_acc_model.pth"
det_model = models.resnet50(weights=None)
det_model.fc = nn.Linear(det_model.fc.in_features, 2)  # 2类: ai=0, real=1
det_model.load_state_dict(torch.load(det_model_path, map_location=device))
det_model = det_model.to(device)
det_model.eval()

# =========================
# 图像预处理
# =========================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# 5️⃣ 获取 JPG 图片
# =========================
image_files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith((".jpg", ".jpeg"))
])
print(f"Found {len(image_files)} jpg images.")

# =========================
# 循环生成 + 检测
# =========================


check_every_n_steps = 1  # 开始检测后，每隔多少 epoch 检测一次
threshold_prob_real = 0.8  # 概率阈值

counter = 0  

for img_name in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(input_dir, img_name)
    init_image = Image.open(img_path).convert("RGB")
    # 保持比例 resize 到 SD 输入尺寸
    init_image = ImageOps.fit(init_image, (512, 512), Image.LANCZOS)

    current_image = init_image
    step_done = 0
    saved = False

    while step_done < max_steps:
        steps_to_run = min(check_every_n_steps, max_steps - step_done)
        step_done += steps_to_run

        # 逐步生成
        result = pipe(
            prompt="",
            image=current_image,
            strength=strength,
            num_inference_steps=steps_to_run
        )
        current_image = result.images[0]

        # 至少训练 min_epochs_before_detect 个 epoch 后才开始检测，之后每隔 check_every_n_steps 个 epoch 检测一次
        should_detect = (
            step_done >= min_epochs_before_detect and
            (step_done - min_epochs_before_detect) % check_every_n_steps == 0
        )
        if not should_detect:
            continue  # 未到检测时机，继续下一个 epoch

        # ResNet50 检测
        img_tensor = preprocess(current_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = det_model(img_tensor)
            probs = F.softmax(output, dim=1)
            prob_real = probs[0, 1].item()

        if prob_real > threshold_prob_real:
            save_name = f"{counter:06d}.jpg"
            save_path = os.path.join(output_dir, save_name)
            current_image.save(save_path, format="JPEG", quality=95)
            counter += 1
            saved = True
            print(f"{img_name} saved at step {step_done}, P(real)={prob_real:.4f}")
            break  # 提前停止生成，处理下一张图

    if not saved:
        print(f"{img_name} 未检测为真实，跳过保存。")

print("Generation and detection finished.")