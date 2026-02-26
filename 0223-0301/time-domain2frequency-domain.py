import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# 1. 读取图片（灰度图） | Load image in grayscale
# =========================
img = cv2.imread('张林方正.png', cv2.IMREAD_GRAYSCALE)

# =========================
# 2. 计算傅里叶变换 | Compute 2D Fourier Transform
# =========================
f = np.fft.fft2(img)           # 傅里叶变换
fshift = np.fft.fftshift(f)    # 将零频率移到中心
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # 幅度谱

# =========================
# 3. 显示时域和频域图像 | Display image and its frequency spectrum
# =========================
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image (Time Domain)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (Frequency Domain)')
plt.axis('off')

time.sleep(3)
plt.show()