import cv2
import numpy as np
import matplotlib.pyplot as plt

def single_scale_retinex(img, sigma):
    img = img.astype(np.float32) + 1.0
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex

def multi_scale_retinex(img, sigmas):
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    retinex /= len(sigmas)
    return retinex

def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    img_sum[img_sum == 0] = 1  # bo‘linishni oldini olish
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration

def msrcr(img, sigmas, G=5, b=25, alpha=125, beta=46, low_clip=0.01, high_clip=0.99):
    img = img.astype(np.float32) + 1.0
    retinex = multi_scale_retinex(img, sigmas)
    color_rest = color_restoration(img, alpha, beta)
    msrcr_result = G * (retinex * color_rest + b)

    # Ranglarni normallashtirish (0-255 oralig‘iga)
    for i in range(msrcr_result.shape[2]):
        channel = msrcr_result[:, :, i]
        low_val = np.percentile(channel, low_clip * 100)
        high_val = np.percentile(channel, high_clip * 100)
        channel = np.clip(channel, low_val, high_val)
        msrcr_result[:, :, i] = (channel - low_val) / (high_val - low_val) * 255

    return np.uint8(msrcr_result)

def msrcr_enhancement(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sigmas = [15, 80, 250]
    result = msrcr(img, sigmas)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Togri tasvir')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Algoritmm natijasi')
    plt.imshow(result)
    plt.axis('off')

    plt.show()

# Dastur ishga tushiriladi
image_path = 'img_1.png'  # O‘zingizdagi rasm nomini yozing
msrcr_enhancement(image_path)