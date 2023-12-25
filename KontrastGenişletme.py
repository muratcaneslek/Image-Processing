import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image):
    # Görüntüdeki en karanlık ve en açık piksel değerlerini bulma
    min_val = np.min(image)
    max_val = np.max(image)

    # Kontrast genişletme işlemi
    stretched_image = (image - min_val) * (255.0 / (max_val - min_val))

    # Veri tipini uint8'ye dönüştürme
    stretched_image = stretched_image.astype(np.uint8)

    return stretched_image

img = cv2.imread('./Pictures/picture1.jpg', 0)

# Kontrast genişletme işlemi
stretched_img = contrast_stretching(img)

# Giriş ve çıkış görüntülerini gösterme
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Orijinal Görüntü')

plt.subplot(1, 2, 2)
plt.imshow(stretched_img, cmap='gray')
plt.title('Kontrast Genişletme Sonucu')

plt.show()
