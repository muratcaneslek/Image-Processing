import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_laplacian_filter(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Laplace filtresi uygulama
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    laplacian = np.abs(laplacian)

    return laplacian

img = cv2.imread('./Pictures/picture1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Laplace filtresi uygulama
laplacian_img = apply_laplacian_filter(img)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Orijinal Görüntü')

plt.subplot(1, 2, 2)
plt.imshow(laplacian_img, cmap='gray')
plt.title('Laplace Filtre Uygulaması')

plt.show()
