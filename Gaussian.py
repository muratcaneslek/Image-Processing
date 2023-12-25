import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_filter(image, kernel_size):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

img = cv2.imread('./Pictures/picture2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel_size = 5
blurred_img = apply_gaussian_filter(img, kernel_size)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Orijinal Görüntü')

plt.subplot(1, 2, 2)
plt.imshow(blurred_img)
plt.title(f'Gaussian Filtre (Kernel Size = {kernel_size})')

plt.show()
