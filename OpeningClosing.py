import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_opening(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening_result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening_result

def apply_closing(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing_result

img = cv2.imread('./Pictures/picture1.jpg', 0)

opening_result = apply_opening(img, kernel_size=5)

closing_result = apply_closing(img, kernel_size=5)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Orijinal Görüntü')

plt.subplot(1, 3, 2)
plt.imshow(opening_result, cmap='gray')
plt.title('Opening Uygulaması')

plt.subplot(1, 3, 3)
plt.imshow(closing_result, cmap='gray')
plt.title('Closing Uygulaması')

plt.show()
