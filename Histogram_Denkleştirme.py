import cv2
import matplotlib.pyplot as plt

# Görüntüyü yükle
img = cv2.imread('./Pictures/picture2.jpg', 0)

# Histogram dengeleme işlemi
equ = cv2.equalizeHist(img)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Orijinal Görüntü')

plt.subplot(1, 2, 2)
plt.imshow(equ, cmap='gray')
plt.title('Histogram Dengeleme Sonucu')

plt.show()
