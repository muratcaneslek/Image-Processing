import cv2
import matplotlib.pyplot as plt

def apply_gaussian_blur(image, kernel_size):
    # Gaussian filtresi uygulama
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image

img = cv2.imread('./Pictures/picture2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Gaussian filtresi uygulama
kernel_size = 5  # Filtre çekirdeği boyutu (tek sayı olmalıdır)
blurred_img = apply_gaussian_blur(img, kernel_size)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Orijinal Görüntü')

plt.subplot(1, 2, 2)
plt.imshow(blurred_img)
plt.title(f'Gaussian Filtre (Kernel Size = {kernel_size})')

plt.show()
