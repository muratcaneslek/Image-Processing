import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Salt gürültüsü ekleme
    salt_pixels = int(total_pixels * salt_prob)
    salt_coordinates = [np.random.randint(0, i-1, salt_pixels) for i in image.shape]
    noisy_image[salt_coordinates[0], salt_coordinates[1], :] = 1

    # Pepper gürültüsü ekleme
    pepper_pixels = int(total_pixels * pepper_prob)
    pepper_coordinates = [np.random.randint(0, i-1, pepper_pixels) for i in image.shape]
    noisy_image[pepper_coordinates[0], pepper_coordinates[1], :] = 0

    return noisy_image

img = cv2.imread('./Pictures/newspaper.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

salt_prob = 0.02
pepper_prob = 0.02
noisy_img = add_salt_and_pepper_noise(img, salt_prob, pepper_prob)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Orijinal Görüntü')

plt.subplot(1, 2, 2)
plt.imshow(noisy_img)
plt.title('Salt and Pepper Gürültülü Görüntü')

plt.show()
