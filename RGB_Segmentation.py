import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_segmentation(image, lower_bound, upper_bound):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    return segmented_image

img = cv2.imread('./Pictures/picture1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lower_bound = np.array([100, 50, 50])
upper_bound = np.array([140, 255, 255])

segmented_img = rgb_segmentation(img, lower_bound, upper_bound)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Orijinal Görüntü')

plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title('RGB Bölütlenmiş Görüntü')

plt.show()
