import cv2
import numpy as np
import matplotlib.pyplot as plt

def bit_plane_slice(image_path, bit_level):
    # Görüntüyü oku
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Belirli bit seviyesini al
    bit_plane = (image >> bit_level) & 1

    # Görselleştirme
    plt.imshow(bit_plane, cmap='gray')
    plt.title(f'Bit Plane {bit_level}')
    plt.axis('off')
    plt.show()

# Örnek kullanım
image_path = './Pictures/picture1.jpg'
bit_level = 7  # İstediğiniz bit seviyesi
bit_plane_slice(image_path, bit_level)
