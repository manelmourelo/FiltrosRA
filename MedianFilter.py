import cv2
import numpy as np

def median(dest, src, i, j, kernelSize):
    dest[i, j] = np.median(src[i:i + kernelSize, j:j + kernelSize], axis=(0, 1))

def medianFilter():
    # Leer imagen y coger tamaño
    img = cv2.imread("img/Icono.jpg", cv2.IMREAD_ANYCOLOR)
    rows, cols, channels = img.shape

    # Crear padding
    padding = np.zeros([rows + 4, cols + 4, channels])
    padding[2:-2, 2:-2] = img

    # Median
    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            median(filtered, padding, i, j, 5)

    # Mostrar imagen
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(filtered))
    cv2.waitKey(0)

if __name__ == "__main__":
    medianFilter()