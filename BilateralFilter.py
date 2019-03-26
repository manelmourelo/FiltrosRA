import cv2
import numpy as np

def convolve(dest, src, i, j, kernel, threshold):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]

    pixel = src[i, j]
    sum = np.zeros(3)
    weightsum = np.zeros(3)
    for ki in range(0, krows):
        for kj in range(0, kcols):
            kpixel = srctmp[ki, kj]
            w1 = kernel[ki, kj]
            w2 = np.uint8(np.abs(pixel - kpixel) < threshold)
            sum += kpixel * w1 * w2
            weightsum += w1 * w2

    dest[i, j] = sum / weightsum

def bilateralFilter():
    # Leer imagen y coger tamaño
    img = cv2.imread("img/Icono.jpg", cv2.IMREAD_ANYCOLOR)
    rows, cols, channels = img.shape

    # Crear imagen con padding
    padding = np.zeros([rows + 4, cols + 4, channels])
    padding[2:-2, 2:-2] = img

    # Crear Kernel
    kernel = np.array([
        [1.,  4.,  7.,  4., 1.],
        [4., 16., 26., 16., 4.],
        [7., 26., 41., 26., 7.],
        [4., 16., 26., 16., 4.],
        [1.,  4.,  7.,  4., 1.]
    ])

    # Crear threshold
    threshold = 50

    # Convolve
    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(filtered, padding, i, j, kernel, threshold)

    # Pintar imagen
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(filtered))
    cv2.waitKey(0)

if __name__ == "__main__":
    bilateralFilter()