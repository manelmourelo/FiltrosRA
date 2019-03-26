import cv2
import numpy as np

def convolve(dest, src, i, j, kernel):
    kernelrows, kernelcols = kernel.shape
    srctmp = src[i:i + kernelrows, j:j + kernelcols]
    dest[i, j] = (srctmp * kernel[:, :, np.newaxis]).sum(axis=(0, 1))

def boxFilter():
    # Leer la imagen y coger tamaño
    img = cv2.imread("img/Icono.jpg", cv2.IMREAD_ANYCOLOR)
    rows, cols, _ = img.shape

    # Crear imagen con padding
    padding = np.zeros((rows+4, cols+4, 3))
    padding[2:-2, 2:-2] = img

    # Crear Kernel
    kernel = np.ones((5, 5))

    # Convolution
    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(filtered, padding, i, j, kernel)
    filtered /= kernel.sum()

    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(filtered))
    cv2.waitKey(0)

if __name__ == "__main__":
    boxFilter()