import cv2
import numpy as np

def convolve(dest, src, i, j, kernel):
    kernelrows, kernelcols = kernel.shape
    srctmp = src[i:i + kernelrows, j:j + kernelcols]
    dest[i,j] = (srctmp * kernel[:, :, np.newaxis]).sum(axis=(0, 1))

def gaussianFilter():
    # Leer la imagen y coger su tamaño
    img = cv2.imread("img/Icono.jpg", cv2.IMREAD_ANYCOLOR)
    rows, cols, channels = img.shape

    # Crear imagen con padding
    padding = np.zeros((rows + 4, cols + 4, channels))
    padding[2:-2, 2:-2] = img

    # Crear Kernel
    kernel = np.array([
        [1., 4.,  7.,  4,   1.],
        [4., 16., 26., 16., 4.],
        [7., 26., 41., 26., 7.],
        [4., 16., 26., 16., 4.],
        [1., 4.,  7.,  4.,  1.]
    ])

    # Convololve
    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(filtered, padding, i, j, kernel)
    filtered /= kernel.sum()

    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(filtered))
    cv2.waitKey(0)

if __name__ == "__main__":
    gaussianFilter()