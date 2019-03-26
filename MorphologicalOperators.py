import cv2
import numpy as np

def erode(img, radius):
    # Coger informacion de la imagen
    ksize = 2 * radius + 1
    karea = ksize * ksize
    rows, cols = img.shape

    # Crear padding
    padding = np.zeros((rows + 2 * radius, cols + 2 * radius))
    padding[radius:-radius, radius:-radius] = img

    # Convolution
    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            black_count = np.uint8(padding[i:i + ksize, j:j + ksize] == 0).sum()
            filtered[i, j] = 1.0 - np.uint8(black_count > 0)

    filtered *= 255
    return filtered

def dilate(img, radius):
    # Coger informacion de la imagen
    ksize = 2 * radius + 1
    karea = ksize * ksize
    rows, cols = img.shape

    # Crear padding
    padding = np.zeros((rows + 2 * radius, cols + 2 * radius))
    padding[radius:-radius, radius:-radius] = img

    # Convolution
    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            filtered[i, j] = np.uint8(padding[i:i+ksize, j:j+ksize].sum() > 0)

    filtered *= 255
    return filtered

def opening(img, radius):
    img = erode(img, radius)
    img = dilate(img, radius)
    return img

def closing(img, radius):
    img = dilate(img, radius)
    img = erode(img, radius)
    return img

def execute():
    # Cargar la imagen
    img = cv2.imread("img/j.png", cv2.IMREAD_GRAYSCALE)

    # Filtrar
    filtered = erode(img, 1)
    filtered2 = dilate(img, 1)

    # Pintar imagenes
    cv2.imshow("Original", img)
    cv2.imshow("Erode", np.uint8(filtered))
    cv2.imshow("Dilate", np.uint8(filtered2))

    cv2.waitKey(0)

if __name__ == "__main__":
    execute()