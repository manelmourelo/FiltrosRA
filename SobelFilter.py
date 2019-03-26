import cv2
import numpy as np

def sobelFilter():
    # Abrir imagen y coger informacion
    img = cv2.imread("img/Matricula.jpg", cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    #Sobel Kernel
    K_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    K_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

    # Imagenes sin nada
    G_x = np.zeros(img.shape)
    G_y = np.zeros(img.shape)

    # Padding
    pad = np.zeros((rows + 2, cols + 2))
    pad[1:-1, 1:-1] = img
    for i in range(0, rows):
        for j in range(0, cols):
            G_x[i, j] = (pad[i:i+3, j:j+3] * K_x).sum()
            G_y[i, j] = (pad[i:i+3, j:j+3] * K_y).sum()

    # Magnitud del gradiente
    G = np.sqrt(G_x**2, G_y**2)

    # Pintar imagenes
    cv2.imshow("Original", img)
    cv2.imshow("G_x", np.uint8(G_x))
    cv2.imshow("G_y", np.uint8(G_y))
    cv2.imshow("G", np.uint8(G))
    cv2.waitKey(0)

if __name__ == "__main__":
    sobelFilter()