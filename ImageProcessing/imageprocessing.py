import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('D:\\Computer-Vision\\ImageProcessing\\child.jpg').astype(np.float32) / 255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

def gamma_correction(image, gamma=1/7):
    gamma_img = np.power(image, gamma)
    plt.imshow(gamma_img)
    plt.show()
    return gamma_img

def prewitt_edge_detection(image):
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    ky = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])
    gradient_x = cv2.filter2D(image, -1, kx)
    gradient_y = cv2.filter2D(image, -1, ky)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.show()
    return gradient_magnitude

def dct_transform(block):
    return cv2.dct(np.float32(block))

def perform_dct(image):
    M, N = image.shape
    transformed_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(0, M, 8):
        for j in range(0, N, 8):
            block = image[i:i+8, j:j+8]
            transformed_block = dct_transform(block)
            transformed_image[i:i+8, j:j+8] = transformed_block
    
    return transformed_image

gamma_corrected_image = gamma_correction(img)
prewitt_edges = prewitt_edge_detection(img)

# Convert the image to grayscale
image = cv2.imread('D:\\Computer-Vision\\ImageProcessing\\child.jpg', cv2.IMREAD_GRAYSCALE)
transformed_image = perform_dct(image)
cv2.imshow('DCT Transform', np.uint8(transformed_image))


gamma_corrected_image_normalized = cv2.normalize(gamma_corrected_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
prewitt_edges_normalized = cv2.normalize(prewitt_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imwrite('gamma_corrected_image.jpg', cv2.cvtColor(gamma_corrected_image_normalized, cv2.COLOR_RGB2BGR))
cv2.imwrite('prewitt_edges.jpg', prewitt_edges_normalized)
cv2.imwrite('dct_transformed_image.jpg', np.uint8(transformed_image))



cv2.waitKey(0)
cv2.destroyAllWindows()