import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import imutils
import skimage.segmentation as seg
from sklearn.preprocessing import minmax_scale
from skimage import io, img_as_float32, img_as_float, img_as_ubyte
from skimage.color import label2rgb
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import gaussian
from collections import Counter

def get_first_unvisited(matrix, visited, color):
    rows = matrix.shape[0]
    cols = matrix.shape[1]

    for i in range(rows):
        for j in range(cols):
            if (matrix[i][j] == color) and (visited[i][j] == 0):
                return i, j

    return -1, -1

# Read image and reference
original_image = np.loadtxt(sys.argv[1])
mask = np.loadtxt(sys.argv[2])

# Scale
center = (original_image.min() + original_image.max()) / 2
original_image -= center
original_image /= abs(original_image.min())
original_image = img_as_ubyte(original_image)

plt.subplot(2, 4, 1)
plt.title("Before Preprocesing")
plt.imshow(original_image, cmap="gray")

# Preprocessing

# Enhance image
enhanced_image = cv2.equalizeHist(original_image)

# Smooth image
smoothed_image = cv2.medianBlur(enhanced_image, 3)

# Denoise
denoised_image = cv2.fastNlMeansDenoising(
    src=smoothed_image, dst=None, h=17, templateWindowSize=14, searchWindowSize=25)

# Show after preprocessing
plt.subplot(2, 4, 2)
plt.title("After Preprocesing")
plt.imshow(denoised_image, cmap="gray")

# Segment by Color With K-Means
vectorized_image = denoised_image.reshape((-1, 1))
vectorized_image = img_as_float32(vectorized_image)

K = 10
segmentation_found = 0

while segmentation_found != 1:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        vectorized_image, K, None, criteria, 70, cv2.KMEANS_PP_CENTERS)

    center = img_as_ubyte(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((denoised_image.shape))
    segmented_image = img_as_ubyte(segmented_image)

    # Extract our segment
    intersection = cv2.bitwise_or(
        segmented_image, segmented_image, mask=img_as_ubyte(mask))
    plt.subplot(2, 4, 6)
    plt.title("Intersection")
    plt.imshow(intersection)

    # Extract our dominant color
    non_black_pixels = np.array(intersection).flat
    non_black_pixels = non_black_pixels[non_black_pixels != 0]
    bincount = np.bincount(non_black_pixels)
    dominant_color = bincount.argmax()
    total_pixels = np.sum(mask[mask > 0])
    print(dominant_color)
    print(total_pixels)

    # Extract the organ
    body_part = np.zeros((segmented_image.shape), np.uint8)
    start_x, start_y = get_first_unvisited(
        intersection, body_part, dominant_color)
    print(start_x, start_y)

    selected_pixels = 0
    stack = []
    stack.append((start_x, start_y))
    while stack:
        x, y = stack.pop()
        if segmented_image[x][y] == dominant_color:
            if body_part[x, y] == 0:
                selected_pixels += 1
                body_part[x, y] = 255
                if x > 0:
                    stack.append((x - 1, y))
                if x < 511:
                    stack.append((x + 1, y))
                if y > 0:
                    stack.append((x, y - 1))
                if y < 511:
                    stack.append((x, y + 1))

    if selected_pixels * 100 > 56 * total_pixels:
        segmentation_found = 1
    K -= 1

plt.subplot(2, 4, 3)
plt.title("K-Means Color Segmentation - RGB View")
plt.imshow(label2rgb(segmented_image))

plt.subplot(2, 4, 4)
plt.title("K-Means Color Segmentation - Grayscale View")
plt.imshow(segmented_image)

plt.subplot(2, 4, 5)
plt.title("Mask")
plt.imshow(mask)

plt.subplot(2, 4, 7)
plt.title("Result")
plt.imshow(body_part, cmap="gray")

# Postprocessing

# Enhance image
enhanced_image = cv2.equalizeHist(body_part)

# Smooth image
smoothed_image = cv2.GaussianBlur(enhanced_image, (5, 5), 2)

# Denoise
denoised_image = cv2.fastNlMeansDenoising(
    src=smoothed_image, dst=None, h=10, templateWindowSize=9, searchWindowSize=23)

# Dilated
dilated_image = cv2.dilate(denoised_image, (3, 3),iterations = 5)

# Threshold Filter
thresh_image = cv2.threshold(dilated_image, 180, 255, cv2.THRESH_BINARY)[1]
thresh_image[thresh_image > 0] = 1
np.savetxt("optim.out", thresh_image)

plt.subplot(2, 4, 8)
plt.title("After Postprocessing - FINAL")
plt.imshow(img_as_ubyte(thresh_image))

plt.show()
