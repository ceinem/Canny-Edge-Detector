# Implementation of the Canny Edge Detector
import numpy as np
import scipy
import scipy.misc
from scipy.ndimage.filters import gaussian_filter, convolve
import matplotlib.pyplot as plt
from scipy import *
from scipy.ndimage import *

# Import Image
im = scipy.misc.imread('input_image.png', mode='F')

# Plot Imported Image and find out dimensions
# print(im.shape)
# plt.figure(figsize=(7, 7))
# plt.imshow(im, cmap='gray')
# plt.show()

# Smooth Image using Gaussian filter
sigma = 2.0
blurred_image = gaussian_filter(im, sigma)

# Determine Gradients using Sobel Operator
sobel_operator_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_operator_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

im_x = convolve(blurred_image, sobel_operator_x)
im_y = convolve(blurred_image, sobel_operator_y)

# Compute the Magnitue of the gradient, as well as its direction
gradient_magnitude = np.sqrt(im_x**2 + im_y**2)

theta = arctan2(im_y, im_x)

# Edge Detection according to threshold criterion
threshold = 30
threshold_Edges = (gradient_magnitude > threshold)

# Plotting for testing
# plt.figure(figsize=(7, 7))
# plt.imshow(threshold_Edges, cmap='gray')
# plt.show()


# Implement Non-Maximum Suppression for finer boundaries
# First discretize theta

theta = theta * 180 / np.pi
theta = np.absolute(around(theta / 45.0) * 45.0)

# Apply Non - Maximum Suppression
h, w = theta.shape[:2]
# Initialize the Edge Detection as Black, so no Edges
Edges = np.zeros(theta.shape)
for y in range(1, h-2):
    for x in range(1, w-2):
        # Each time the gradient reaches above a threshold, check for the gradient direction and add the appropriate Edge to the finished Image
        if threshold_Edges[y, x] == 1:
            if (theta[y, x] == 0) | (theta[y, x] == 180):
                if (gradient_magnitude[y, x] >= gradient_magnitude[y, x-1]) & (gradient_magnitude[y, x] >= gradient_magnitude[y, x+1]):
                    Edges[y, x] = 1
            elif theta[y, x] == 45:
                if (gradient_magnitude[y, x] >= gradient_magnitude[y-1, x+1]) & (gradient_magnitude[y, x] >= gradient_magnitude[y+1, x-1]):
                    Edges[y, x] = 1
            elif theta[y, x] == 90:
                if (gradient_magnitude[y, x] >= gradient_magnitude[y-1, x]) & (gradient_magnitude[y, x] >= gradient_magnitude[y+1, x]):
                    Edges[y, x] = 1
            elif theta[y, x] == 135:
                if (gradient_magnitude[y, x] >= gradient_magnitude[y-1, x-1]) & (gradient_magnitude[y, x] >= gradient_magnitude[y+1, x+1]):
                    Edges[y, x] = 1
            else:
                print("ERROR")
                print(theta[y, x])


# Save Detected Edges as new image
scipy.misc.imsave('canny_edges.png', Edges)
