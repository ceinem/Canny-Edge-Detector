# Implementation of the Canny Edge Detector
import numpy as np
import scipy
import scipy.misc
from scipy.ndimage.filters import gaussian_filter, convolve
import matplotlib.pyplot as plt
from scipy import *
from scipy.ndimage import *

# Import Image
im = scipy.misc.imread('eurowings.jpg', mode='F')

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
