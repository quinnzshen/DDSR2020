import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import metrics as m

def compute_l1_error(depth, lidar):
    """
    This function computes the metrics of the loss and allows an understanding of how well the model is
    predicting depth compared to the "true" groundtruth LiDAR depth while also returning an array of the L1 loss
    at each LiDAR point
    :param [numpy.array] depth: a one dimensional array where each value within the array represents the 
    predicted depth of a pixel within the image
    :param [numpy.array] lidar: an [N, 4] array containing # of LiDAR points within the image,
    x position, y position, LiDAR depth, and reflectivity
    :return [numpy.array] l1_error: an [N, 3] array containing # of LiDAR points within the image, x position,
    y position, and abs difference between predicted and LiDAR depth
    """
    metrics = []
    for x, y, d in lidar[:, :3]:
        difference = abs(depth[y][x] - d)
        metrics.append([x, y, difference])
    l1_error = np.array(metrics)
    return l1_error

def plot_depth_error(error_array, image, fig_h=20, fig_w=20, alpha=.6, s=15):
    """
    This function takes the error_array returned by the compute_error function and overlays on top of
    a grayscaled version of the original image. The combined visualization is then displayed.
    This allows users to see where the predicted depth is accurate and inaccurate compared to the "true" 
    groundtruth LiDAR depth
    :param [numpy.array] error_array: an [N, 3] array containing # of LiDAR points within the image, x position, y position, 
    and abs difference between predicted and LiDAR depth
    :param [numpy.array] image: an array of the original RGB image
    :param [int] fig_h: the height of the figure that will be displayed	
    :param [int] fig_w: the width of the figure that will be displayed
    :param [int] alpha: opacity of the grayscale image
    :param [int] s: size of the Xs that are used to plot the error at each of the LiDAR points
    """
    sorted_error_array = np.array(sorted(error_array, key=lambda x:x[2]))
    grayscale_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(fig_h, fig_w))
    plt.imshow(grayscale_array, cmap='gray', alpha=alpha)
    plt.scatter(sorted_error_array[:, 0], sorted_error_array[:, 1], c=sorted_error_array[:, 2], cmap='plasma', s=s, marker='x')


    
        


