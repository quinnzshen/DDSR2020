import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def compute_error(depth, lidar):
    """
    This function computes the metrics of the loss and allows an understanding of how well the model is
    predicting depth compared to the "true" groundtruth LiDAR depth while also returning an array of the L1 loss
    at each LiDAR point
    :param [numpy.array] depth_image_array: a one dimensional array where each value within the array represents the 
    predicted depth of a pixel within the image
    :param [numpy.array] lidar_points: an [N, 4] array containing # of LiDAR points within the image,
    x position, y position, LiDAR depth, and reflectivity
    :return [float] mean_diff: avg abs difference between predicted and LiDAR depth for the entire image
            [numpy.array] final_error_array: an [N, 3] array containing # of LiDAR points within the image, x position,
            y position, and abs difference between predicted and LiDAR depth
    """
    total_diff = 0
    metrics = []
    for x, y, depth in lidar[:, :3]:
        difference = abs(depth[y, x] - depth)
        metrics.append([x, y, difference])
        total_diff += difference
    l1_error = np.array(metrics)
    mean_diff = total_diff / len(lidar)
    return float(mean_diff), l1_error

def plot_depth_error(error_array, image, alpha=.6, s=15):
    """
    This function takes the error_array returned by the compute_error function and overlays on top of
    a grayscaled version of the original image. The combined visualization is then displayed.
    This allows users to see where the predicted depth is accurate and inaccurate compared to the "true" 
    groundtruth LiDAR depth
    :param [numpy.array] error_array: an [N, 3] array containing # of LiDAR points within the image, x position, y position, 
    and abs difference between predicted and LiDAR depth
    :param [numpy.array] image: an array of the original RGB image
    :param [int] alpha: opacity of the grayscale image
    :param [int] s: size of the Xs that are used to plot the error at each of the LiDAR points
    """
    sorted_error_array = np.array(sorted(error_array, key=lambda x:x[2]))
    grayscale_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(20, 20))
    plt.imshow(grayscale_array, cmap='gray', alpha=alpha)
    plt.scatter(sorted_error_array[:, 0], sorted_error_array[:, 1], c=sorted_error_array[:, 2], cmap='plasma', s=s, marker='x')
    plt.show()  


    
        


