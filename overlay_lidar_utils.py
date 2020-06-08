import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import colorsys


def compute_velo2image_plane_matrix(velo2cam, cam2cam, cam):
    """
    This function computes the matrix needed to project the 3D velodyne points onto the 2D image plane
    :param [Dictionary] velo2cam: From calib_velo_to_cam.txt file, contains values needed for extrinsics matrix (rotation, translation)
    :param [Dictionary] cam2cam: From calib_velo_to_cam.txt file, contains values needed for intrinsics matrix (optical center, scaling)
    :param [int] cam: Camera # that matrix is being computed for (0, 1, 2, or 3)
    :return: numpy.array of shape [3,4] that converts 3D velo pts to 2D image plane velo matrix is multiplied by it 
    """
    #based on code from monodepth2 repo
    
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'].reshape(3,1)))#Adds T vals in 4th column
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_0'+str(cam)].reshape(3,3)#Fills top left 3x3 with R vals
    P_rect = cam2cam['P_rect_0' +str(cam)].reshape(3,4)
    
    velo2image_plane = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    
    return velo2image_plane

def generate_velo_pts_im(velo, velo2image_plane, im_width, im_height):
    """
    This function removes the velo pts that are not in the image plane, rounds x/y pixel vals for velo pts, and projects the velo pts onto the image plane
    :param [numpy.array] velo: [N,4], matrix of velo pts, each column is format [X, Y, Z, Reflectance]
    :param [numpy.array] velo2image_plane: [3,4], converts 3D velo pts to 2D image plane velo matrix is multiplied by it 
    :param [int] im_width: width of image in pixels
    :param [int] im_height: height of image in pixels
    :return: numpy.array of shape [N,3], contains velo pts on image plane, each row is format [X, Y, depth]
    """
    #based on code from monodepth2 repo
    
    #remove pts behind camera
    velo = velo[velo[:,0] >=0, :]
    
    #project points to image plane
    velo_pts_im = np.dot(velo2image_plane, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]
    
   
    
    
    # check if in image FOV
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_width) & (velo_pts_im[:, 1] < im_height)
    velo_pts_im = velo_pts_im[val_inds, :]
    

    return velo_pts_im

def render_lidar_on_image(image, velo_pts_im):
    """
    This function plots lidar points on the image with colors corresponding to their depth(higher hsv hue val = further away) 
    :param [numpy.array] velo_pts_im: [N,3], contains velo pts on image plane, each row is format [X, Y, depth]
    :param [numpy.array] image: [H, W], contains image data
    :return: no return val, shows image w/ lidar overlay
    """
    #normalize depth vals
    depth = velo_pts_im[:, 2].reshape(-1,1)
    max = depth.max()
    min = depth.min()
    depth_norm = (depth - min)/(max-min)
    #add normalized depth vals as 4th column
    velo_pts_im = np.hstack((velo_pts_im, depth_norm))
   
    #show grayscale image
    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    plt.imshow(image, cmap='Greys_r')
    
    #plot lidar points
    for row in velo_pts_im:
        norm_depth_val = row[3]
        col = colorsys.hsv_to_rgb(norm_depth_val * (240/360), 1.0, 1.0)
        circ = patches.Circle((int(row[0]),int(row[1])),2)
        circ.set_facecolor(col) 
        ax.add_patch(circ)
    
    plt.show()

