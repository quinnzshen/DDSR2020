import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import matplotlib as mpl
#If possible, use pip install waymo-open-dataset. If that doesn't work, clone the repo add at it to your path.
sys.path.append("C:/Users/alexj/Documents/GitHub/waymo-od")
from waymo_open_dataset import dataset_pb2 as open_dataset
import waymo_utils as wu
from waymodataloader import WaymoDataset

def plot_sparse_image(lidar_point_coord_camera_image, image, shape):
    """
    Plots the pixels of an image with corresponding depth values
    :param:[numpy.array] lidar_point_coord_camera_image: [N,3], lidar coordinates in camera frame
    :param: [numpy.array] image: [H,W,3], contains R, G, B values for every pixel of the image
    :return: none, plots a sparse image containing pixels with corresponding depth values
    """
    image = wu.conv_to_image(image)
    xs = []
    ys = []
    colors = []

    for point in lidar_point_coord_camera_image:
        if (point[0]<=shape[1] and point[1]<=shape[0]):
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colortuple = (image[int(point[1])][int(point[0])][0], image[int(point[1])][int(point[0])][1], image[int(point[1])][int(point[0])][2])
            colors.append('#%02x%02x%02x' % colortuple)
    plt.figure(figsize=(20, 12)).gca().invert_yaxis()
    plt.scatter(xs,ys,c=colors,s=10,marker='s')

def pixel_to_world(pixel, intrinsics, extrinsics, pose, depth):
    """
    pose_rot = pose[:3,:3]
    pose_trans = pose[:3,3:]
    ext_rot = extrinsics[:3,:3]
    ext_trans = extrinsics[:3,3:]
    pixel = np.concatenate((pixel, [[1]]))
    cam = np.matmul(np.linalg.inv(intrinsics),depth*pixel)
    vehicle = np.matmul(np.linalg.inv(ext_rot), cam)+ext_trans
    world = np.matmul(np.linalg.inv(pose_rot), vehicle)+pose_trans
    [[-7.77104399e+04]
     [-1.29457807e+04]
     [ 3.25182212e+01]]
    ------
    pixel = depth*np.concatenate((pixel, [[1]]))
    camera_coord = np.concatenate((np.matmul(np.linalg.inv(intrinsics), pixel), [[1]]))
    vehicle_coord = np.matmul(np.linalg.inv(extrinsics), camera_coord)
    world = np.matmul(np.linalg.inv(pose), vehicle_coord)
    [[-5.23229696e+04]
     [-6.51712132e+03]
     [ 3.25569636e+01]]
    -----
    pixel = np.concatenate((pixel, [[1]]))
    rot = pose[:3,:3] 
    trans = pose[:3,3:]
    world = np.matmul(np.linalg.inv(rot),np.matmul(np.linalg.inv(intrinsics),pixel))+trans
    [[-7.47743082e+04]
     [-1.16970724e+04]
     [ 3.25092834e+01]]
    -----
    extrinsics = extrinsics[:3,:]
    pose = pose[:3,:]
    pixel = np.concatenate((pixel, [[1]]))
    cam = np.matmul(np.linalg.inv(intrinsics),depth*pixel)
    vehicle = np.matmul(extrinsics, np.concatenate((cam, [[1]])))
    output = np.matmul(pose, np.concatenate((vehicle, [[1]])))
    [[1.37474261e+05]
     [1.34586117e+04]
     [3.37785696e+01]]
    -----
    pixel = np.concatenate((pixel, [[1]]))
    rot = extrinsics[:3,:3]
    trans = extrinsics[:3,3:]
    pixel_to_vehicle = np.matmul(np.linalg.inv(rot),(np.matmul((np.linalg.inv(intrinsics)),(depth*pixel))-trans))
    vehicle_to_world = np.matmul((pose[0:3, 0:3]), pixel_to_vehicle) + pose[:3, 3:]
    [[1.43030492e+05]
     [1.58508770e+04]
     [3.37590930e+01]]
    """
    return world
def world_to_pixel(world_coord, intrinsics, extrinsics, pose):
    """    
    pose_rot = pose[:3,:3]
    pose_trans = pose[:3,3:]
    ext_rot = extrinsics[:3,:3]
    ext_trans = extrinsics[:3,3:]
    vehicle = np.matmul(pose_rot, (world_coord-pose_trans))
    cam = np.matmul(ext_rot, (vehicle-ext_trans))
    image_coord = np.matmul(intrinsics, cam)
    [[-7.77104399e+04]
     [-1.29457807e+04]
     [ 3.25182212e+01]]
    ------
    vehicle_coord = np.matmul(pose, world_coord)
    camera_coord = np.matmul(extrinsics, vehicle_coord)
    image_coord = np.matmul(intrinsics, camera_coord[:3,:])
    [[-5.23229696e+04]
     [-6.51712132e+03]
     [ 3.25569636e+01]]
    ------
    world_coord = np.concatenate((world_coord,[[1]]))
    identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
    transform = np.concatenate((identity, -pose[:3,3:]), axis = 1)
    rot = pose[:3,:3]
    image_coord = np.matmul(intrinsics,np.matmul(rot,np.matmul(transform,world_coord)))
    [[-7.47743082e+04]
     [-1.16970724e+04]
     [ 3.25092834e+01]]
    ------
    pose_rot = pose[:3,:3]
    pose_trans = pose[:3,3:]
    ext_rot = extrinsics[:3,:3]
    ext_trans = extrinsics[:3,3:]
    vehicle = np.matmul(np.linalg.inv(pose_rot), (world_coord-pose_trans))
    cam = np.matmul(np.linalg.inv(ext_rot), (vehicle-ext_trans))
    output = np.matmul(intrinsics, cam)
    [[1.37474261e+05]
     [1.34586117e+04]
     [3.37785696e+01]]
    ------
    rot = pose[:3,:3]
    trans = pose[:3,3:]
    world_to_cam = np.matmul(np.linalg.inv(rot),(world_coord-trans))
    cam_cord = np.concatenate((world_to_cam, [[1]]))
    extrinsics = extrinsics[:3,:]
    image_coord = np.matmul(intrinsics, np.matmul(extrinsics,cam_cord))
    [[1.43030492e+05]
     [1.58508770e+04]
     [3.37590930e+01]]
    """
    output = np.array([image_coord[0]/image_coord[2], image_coord[1]/image_coord[2], image_coord[2]])
    return output

def relative_pose(pose, prev_pose):
    relpose = np.matmul(np.linalg.inv(prev_pose), pose)
    return relpose

def pixel_to_target(K, T, D, pt):
    convto3d = np.concatenate((np.matmul(np.linalg.inv(K),pt), [[1]]))
    K_rescale = np.concatenate((np.concatenate((K,[[0,0,0]])),[[0],[0],[0],[1]]), axis=1)
    return np.matmul(K, np.matmul(T, D*convto3d)[:3,:])
def compute_reprojection_loss(target_frame, source_frame):
    return 2

data = WaymoDataset.init_from_config("waymoloader_test_config.yml")

target = data[1]['frame']
source1 = data[5]['frame']

targetimage=data[1]['front_image']
source1image = data[5]['front_image']

#Attempting to plot a point on target onto source
target_lidar = data[1]['lidar_point_coord']
target_lidar_cp =  data[1]['camera_proj_point_coord']
target_intrinsic = data[1]['front_intrinsics']
target_pose= data[1]['front_pose']
target_extrinsics = data[1]['front_extrinsics']

target_lidar_to_cam = wu.generate_lidar_point_coord_camera_image(target, target_lidar, target_lidar_cp, 'front') 

#target_lidar_to_cam = np.vstack(target_lidar)


#plot_sparse_image(target_lidar_to_cam, data[1]['front_image'], data[1]['front_shape'])
wu.plot_image(data[1]['front_image'])

source1_pose = data[5]['front_pose']
source1_intrinsic = data[5]['front_intrinsics']
source1_extrinsics = data[5]['front_extrinsics']

rel_pose = relative_pose(source1_pose, target_pose)

i = 1000
plt.plot(target_lidar_to_cam[i][0],target_lidar_to_cam[i][1],'ro',markersize=10)
plt.figure()
wu.plot_image(data[5]['front_image'])

point = np.array([[target_lidar_to_cam[i][0]],[target_lidar_to_cam[i][1]]])
world = pixel_to_world(point, target_intrinsic, target_extrinsics, target_pose, target_lidar_to_cam[i][2])    
newpoint = world_to_pixel(world, source1_intrinsic, source1_extrinsics, source1_pose)            
#newpoint= pixel_to_target(target_intrinsic, rel_pose, target_lidar_to_cam[5][2], point)
wu.plot_image(data[5]['front_image'])
print(target_lidar_to_cam[i][2])
print(point)
print(newpoint)
plt.plot(newpoint[0],newpoint[1], 'ro',markersize=10)
