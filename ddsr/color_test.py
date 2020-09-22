import numpy as np
import cv2
from PIL import Image

from kitti_utils import get_camera_data



pilimage = np.array(Image.open("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000000.png"))
print(pilimage)

d = get_camera_data("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync", 11, is_jpeg=False)
image = np.array(d["stereo_left_image"]) * 255
print(image.shape)
print(image)
himage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

print(himage.shape)


print(himage)
print(himage.max(0).max(0))
















