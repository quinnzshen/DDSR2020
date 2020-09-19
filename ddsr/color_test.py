import numpy as np
import cv2
from PIL import Image

from kitti_utils import get_camera_data






d = get_camera_data("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync", 11, is_jpeg=False)
image = np.array(d["stereo_left_image"])
print(image.shape)

himage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

print(himage.shape)


print(d["stereo_left_image"])
















