import numpy as np
import cv2
from PIL import Image

from kitti_utils import get_camera_data



pilimage = np.array(Image.open("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000000.png"))

d = get_camera_data("data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync", 11, is_jpeg=False)
image = np.array(d["stereo_left_image"])
image2 = (np.array(d["stereo_left_image"]) * 255)

himage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
himage2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)

np.set_printoptions(suppress=True)

print("zero to one")
print(himage[205:210, 600:605])
print(himage.max(0).max(0))

print("255")
print(himage2[205:210, 600:605])
print(himage2.max(0).max(0))


cv2.imshow("ble", cv2.cvtColor(himage, cv2.COLOR_HSV2BGR))
cv2.waitKey(0)














