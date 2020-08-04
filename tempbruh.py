from kitti_dataset import KittiDataset
from kitti_utils import get_pose

if __name__ == "__main__":
    train_config_path = "configs/full_train_dataset.yml"
    data = KittiDataset.init_from_config(train_config_path)

    print("start")
    print(get_pose("data/kitti_data/2011_10_03/2011_10_03_drive_0034_sync", 1757))


    print("end")


    # print(data[0].keys())