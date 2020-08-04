from kitti_dataset import KittiDataset

if __name__ == "__main__":
    train_config_path = "configs/full_train_dataset.yml"
    data = KittiDataset.init_from_config(train_config_path)

    print(data[0].keys())