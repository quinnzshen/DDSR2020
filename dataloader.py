import numpy as np
from torch.utils.data import Dataset, DataLoader

import os
import glob


class KittiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.len = -1


    def __len__(self):
        if self.len > 0:
            return self.len

        for subdir, dirs, files in os.walk(self.root_dir):
            # Upon further realization i realize that this approach is stupid and im just keeping this here
            # in case i need to remember this exists

            pass



    def __getitem__(self, item):
        pass


