from dataloader import KittiDataset
from loss import calc_loss
import sys
sys.path.append('third_party\monodepth2')
import ResnetEncoder, DepthDecoder, layers
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

#Set up dataloader
train_config_path = 'configs/kitti_dataset.yml'
dataset = KittiDataset.init_from_config(train_config_path)

#Models
depth_encoder = ResnetEncoder.ResnetEncoder(18, pretrained = False)
depth_decoder = DepthDecoder.DepthDecoder(num_ch_enc = depth_encoder.num_ch_enc)

#Parameters
parameters_to_train = []
parameters_to_train += list(depth_encoder.parameters())
parameters_to_train += list(depth_decoder.parameters())

#Optimizer
learning_rate = 0.0001
optimizer = optim.Adam(parameters_to_train, learning_rate)

#Scheduler
scheduler_step_size = 15
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, learning_rate)

batch_size = 6 #Default is 12
epochs = 10

start_time = time.time()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for epoch in range (epochs):
    lr_scheduler.step()
    print("Training")
    depth_encoder.train()
    depth_decoder.train()
    
    inputs = torch.cat([torch.tensor(dataset[i]["stereo_left_image"], device=device) for i in range(len(dataset))])
    
        