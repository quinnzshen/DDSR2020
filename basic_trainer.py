from dataloader import KittiDataset
from loss import calc_loss
import sys
sys.path.append('third_party\monodepth2')
import ResnetEncoder, DepthDecoder, layers
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time

#Filter out warnings, may delete later
import warnings
warnings.filterwarnings("ignore")


#GPU/CPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Set up dataloader
train_config_path = 'configs/kitti_dataset.yml'
dataset = KittiDataset.init_from_config(train_config_path)

#Models
depth_encoder = ResnetEncoder.ResnetEncoder(18, pretrained = False).to(device)
depth_decoder = DepthDecoder.DepthDecoder(num_ch_enc = depth_encoder.num_ch_enc).to(device)

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


width = 640
height = 192
for epoch in range (epochs):
    start_time = time.time()
    print("Starting epoch {}".format(epoch), end=", ")
    lr_scheduler.step()
    depth_encoder.train()
    depth_decoder.train()
    inputs = torch.cat([F.interpolate((torch.tensor(dataset[i]["stereo_left_image"].transpose(2,0,1), device=device, dtype=torch.float32).unsqueeze(0)), [width,height], mode = "bilinear", align_corners = False) for i in range(len(dataset))])
    features = depth_encoder(torch.tensor(inputs))
    output = depth_decoder(features)
    
    optimizer.zero_grad()

    end_time = time.time()
    print("Time spent: {}".format(end_time-start_time))
    