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
import math
#Filter out warnings, may delete later
import warnings
warnings.filterwarnings("ignore")


#GPU/CPU setup,"cuda:0" if torch.cuda.is_available() else
device = torch.device("cpu")

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

width = 1024
height = 320

start_tracker = 0
end_tracker = batch_size
for epoch in range (epochs):
    start_time = time.time()
    print("Starting epoch {}".format(epoch), end=", ")
    lr_scheduler.step()
    
    depth_encoder.train()
    depth_decoder.train()
    
    num_batches = math.ceil(len(dataset)/batch_size)
    for batch_idx in range(num_batches):
        inputs = torch.cat([F.interpolate((torch.as_tensor(dataset[i]["stereo_left_image"].transpose(2,0,1), device=device, dtype=torch.float32).unsqueeze(0)), [width,height], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)])
        features = depth_encoder(torch.as_tensor(inputs))
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        
        #generate losses - waiting on Evan's reprojection code
        
        optimizer.zero_grad()
        #backprop
        
        if end_tracker == len(dataset):
            start_tracker = 0
            end_tracker = batch_size
            break
        else:
            start_tracker+=batch_size
            
        if (end_tracker+batch_size) <= len(dataset):
            end_tracker += batch_size
        else:
            end_tracker = len(dataset)
    

    end_time = time.time()
    print("Time spent: {}".format(end_time-start_time))  