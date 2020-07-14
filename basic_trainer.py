from dataloader import KittiDataset
from loss import calc_loss
"""import sys
sys.path.append('third_party\monodepth2')"""
#import third_party.monodepth2.ResnetEncoder
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
import third_party.monodepth2.layers
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time
import os

#Filter out warnings, may delete later
import warnings
warnings.filterwarnings("ignore")


class Trainer:
    

    def __init__(self):
        #GPU/CPU setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Set up dataloader for one frame.
        train_config_path = 'configs/kitti_dataset.yml'
        self.dataset = KittiDataset.init_from_config(train_config_path)[2]

        #Models
        self.models = {}
        self.models['resnet_encoder'] = ResnetEncoder(18, pretrained = False).to(self.device)
        self.models['depth_decoder'] = DepthDecoder(num_ch_enc = self.models['resnet_encoder'].num_ch_enc).to(self.device)
    

        #Parameters
        parameters_to_train = []
        parameters_to_train += list(self.models['resnet_encoder'].parameters())
        parameters_to_train += list(self.models['depth_decoder'].parameters())

        #Optimizer
        learning_rate = 0.0001
        self.optimizer = optim.Adam(parameters_to_train, learning_rate)

        #Scheduler
        scheduler_step_size = 15
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, scheduler_step_size, learning_rate)

    
        """batch_size = 6 #Default is 12
        epochs = 10"""

    def train(self):
        self.width = 640
        self.height = 192
        epochs = 10
        for self.epoch in range (epochs):
            self.run_epoch()
        return self.output
        """self.save_model()
        print('Model saved.')"""
        
    def run_epoch(self):
        start_time = time.time()
        print("Starting epoch {}".format(self.epoch), end=", ")
        self.lr_scheduler.step()
        self.models['resnet_encoder'].train()
        self.models['depth_decoder'].train()
        inputs = torch.cat([F.interpolate((torch.tensor(self.dataset["stereo_left_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.width, self.height], mode = "bilinear", align_corners = False)])
        features = self.models['resnet_encoder'](torch.tensor(inputs))
        self.output = self.models['depth_decoder'](features)
        self.optimizer.zero_grad()

        end_time = time.time()
        print("Time spent: {}".format(end_time-start_time))
    
    def save_model(self): 
        """Save model weights to disk (from monodepth2 repo)
        """
        save_folder = os.path.join("models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'resnet_encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.height
                to_save['width'] = self.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)