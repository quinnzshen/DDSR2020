from dataloader import KittiDataset
from loss import calc_loss
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
import third_party.monodepth2.layers
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import time
import math
import os
from compute_photometric_error_utils import compute_relative_pose_matrix, reproject_source_to_target
import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self):
        #GPU/CPU setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Set up dataloader for one frame.
        train_config_path = 'configs/oneframe_overfit.yml'
        self.dataset = KittiDataset.init_from_config(train_config_path)
        
        #Models
        self.models = {}
        self.models['resnet_encoder'] = ResnetEncoder(18, pretrained = False).to(self.device)
        self.scales = range(1)
        self.models['depth_decoder'] = DepthDecoder(num_ch_enc = self.models['resnet_encoder'].num_ch_enc, scales=self.scales).to(self.device)

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
        
    def train(self):
        self.width = 1242
        self.height = 375
        epochs = 10
        for self.epoch in range (epochs):
            self.run_epoch()
        #return self.output
        """self.save_model()
        print('Model saved.')"""
        
    def run_epoch(self):
        start_time = time.time()
        print("Starting epoch {}".format(self.epoch+1), end=", ")
        self.lr_scheduler.step()
        self.models['resnet_encoder'].train()
        self.models['depth_decoder'].train()
        
        batch_size = 1
        start_tracker = 0
        end_tracker = batch_size
        num_batches = math.ceil(len(self.dataset)/batch_size)
        #Iterate through batches
        for batch_idx in range(num_batches):
            inputs = {"images" : torch.cat([F.interpolate((torch.tensor(self.dataset[i]["stereo_left_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.width, self.height], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)]),
                      "intrinsics" : torch.cat([torch.tensor(self.dataset[i]["intrinsics"]["stereo_left"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
                      }
            inputs_stereo = {"images" : torch.cat([F.interpolate((torch.tensor(self.dataset[i]["stereo_right_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.width, self.height], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)]),
                      "intrinsics" : torch.cat([torch.tensor(self.dataset[i]["intrinsics"]["stereo_right"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
                      }                           
            inputs_temporal_forward = {"images" : torch.cat([F.interpolate((torch.tensor(self.dataset[i]["nearby_frames"][1]["camera_data"]["stereo_left_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.width, self.height], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)]),
                                       "intrinsics" : torch.cat([torch.tensor(self.dataset[i]["nearby_frames"][1]["camera_data"]["intrinsics"]["stereo_left"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
                      }
            inputs_temporal_backward = torch.cat([F.interpolate((torch.tensor(self.dataset[i]["nearby_frames"][-1]["camera_data"]["stereo_left_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.width, self.height], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)])
            
            features = self.models['resnet_encoder'](torch.tensor(inputs["images"]))
            self.outputs = self.models['depth_decoder'](features)
            self.disp = self.outputs[("disp", 0)]
            
            #Generate Losses - Waiting on Evan's reprojection code
            self.generate_images_pred(inputs, self.outputs)
            self.optimizer.zero_grad()

            #Back Propogate - TBD
            

            if end_tracker == len(self.dataset):
                start_tracker = 0
                end_tracker = batch_size
                break
            else:
                start_tracker+=batch_size
                
            if (end_tracker+batch_size) <= len(self.dataset):
                end_tracker += batch_size
            else:
                end_tracker = len(self.dataset)
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
            
def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth            

test = Trainer()
test.train()