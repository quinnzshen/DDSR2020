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
from loss import process_depth, calc_loss
import warnings
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
        self.width = 1280
        self.height = 384
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
            curr_batch_size = end_tracker-start_tracker
            inputs = torch.cat([F.interpolate((torch.tensor(self.dataset[i]["stereo_left_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.height, self.width], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)])

            features = self.models['resnet_encoder'](torch.tensor(inputs))
            outputs = self.models['depth_decoder'](features)
            disp = outputs[("disp", 0)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            display_depth_map(disp, self.height, self.width)
            _, depths = disp_to_depth(disp, 0.1, 100)
            
            #Source images
            stereo_images = torch.cat([F.interpolate((torch.tensor(self.dataset[i]["stereo_right_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.height, self.width], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)])
            temporal_forward_images = torch.cat([F.interpolate((torch.tensor(self.dataset[i]["nearby_frames"][1]["camera_data"]["stereo_left_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.height, self.width], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)])
            temporal_backward_images = torch.cat([F.interpolate((torch.tensor(self.dataset[i]["nearby_frames"][-1]["camera_data"]["stereo_left_image"].transpose(2,0,1), device=self.device, dtype=torch.float32).unsqueeze(0)), [self.height, self.width], mode = "bilinear", align_corners = False) for i in range(start_tracker, end_tracker)])
            sources = torch.stack((stereo_images, temporal_forward_images, temporal_backward_images))
            
            #Poses
            tgt_poses = torch.cat([torch.tensor(self.dataset[i]["pose"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
            temporal_forward_poses = torch.cat([torch.tensor(self.dataset[i]["nearby_frames"][1]["pose"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
            temporal_backward_poses = torch.cat([torch.tensor(self.dataset[i]["nearby_frames"][-1]["pose"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
            
            #Relative Poses
            rel_pose_stereo = torch.cat([torch.tensor(self.dataset[i]["rel_pose_stereo"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
            rel_pose_forward = torch.matmul(torch.inverse(tgt_poses), temporal_forward_poses)
            rel_pose_backward = torch.matmul(torch.inverse(tgt_poses), temporal_backward_poses)
            poses = torch.stack((rel_pose_stereo, rel_pose_forward, rel_pose_backward))
            
            #Intrinsics
            tgt_intrinsics = torch.cat([torch.tensor(self.dataset[i]["intrinsics"]["stereo_left"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
            src_intrinsics_stereo = torch.cat([torch.tensor(self.dataset[i]["intrinsics"]["stereo_right"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in range(start_tracker, end_tracker)])
            src_intrinsics = torch.stack((src_intrinsics_stereo, tgt_intrinsics, tgt_intrinsics))
            
            #Adjust intrinsics based on input size
            for i in range(0, curr_batch_size):
                tgt_intrinsics[i][0] = tgt_intrinsics[i][0] * (self.width / 1242)
                tgt_intrinsics[i][1] = tgt_intrinsics[i][1] * (self.height / 375) 
                src_intrinsics_stereo[i][0] = src_intrinsics_stereo[i][0] * (self.width / 1242)
                src_intrinsics_stereo[i][1] = src_intrinsics_stereo[i][1] * (self.height / 375)
            
            reprojected = process_depth(sources, depths, poses, tgt_intrinsics, src_intrinsics, (self.height, self.width))
            
            loss_inputs = {"targets":inputs,
                           "sources":sources
                }
            loss_outputs = {"reproj":reprojected,
                            "depth":disp
                }
            losses = calc_loss(loss_inputs, loss_outputs)
            
            #Back Propogate - TBD
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

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
        print("Loss: {}".format(losses.item()))


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

def display_depth_map(disp, height, width):
    disp_resized = torch.nn.functional.interpolate(
        disp, (height, width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    plt.figure()    
    plt.imshow(im)
    
test = Trainer()
test.train()