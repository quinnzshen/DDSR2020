from dataloader import KittiDataset
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
import yaml

warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, config_filename):

        # Config setup
        with open(config_filename) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)\
        # GPU/CPU setup
        self.device = torch.device("cpu")

        # Set up dataloader
        train_config_path = self.config["train_config_path"]
        self.dataset = KittiDataset.init_from_config(train_config_path)

        # Models
        self.models = {}
        self.pretrained = self.config["pretrained"]
        self.models['resnet_encoder'] = ResnetEncoder(self.config["encoder_layers"], pretrained=self.pretrained).to(
            self.device)
        self.scales = range(self.config["num_scales"])
        self.models['depth_decoder'] = DepthDecoder(num_ch_enc=self.models['resnet_encoder'].num_ch_enc,
                                                    scales=self.scales).to(self.device)

        # Parameters
        parameters_to_train = []
        parameters_to_train += list(self.models['resnet_encoder'].parameters())
        parameters_to_train += list(self.models['depth_decoder'].parameters())

        # Optimizer
        learning_rate = self.config["learning_rate"]
        self.optimizer = optim.Adam(parameters_to_train, learning_rate)

        # Learning rate scheduler
        scheduler_step_size = self.config["scheduler_step_size"]
        weight_decay = self.config["weight_decay"]
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, scheduler_step_size, weight_decay)

        # Image dimensions
        self.width = self.config["width"]
        self.height = self.config["height"]

        # Epoch and batch info
        self.num_epochs = self.config["num_epochs"]
        self.batch_size = self.config["batch_size"]

        # Display Predictions
        self.display_predictions = self.config["display_predictions"]

    def train(self):

        self.writer = SummaryWriter()

        for self.epoch in range(self.num_epochs):
            self.run_epoch()
            self.save_model()
            print('Model saved. Epoch {}'.format(self.epoch))

        self.writer.close()

        self.save_model()
        print('Model saved.')

    def run_epoch(self):

        start_time = time.time()
        print("Starting epoch {}".format(self.epoch + 1), end=", ")
        self.lr_scheduler.step()
        self.models['resnet_encoder'].train()
        self.models['depth_decoder'].train()

        start_tracker = 0
        end_tracker = self.batch_size

        num_batches = math.ceil(len(self.dataset) / self.batch_size)

        # Iterate through batches
        for batch_idx in range(num_batches):

            curr_batch_size = end_tracker - start_tracker
            inputs = torch.cat([F.interpolate((torch.tensor(self.dataset[i]["stereo_left_image"].transpose(2, 0, 1),
                                                            device=self.device, dtype=torch.float32).unsqueeze(0)),
                                              [self.height, self.width], mode="bilinear", align_corners=False) for i in
                                range(start_tracker, end_tracker)])

            features = self.models['resnet_encoder'](torch.tensor(inputs))
            outputs = self.models['depth_decoder'](features)
            disp = outputs[("disp", 0)]
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            
            #if self.display_predictions:
            #    display_depth_map(disp, self.height, self.width)
            
            #Tensorboard images
            self.add_disparity_map_to_tensorboard(disp, batch_idx, start_tracker, 0)
 
            _, depths = disp_to_depth(disp, 0.1, 100)
            outputs[("depths", 0)] = depths

            # Source images
            stereo_images = torch.cat([F.interpolate((torch.tensor(
                self.dataset[i]["stereo_right_image"].transpose(2, 0, 1), device=self.device,
                dtype=torch.float32).unsqueeze(0)), [self.height, self.width], mode="bilinear", align_corners=False) for
                                       i in range(start_tracker, end_tracker)])
            temporal_forward_images = torch.cat([F.interpolate((torch.tensor(
                self.dataset[i]["nearby_frames"][1]["camera_data"]["stereo_left_image"].transpose(2, 0, 1),
                device=self.device, dtype=torch.float32).unsqueeze(0)), [self.height, self.width], mode="bilinear",
                                                               align_corners=False) for i in
                                                 range(start_tracker, end_tracker)])
            temporal_backward_images = torch.cat([F.interpolate((torch.tensor(
                self.dataset[i]["nearby_frames"][-1]["camera_data"]["stereo_left_image"].transpose(2, 0, 1),
                device=self.device, dtype=torch.float32).unsqueeze(0)), [self.height, self.width], mode="bilinear",
                                                                align_corners=False) for i in
                                                  range(start_tracker, end_tracker)])
            sources = torch.stack((stereo_images, temporal_forward_images, temporal_backward_images))

            # Poses
            tgt_poses = torch.cat(
                [torch.tensor(self.dataset[i]["pose"], device=self.device, dtype=torch.float32).unsqueeze(0) for i in
                 range(start_tracker, end_tracker)])
            temporal_forward_poses = torch.cat([torch.tensor(self.dataset[i]["nearby_frames"][1]["pose"],
                                                             device=self.device, dtype=torch.float32).unsqueeze(0) for i
                                                in range(start_tracker, end_tracker)])
            temporal_backward_poses = torch.cat([torch.tensor(self.dataset[i]["nearby_frames"][-1]["pose"],
                                                              device=self.device, dtype=torch.float32).unsqueeze(0) for
                                                 i in range(start_tracker, end_tracker)])

            # Relative Poses
            rel_pose_stereo = torch.cat(
                [torch.tensor(self.dataset[i]["rel_pose_stereo"], device=self.device, dtype=torch.float32).unsqueeze(0)
                 for i in range(start_tracker, end_tracker)])
            rel_pose_forward = torch.matmul(torch.inverse(tgt_poses), temporal_forward_poses)
            rel_pose_backward = torch.matmul(torch.inverse(tgt_poses), temporal_backward_poses)
            
            poses = torch.stack((rel_pose_stereo, rel_pose_forward, rel_pose_backward))

            # Intrinsics
            tgt_intrinsics = torch.cat([torch.tensor(self.dataset[i]["intrinsics"]["stereo_left"], device=self.device,
                                                     dtype=torch.float32).unsqueeze(0) for i in
                                        range(start_tracker, end_tracker)])
            src_intrinsics_stereo = torch.cat([torch.tensor(self.dataset[i]["intrinsics"]["stereo_right"],
                                                            device=self.device, dtype=torch.float32).unsqueeze(0) for i
                                               in range(start_tracker, end_tracker)])

            # Adjust intrinsics based on input size
            for i in range(0, curr_batch_size):
                tgt_intrinsics[i][0] = tgt_intrinsics[i][0] * (self.width / 1242)
                tgt_intrinsics[i][1] = tgt_intrinsics[i][1] * (self.height / 375)
                src_intrinsics_stereo[i][0] = src_intrinsics_stereo[i][0] * (self.width / 1242)
                src_intrinsics_stereo[i][1] = src_intrinsics_stereo[i][1] * (self.height / 375)
            
            src_intrinsics = torch.stack((src_intrinsics_stereo, tgt_intrinsics, tgt_intrinsics))
            
            reprojected, mask = process_depth(sources, depths, poses, tgt_intrinsics, src_intrinsics,
                                              (self.height, self.width))

            loss_inputs = {"targets": inputs,
                           "sources": sources
                           }
            loss_outputs = {"reproj": reprojected,
                            "disparities": disp,
                            "initial_masks": mask
                            }

            losses = calc_loss(loss_inputs, loss_outputs)
            self.writer.add_scalar('loss', losses.item(), self.epoch * self.batch_size + batch_idx)
            
            # Back Propogate
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            #Adjust trackers for batches
            if end_tracker == len(self.dataset):
                start_tracker = 0
                end_tracker = self.batch_size
                break
            else:
                start_tracker += self.batch_size

            if (end_tracker + self.batch_size) <= len(self.dataset):
                end_tracker += self.batch_size
            else:
                end_tracker = len(self.dataset)

        end_time = time.time()
        print("Time spent: {}".format(end_time - start_time))
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

    def add_disparity_map_to_tensorboard(self, disp, batch_idx, start_tracker, index):
        disp_resized = torch.nn.functional.interpolate(
        disp[index].unsqueeze(0), (self.height, self.width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = transforms.ToTensor()(colormapped_im)
        self.writer.add_image('Epoch: {}, '.format(self.epoch) + 'Image: {}'.format(start_tracker), im, self.epoch * self.batch_size + batch_idx)

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


def display_depth_map(disp, height, width, index=0):
    disp_resized = torch.nn.functional.interpolate(
        disp[index].unsqueeze(0), (height, width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    plt.figure()
    plt.imshow(im)

test = Trainer("configs/scene_model.yml")
test.train()
plt.show()