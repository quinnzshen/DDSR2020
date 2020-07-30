import math
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import yaml

from collate import TrainerCollator
from dataloader import KittiDataset
from loss import process_depth, calc_loss
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder


class Trainer:
    def __init__(self, config_path):
        """
        Creates an instance of tranier using a config file
        The config file contains all the information needed to train a model
        :param [str] config_path: The path to the config file
        :return [Trainer]: Object instance of the trainer
        """

        # Load data from config
        with open(config_path) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)

        # GPU/CPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Epoch and batch info
        self.num_epochs = self.config["num_epochs"]
        self.batch_size = self.config["batch_size"]

        # Image dimensions
        self.width = self.config["width"]
        self.height = self.config["height"]

        # Dataloader setup
        train_config_path = self.config["train_config_path"]
        self.dataset = KittiDataset.init_from_config(train_config_path)
        self.collate = TrainerCollator(self.height, self.width)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate)

        # Neighboring frames
        self.prev_frames = self.dataset.previous_frames
        self.next_frames = self.dataset.next_frames

        # Model setup
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

        # Writer for tensorboard
        self.writer = SummaryWriter()

    def train(self):
        """
        Runs the entire training pipeline
        Saves the model's weights at the end of training
        """

        for self.epoch in range(self.num_epochs):
            self.run_epoch()

        self.writer.close()

        self.save_model()
        print('Model saved.')

    def run_epoch(self):
        """
        Runs a single epoch of training
        """

        start_time = time.time()

        print("Starting epoch {}".format(self.epoch + 1), end=", ")

        self.models['resnet_encoder'].train()
        self.models['depth_decoder'].train()

        img_num = 1

        num_batches = math.ceil(len(self.dataset) / self.batch_size)
        for batch_idx, item in enumerate(self.dataloader):
            # Predict disparity map for images in batch
            inputs = F.interpolate(item["stereo_left_image"].to(self.device).permute(0, 3, 1, 2).float(), (self.height, self.width), mode="bilinear", align_corners=False)
            print(item["stereo_left_image"].shape)
            features = self.models['resnet_encoder'](inputs)
            outputs = self.models['depth_decoder'](features)
            disp = outputs[("disp", 0)]
            print(disp.shape)
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)

            # Add disparity map of the first image in each batch to tensorboard
            self.add_disparity_map_to_tensorboard(disp, img_num, num_batches, batch_idx)

            # Convert disparity to depth
            _, depths = disp_to_depth(disp, 0.1, 100)
            outputs[("depths", 0)] = depths

            tgt_poses = item["pose"].to(self.device)
            sources_list = [F.interpolate(item["stereo_right_image"].to(self.device).permute(0, 3, 1, 2).float(), (self.height, self.width), mode="bilinear", align_corners=False)]
            print(item["stereo_right_image"].shape)
            poses_list = [item["rel_pose_stereo"].to(self.device)]

            for i in range(-self.prev_frames, self.next_frames + 1):
                if i == 0:
                    continue

                # Source images and poses
                sources_list.append(F.interpolate(item["nearby_frames"][i]["camera_data"]["stereo_left_image"].to(self.device).permute(0, 3, 1, 2).float(), (self.height, self.width), mode="bilinear", align_corners=False))
                poses_list.append(
                    torch.matmul(torch.inverse(tgt_poses), item["nearby_frames"][i]["pose"].to(self.device)))

            # Stacking to turn into tensors
            sources = torch.stack(sources_list, dim=0)
            poses = torch.stack(poses_list, dim=0)
            print(sources.shape)
            print(inputs.shape)
            # Intrinsics
            tgt_intrinsics = item["intrinsics"]["stereo_left"].to(self.device)
            src_intrinsics_stereo = item["intrinsics"]["stereo_right"].to(self.device)

            # Adjust intrinsics based on input size
            tgt_intrinsics[:, 0] = tgt_intrinsics[:, 0] * (self.width / 1242)
            tgt_intrinsics[:, 1] = tgt_intrinsics[:, 1] * (self.height / 375)
            src_intrinsics_stereo[:, 0] = src_intrinsics_stereo[:, 0] * (self.width / 1242)
            src_intrinsics_stereo[:, 1] = src_intrinsics_stereo[:, 1] * (self.height / 375)

            intrinsics = [src_intrinsics_stereo]
            for i in range(len(poses_list) - 1):
                intrinsics.append(tgt_intrinsics)
            src_intrinsics = torch.stack(intrinsics)

            reprojected, mask = process_depth(sources, depths, poses, tgt_intrinsics, src_intrinsics,
                                              (self.height, self.width))
            # Compute Losses
            loss_inputs = {"targets": inputs,
                           "sources": sources
                           }
            loss_outputs = {"reproj": reprojected,
                            "disparities": disp,
                            "initial_masks": mask
                            }
            losses = calc_loss(loss_inputs, loss_outputs)

            # Add loss to tensorboard
            self.writer.add_scalar('loss', losses.item(), self.epoch * num_batches + batch_idx)

            # Back Propogate
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            img_num += self.batch_size

        self.lr_scheduler.step()

        end_time = time.time()
        print("Time spent: {}".format(end_time - start_time))
        print("Loss: {}".format(losses.item()))

    def save_model(self):
        """
        Saves model weights to disk (from monodepth2 repo)
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

    def add_disparity_map_to_tensorboard(self, disp, img_num, num_batches, batch_idx):
        """
        Adds output disparity map to tensorboard
        :param [tensor] disp: The disparity map outputted by the network
        :param [int] img_num: The index of the input image in the training data file
        :param [int] num_batches: The number of batches in each epoch
        :param [int] batch_idx: The current batch number
        """
        disp_resized = torch.nn.functional.interpolate(
            disp[0].unsqueeze(0), (self.height, self.width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = transforms.ToTensor()(colormapped_im)
        self.writer.add_image('Epoch: {}, '.format(self.epoch + 1) + 'Image: {}'.format(img_num), im,
                              self.epoch * num_batches + batch_idx)


def disp_to_depth(disp, min_depth, max_depth):
    """
    Converts network's sigmoid output into depth prediction (from monodepth 2 repo)
    The formula for this conversion is given in the 'additional considerations'
    section of the paper
    :param [tensor] disp: The disparity map outputted by the network
    :param [int] min_depth: The minimum depth value
    :param [int] max_depth: The maximum depth value
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


if __name__ == "__main__":
    test = Trainer("configs/oneframe_model.yml")
    test.train()
