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

from collate import Collator
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
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"

        # Epoch and batch info
        self.num_epochs = self.config["num_epochs"]
        self.batch_size = self.config["batch_size"]

        # Image dimensions
        self.width = self.config["width"]
        self.height = self.config["height"]

        # Dataloader Setup
        self.collate = Collator(self.height, self.width)

        train_config_path = self.config["train_config_path"]
        self.train_dataset = KittiDataset.init_from_config(train_config_path)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                           collate_fn=self.collate)

        val_config_path = self.config["valid_config_path"]
        self.val_dataset = KittiDataset.init_from_config(val_config_path)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                         collate_fn=self.collate)

        test_config_path = self.config["test_config_path"]
        self.test_dataset = KittiDataset.init_from_config(test_config_path)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                          collate_fn=self.collate)

        # Neighboring frames
        self.prev_frames = self.train_dataset.previous_frames
        self.next_frames = self.train_dataset.next_frames

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
        self.num_batches_train = math.ceil(len(self.train_dataset) / self.batch_size)

        for self.epoch in range(self.num_epochs):
            self.run_epoch()

        # Calculating loss on test data
        self.img_num = 1

        total_loss = count = 0
        for batch_idx, batch in enumerate(self.test_dataloader):
            with torch.no_grad():
                count += 1
                total_loss += self.process_batch(batch_idx, batch, len(self.test_dataset), False).item()
        total_loss /= count
        self.writer.add_scalar("Testing" + ' Loss', total_loss, 0)
        print(f"Testing Loss: {total_loss}")

        self.writer.close()
        self.save_model()
        print('Model saved.')

    def run_epoch(self):
        """
        Runs a single epoch of training and validation
        """
        
        # Training
        train_start_time = time.time()

        print(f"Training epoch {self.epoch + 1}", end=", ")

        self.models['resnet_encoder'].train()
        self.models['depth_decoder'].train()

        self.img_num = 1
        
        total_loss = count = 0
        for batch_idx, batch in enumerate(self.train_dataloader):
            count += 1
            loss = self.process_batch(batch_idx, batch, len(self.train_dataset), True).item()
            self.writer.add_scalar("Training" + ' Loss', loss, self.epoch * self.num_batches_train + batch_idx)
            total_loss += loss

        total_loss /= count

        self.lr_scheduler.step()

        train_end_time = time.time()

        print(f"Training Loss: {total_loss}")
        print(f"Time spent: {train_end_time - train_start_time}")

        # Validation
        val_start_time = time.time()

        print(f"Validating epoch {self.epoch + 1}", end=", ")

        self.models['resnet_encoder'].eval()
        self.models['depth_decoder'].eval()

        self.img_num = 1

        total_loss = count = 0
        for batch_idx, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                count += 1
                total_loss += self.process_batch(batch_idx, batch, len(self.val_dataset), False).item()
        total_loss /= count

        self.writer.add_scalar("Validation" + ' Loss', total_loss, self.epoch)

        val_end_time = time.time()

        print(f"Validation Loss: {total_loss}")
        print(f"Time spent: {val_end_time - val_start_time}\n")

    def process_batch(self, batch_idx, batch, dataset_length, train):
        """
        Computes loss for a single batch
        :param [int] batch_idx: The batch index
        :param [dict] batch: The batch data
        :param [int] img_num: The index of the input image in the training/validation/testing file
        :param [int] dataset_length: The length of the training/validation/testing dataset
        :param [boolean] train: Differentiates between training and evaluation
        :return [tensor] losses: A 0-dimensional tensor containing the loss of the batch
        """

        # Predict disparity map
        inputs = F.interpolate(batch["stereo_left_image"].to(self.device).permute(0, 3, 1, 2).float(),
                               (self.height, self.width), mode="bilinear", align_corners=False)
        features = self.models['resnet_encoder'](inputs)
        outputs = self.models['depth_decoder'](features)
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)

        # Add disparity map to tensorboard
        for i, disp_map in enumerate(disp):
            self.add_disparity_map_to_tensorboard(disp_map, self.img_num + i, dataset_length, train)

        # Convert disparity to depth
        _, depths = disp_to_depth(disp, 0.1, 100)
        outputs[("depths", 0)] = depths

        # Source image and pose data
        tgt_poses = batch["pose"].to(self.device)
        sources_list = [F.interpolate(batch["stereo_right_image"].to(self.device).permute(0, 3, 1, 2).float(),
                                      (self.height, self.width), mode="bilinear", align_corners=False)]
        poses_list = [batch["rel_pose_stereo"].to(self.device)]

        for i in range(-self.prev_frames, self.next_frames + 1):
            if i == 0:
                continue

            sources_list.append(F.interpolate(
                batch["nearby_frames"][i]["camera_data"]["stereo_left_image"].to(self.device).permute(0, 3, 1,
                                                                                                      2).float(),
                (self.height, self.width), mode="bilinear", align_corners=False))
            poses_list.append(
                torch.matmul(torch.inverse(tgt_poses), batch["nearby_frames"][i]["pose"].to(self.device)))

        # Stacking sources and poses
        sources = torch.stack(sources_list, dim=0)
        poses = torch.stack(poses_list, dim=0)

        # Intrinsics
        tgt_intrinsics = batch["intrinsics"]["stereo_left"].to(self.device)
        src_intrinsics_stereo = batch["intrinsics"]["stereo_right"].to(self.device)

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

        # Backpropogates if train is set to True
        if train:
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

        # Adjusts image number    
        self.img_num += self.batch_size

        return losses

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
                to_save['height'] = self.height
                to_save['width'] = self.width
            torch.save(to_save, save_path)
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def add_disparity_map_to_tensorboard(self, disp, img_num, dataset_length, train):
        """
        Adds output disparity map to tensorboard
        :param [tensor] disp: The disparity map outputted by the network
        :param [int] img_num: The index of the input image in the training/validation file
        :param [int] dataset_length: The length of the training/validation dataset
        :param [boolean] train: Differentiates between training and validation
        """
        # Processing disparity map
        disp_resized = torch.nn.functional.interpolate(
            disp.unsqueeze(0), (self.height, self.width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = transforms.ToTensor()(colormapped_im)

        # Decides whether to add disparity map to the training or validation set
        train_or_val = "Validation"
        if train:
            train_or_val = "Training"

        # Add image to tensorboard
        self.writer.add_image(train_or_val + " - " + f'Epoch: {self.epoch + 1}, ' + f'Image: {img_num}',
                              im,
                              self.epoch * dataset_length + img_num)


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
    test = Trainer("configs/basic_model.yml")
    test.train()
