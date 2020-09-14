import argparse
import csv
from datetime import datetime

import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import shutil
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import yaml
from tensorflow.image import decode_jpeg

from collate import Collator
from kitti_dataset import KittiDataset
from loss import calc_loss, GenerateReprojections
from monodepth_metrics import run_metrics
from third_party.monodepth2.ResnetEncoder import ResnetEncoder
from third_party.monodepth2.DepthDecoder import DepthDecoder
from fpn import FPN
from third_party.monodepth2.PoseDecoder import PoseDecoder
from third_party.monodepth2.layers import transformation_from_parameters, disp_to_depth

LOSS_VIS_SIZE = (10, 4)
LOSS_VIS_CMAP = "cividis"


class Trainer:
    def __init__(self, config_path, start_epoch):
        """
        Creates an instance of tranier using a config file
        The config file contains all the information needed to train a model
        :param [str] config_path: The path to the config file
        :return [Trainer]: Object instance of the trainer
        """
        
        # Epoch to continue training from (0 if new model)
        self.start_epoch = start_epoch
        
        # Load data from config
        with open(config_path) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)
            
        if self.start_epoch > 0:
            self.log_dir = os.path.dirname(config_path)
        else:
            date_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            self.log_dir = self.config["log_dir"] + "_" + date_time
            os.mkdir(self.log_dir)
            shutil.copyfileobj(open(config_path, "r"), open(os.path.join(self.log_dir, "config.yml"),"w+"))

        # GPU/CPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        # Epoch and batch info
        if self.start_epoch > 0:
            self.num_epochs = self.config["num_epochs"]
        else:
            self.num_epochs = self.start_epoch + self.config["num_epochs"]
        self.batch_size = self.config["batch_size"]

        # Image dimensions
        self.width = self.config["width"]
        self.height = self.config["height"]

        # Dataloader Setup
        self.collate = Collator(self.height, self.width)
        self.num_workers = self.config["num_workers"]

        train_config_path = self.config["train_config_path"]
        self.train_dataset = KittiDataset.init_from_config(train_config_path)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           collate_fn=self.collate, num_workers=self.num_workers)

        val_config_path = self.config["valid_config_path"]
        self.val_dataset = KittiDataset.init_from_config(val_config_path)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                         collate_fn=self.collate, num_workers=self.num_workers)

        # Neighboring frames
        self.prev_frames = self.train_dataset.previous_frames
        self.next_frames = self.train_dataset.next_frames

        # Stereo
        self.use_stereo = self.config["use_stereo"]

        # Number of scales
        self.num_scales = self.config["num_scales"]

        # Model setup
        self.models = {}
        self.pretrained = self.config["pretrained"]

        self.models['resnet_encoder'] = ResnetEncoder(self.config["encoder_layers"], pretrained=self.pretrained).to(
            self.device)
        self.num_scales = self.config["num_scales"]
        decoder_num_ch = self.models["resnet_encoder"].num_ch_enc

        if self.config.get("use_fpn"):
            self.models["fpn"] = FPN(decoder_num_ch).to(self.device)
            decoder_num_ch = self.models["fpn"].num_ch_pyramid

        self.models['depth_decoder'] = DepthDecoder(num_ch_enc=decoder_num_ch,
                                                    scales=range(self.num_scales)).to(self.device)
        self.models["pose_encoder"] = ResnetEncoder(
            self.config["encoder_layers"],
            pretrained=self.pretrained,
            num_input_images=2
        ).to(self.device)

        self.models["pose_decoder"] = PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2
        ).to(self.device)
        
        # Loading pretrained weights
        if self.start_epoch > 0:
            weights_folder = os.path.join(self.log_dir, "models", f'weights_{self.start_epoch-1}')
            for model_name in self.models:
                model_path = os.path.join(weights_folder, f"{model_name}.pth")

                # if model_name == "resnet_encoder" or model_name == "depth_encoder":
                model_dict = self.models[model_name].state_dict()
                preset_dict = torch.load(model_path)
                model_dict.update({k: v for k, v in preset_dict.items() if k in model_dict})
                self.models[model_name].load_state_dict(model_dict)

        # Parameters
        parameters_to_train = []
        for model_name in self.models:
            parameters_to_train += list(self.models[model_name].parameters())

        # Optimizer
        if self.start_epoch > 0:
            optimizer_load_path = os.path.join(weights_folder, "adam.pth")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer = optim.Adam(parameters_to_train)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            learning_rate = self.config["learning_rate"]
            self.optimizer = optim.Adam(parameters_to_train, learning_rate)

        # Learning rate scheduler
        scheduler_step_size = self.config["scheduler_step_size"]
        weight_decay = self.config["weight_decay"]
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, scheduler_step_size, weight_decay)

        # Setting up reprojection
        self.gen_reproj = []
        for i in range(self.num_scales):
            h = self.height // (2 ** i)
            w = self.width // (2 ** i)
            self.gen_reproj.append(GenerateReprojections(h, w, self.batch_size).to(self.device))

        # Writer for tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))

        # Step size for tensorboard
        self.tensorboard_step = self.config["tensorboard_step"]

        # Utility variables
        self.steps_until_write = 0

        # Set up for metrics
        self.metrics = self.config["metrics"]
        if self.metrics:
            if self.start_epoch > 0:
                self.metrics_file = open(os.path.join(self.log_dir, "metrics.csv"),"a", newline='')
                self.metrics_writer = csv.writer(self.metrics_file, delimiter=',')
            else:
                self.metrics_file = open(os.path.join(self.log_dir, "metrics.csv"),"w", newline='')
                self.metrics_writer = csv.writer(self.metrics_file, delimiter=',')
                metrics_list = ["epoch", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
                self.metrics_writer.writerow(metrics_list)
       
        # Depth boundaries
        self.min_depth = self.config["min_depth"]
        self.max_depth = self.config["max_depth"]

    def train(self):
        """
        Runs the entire training pipeline
        Saves the model's weights at the end of training
        """
        for self.epoch in range(self.start_epoch, self.num_epochs):
            self.run_epoch()
            self.save_model()
            if self.metrics:
                metrics, metric_labels = run_metrics(self.log_dir, self.epoch+1)
                self.add_metrics_to_tensorboard(metrics, metric_labels)
                metrics = [round(num, 3) for num in metrics]
                metrics.insert(0, self.epoch+1)
                self.metrics_writer.writerow(metrics)
        self.writer.close()
        self.metrics_file.close()
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
        self.models['pose_encoder'].train()
        self.models['pose_decoder'].train()

        self.steps_until_write = total_loss = count = 0
        for batch_idx, batch in enumerate(self.train_dataloader):
            count += 1
            total_loss += self.process_batch(batch_idx, batch, len(self.train_dataset), "Training", True).item()
        total_loss /= count

        self.lr_scheduler.step()

        train_end_time = time.time()

        print(f"Training Loss: {total_loss}")
        print(f"Time spent: {train_end_time - train_start_time}")

        # Validation
        val_start_time = time.time()

        print(f"Validating epoch {self.epoch + 1}", end=", ")

        for model_name in self.models:
            self.models[model_name].eval()

        self.steps_until_write = total_loss = count = 0
        for batch_idx, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                count += 1
                total_loss += self.process_batch(batch_idx, batch, len(self.val_dataset), "Validation", False).item()
        total_loss /= count

        val_end_time = time.time()

        print(f"Validation Loss: {total_loss}")
        print(f"Time spent: {val_end_time - val_start_time}\n")

    def process_batch(self, batch_idx, batch, dataset_length, name, backprop):
        """
        Computes loss for a single batch
        :param [int] batch_idx: The batch index
        :param [dict] batch: The batch data
        :param [int] dataset_length: The length of the training/validation dataset
        :param [String] name: Differentiates between training/validation
        :param [boolean] backprop: Determines whether or not to backpropogate loss
        :return [tensor] losses: A 0-dimensional tensor containing the loss of the batch
        """
        # Predict disparity map
        inputs = batch["stereo_left_image"].to(self.device).float()
        local_batch_size = len(inputs)
        features = self.models['resnet_encoder'](inputs)
        if self.config.get("use_fpn"):
            pyramid = self.models["fpn"](features)
            outputs = self.models["depth_decoder"](pyramid)
        else:
            outputs = self.models['depth_decoder'](features)

        # Loading source images and pose data
        sources_list = []
        poses_list = []
        if self.use_stereo:
            sources_list.append(batch["stereo_right_image"].float().to(self.device))
            poses_list.append(batch["rel_pose_stereo"].to(self.device))

        for i in range(-self.prev_frames, self.next_frames + 1):
            if i == 0:
                continue
            sources_list.append(batch["nearby_frames"][i]["camera_data"]["stereo_left_image"].float().to(self.device))

            if i < 0:
                pose_inputs = [
                    batch["nearby_frames"][i]["camera_data"]["stereo_left_image"].float().to(self.device),
                    inputs
                ]
            else:
                pose_inputs = [
                    inputs,
                    batch["nearby_frames"][i]["camera_data"]["stereo_left_image"].float().to(self.device)
                ]
            pose_features = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
            axisangle, translation = self.models["pose_decoder"](pose_features)
            poses_list.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(i < 0)))

        # Stacking source images and pose data
        sources = torch.stack(sources_list, dim=0)
        poses = torch.stack(poses_list, dim=0)

        # Loading intrinsics
        tgt_intrinsics = batch["intrinsics"]["stereo_left"].to(self.device)
        if self.use_stereo:
            src_intrinsics_stereo = batch["intrinsics"]["stereo_right"].to(self.device)
        shapes = batch["shapes"].to(self.device).float()

        losses = []
        automasks = []
        min_losses = []
        reprojections = []
        total_loss = 0

        for scale in range(self.num_scales):
            h = self.height // (2 ** scale)
            w = self.width // (2 ** scale)

            # Convert disparity to depth
            disp = outputs[("disp", scale)]

            _, depths = disp_to_depth(disp, self.min_depth, self.max_depth)

            # Input scaling
            inputs_scale = F.interpolate(inputs, [h, w], mode="bilinear", align_corners=False).to(self.device)

            # Sources and pose scaling
            sources_scale = []
            for image in sources:
                sources_scale.append(F.interpolate(image, [h, w], mode="bilinear", align_corners=False).to(self.device))
            sources_scale = torch.stack(sources_scale, dim=0)

            # Intrinsics and scaling
            out_shape = torch.tensor([h, w]).to(self.device)
            shapes_scale = out_shape / shapes
            tgt_intrinsics_scale = torch.clone(tgt_intrinsics)
            tgt_intrinsics_scale[:, 0] = tgt_intrinsics_scale[:, 0] * shapes_scale[:, 1].reshape(-1, 1)
            tgt_intrinsics_scale[:, 1] = tgt_intrinsics_scale[:, 1] * shapes_scale[:, 0].reshape(-1, 1)

            if self.use_stereo:
                src_intrinsics_stereo_scale = torch.clone(src_intrinsics_stereo)

                src_intrinsics_stereo_scale[:, 0] = src_intrinsics_stereo_scale[:, 0] * \
                                                    shapes_scale[:, 1].reshape(-1, 1)
                src_intrinsics_stereo_scale[:, 1] = src_intrinsics_stereo_scale[:, 1] * \
                                                    shapes_scale[:, 0].reshape(-1, 1)
                intrinsics_list = [src_intrinsics_stereo_scale]
            else:
                intrinsics_list = [tgt_intrinsics_scale]

            for i in range(len(poses_list) - 1):
                intrinsics_list.append(tgt_intrinsics_scale)

            src_intrinsics_scale = torch.stack(intrinsics_list)

            # Reprojection
            reprojected = self.gen_reproj[scale](sources_scale, depths, poses, tgt_intrinsics_scale,
                                                 src_intrinsics_scale, local_batch_size)
            reprojected = torch.stack(reprojected, dim=0)

            # Compute Losses            
            loss_inputs = {"targets": inputs_scale,
                           "sources": sources_scale}
            loss_outputs = {"reproj": reprojected,
                            "disparities": disp}

            loss, automask, min_loss = calc_loss(loss_inputs, loss_outputs, scale)

            losses.append(loss)
            automasks.append(automask)
            min_losses.append(min_loss)
            reprojections.append(reprojected)

            total_loss += losses[scale]

        total_loss /= self.num_scales

        # Backpropagation
        if backprop:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Add image, disparity map, and loss to tensorboard
        curr_idx = 0
        while curr_idx < local_batch_size:
            curr_idx += self.steps_until_write
            if curr_idx < local_batch_size:
                self.add_img_disparity_loss_to_tensorboard(
                    outputs[("disp", 0)][curr_idx], inputs[curr_idx], automasks[0][curr_idx].unsqueeze(0),
                    min_losses[0][curr_idx], reprojections[0][:, curr_idx],
                    self.batch_size * batch_idx + curr_idx + 1, name
                )
                self.writer.add_scalar(
                    name + " Loss", total_loss.item(),
                    self.epoch * dataset_length + self.batch_size * batch_idx + curr_idx + 1
                )

                self.steps_until_write = self.tensorboard_step
            else:
                self.steps_until_write -= local_batch_size - curr_idx + self.steps_until_write
        return total_loss

    def save_model(self):
        """
        Saves model weights to disk (from monodepth2 repo)
        """
        save_folder = os.path.join(self.log_dir, "models", "weights_{}".format(self.epoch))
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
        
    def add_img_disparity_loss_to_tensorboard(self, disp, img, automask, loss, reproj, img_num, name):
        """
        Adds image disparity map, and automask to tensorboard
        :param [tensor] disp: Disparity map outputted by the network
        :param [tensor] img: Original image
        :param [tensor] automask: Automask
        :param [torch.Tensor] loss: Minimum photometric error as calculated in loss functions
        :param [int] img_num: The index of the input image in the training/validation file
        :param [String] name: Differentiates between training/validation/evaluation
        """
        # Processing disparity map
        disp_np = disp.squeeze().cpu().detach().numpy()
        vmax = np.percentile(disp_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_disp = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
        final_disp = transforms.ToTensor()(colormapped_disp)

        # Processing image
        img_np = img.squeeze().cpu().detach().numpy()
        colormapped_img = img_np.astype(np.uint8).transpose(1, 2, 0)
        final_img = transforms.ToTensor()(colormapped_img)

        loss_mean = loss.mean()
        figure = plt.figure(figsize=LOSS_VIS_SIZE)
        plt.imshow(loss.cpu(), cmap=LOSS_VIS_CMAP)
        plt.ylabel(f"Estim. Loss: {loss_mean:.3f}")
        plt.colorbar(orientation="horizontal")
        buf = io.BytesIO()
        plt.savefig(buf, format="jpg")
        plt.close(figure)
        buf.seek(0)
        loss = torch.from_numpy(decode_jpeg(buf.getvalue()).numpy())
        loss = loss.permute(2, 0, 1)

        reproj /= 255

        # Add image and disparity map to tensorboard
        self.writer.add_image(f"{name} Images/Epoch: {self.epoch + 1}",
                              final_img,
                              img_num)
        self.writer.add_image(f"{name} Disparity Maps/Epoch: {self.epoch + 1}",
                              final_disp,
                              img_num)
        self.writer.add_image(f"{name} Automasks/Epoch: {self.epoch + 1}",
                              automask,
                              img_num)
        self.writer.add_image(f"{name} Losses/Epoch: {self.epoch + 1}",
                              loss,
                              img_num)
        if self.use_stereo:
            self.writer.add_image(f"{name} Stereo Reprojection/Epoch: {self.epoch + 1}",
                                  reproj[0], img_num)
            self.writer.add_image(f"{name} Backward Reprojection/Epoch: {self.epoch + 1}",
                                  reproj[1], img_num)
            self.writer.add_image(f"{name} Forward Reprojection/Epoch: {self.epoch + 1}",
                                  reproj[2], img_num)
        else:
            self.writer.add_image(f"{name} Backward Reprojection/Epoch: {self.epoch + 1}",
                                  reproj[0], img_num)
            self.writer.add_image(f"{name} Forward Reprojection/Epoch: {self.epoch + 1}",
                                  reproj[1], img_num)

    def add_metrics_to_tensorboard(self, metrics, labels):
        """
        Adds metrics to tensorboard with given metric values and their corresponding values
        :param [list] metrics: A list of floats representing each metric
        :param [list] labels: A list of strings (same length as metrics) that describe the title of the metric
        """
        for i in range(len(metrics)):
            self.writer.add_scalar("metrics/" + labels[i], metrics[i], self.epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ddsr options")
    
    parser.add_argument("--config_path",
                        type=str,
                        help="path to the config",
                        default="configs/basic_model.yml")
    
    parser.add_argument("--epoch",
                        type=int,
                        help="epoch to continue training from",
                        default=0)
    
    opt = parser.parse_args()
    test = Trainer(opt.config_path, opt.epoch)
    test.train()
