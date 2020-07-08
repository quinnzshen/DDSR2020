from __future__ import absolute_import, division, print_function

import os
import glob
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('third_party/monodepth2')
from ResnetEncoder import ResnetEncoder
from DepthDecoder import DepthDecoder

from PoseCNN import PoseCNN
from PoseDecoder import PoseDecoder

import torch
from torchvision import transforms

from layers import disp_to_depth
from monodepth2_utils import download_model_if_doesnt_exist

"""
Adapted from Monodepth 2
This program loads existing weights from pretrained models and predicts depth for a given image for folder of images. 

The names of the pretrained models to choose from are listed below.

choices = [
    "mono_640x192",
    "stereo_640x192",
    "mono+stereo_640x192",
    "mono_no_pt_640x192",
    "stereo_no_pt_640x192",
    "mono+stereo_no_pt_640x192",
    "mono_1024x320",
    "stereo_1024x320",
    "mono+stereo_1024x320"
    ]
"""
def test_depth_model(image_path, model_name, **kwargs):
    """Function to predict for a single image or folder of images
    """
    #Specify an image file extension to search for (e.g png, jpeg, jpg)
    ext = kwargs.get('ext', None)
    
    #Setting no_cuda to False sets the device to cuda instead of cpu
    no_cuda = kwargs.get('no_cuda', True)
    
    #Specify an output path for the depth map (default is same path as input image)
    output_path = kwargs.get('output_path', None)
    
    #Set no_filsave to false if you do not want the output to be saved in a file 
    no_filesave = kwargs.get('no_filesave', False)
    
    #Set display_result to True to plot the resulting depth map
    display_result = kwargs.get('display_result', False)

    assert model_name is not None, \
        "You must specify the --model_name parameter"

    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(image_path):
        # Only testing on a single image
        paths = [image_path]
        if (output_path == None):
            output_directory = os.path.dirname(image_path)
        else:
            output_directory = output_path
    elif os.path.isdir(image_path):
        # Searching folder for images
        if (ext == None):
            paths = glob.glob(os.path.join(image_path, '*.*'))
        else:
            paths = glob.glob(os.path.join(image_path, '*.{}'.format(ext)))
        print(paths)
        if (output_path == None):
            output_directory = image_path
        else:
            output_directory = output_path
    else:
        raise Exception("Can not find image_path: {}".format(image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)


            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            
            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))

            if no_filesave == False:
                np.save(name_dest_npy, scaled_disp.cpu().numpy())
                im.save(name_dest_im)
                print("Processed {:d} of {:d} images - saved prediction to {}".format(idx + 1, len(paths), name_dest_im))

            if display_result == True:
                f, axarr = plt.subplots(2,1)
                axarr[0].imshow(mpimg.imread(image_path))
                axarr[1].imshow(colormapped_im)
    
    print('-> Done!')    

def test_pose_model(image_1_path, image_2_path, model_name, **kwargs):
    """Function to predict for a single image or folder of images
    """
    #Specify an image file extension to search for (e.g png, jpeg, jpg)
    ext = kwargs.get('ext', None)
    
    #Setting no_cuda to False sets the device to cuda instead of cpu
    no_cuda = kwargs.get('no_cuda', True)

    assert model_name is not None, \
        "You must specify the --model_name parameter"

    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = PoseCNN(2)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}

   # print(loaded_dict_enc)

    # extract the height and width of image that this model was trained with
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    pose_decoder = PoseDecoder(num_ch_enc=encoder.num_ch_enc, num_input_features=2)

    loaded_dict = torch.load(pose_decoder_path, map_location=device)
    pose_decoder.load_state_dict(loaded_dict)

    pose_decoder.to(device)
    pose_decoder.eval()

    # FINDING INPUT IMAGES
    if not os.path.isfile(image_1_path):
        raise Exception("Can not find image_path: {}".format(image_1_path))

    if not os.path.isfile(image_2_path):
        raise Exception("Can not find image_path: {}".format(image_2_path))
        
    print("-> Predicting on relative pose for test images")

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        # Load image and preprocess
        input_image1 = pil.open(image_1_path).convert('RGB')
        original_width, original_height = input_image1.size
        input_image1 = transforms.ToTensor()(input_image1).unsqueeze(0)

        input_image2 = pil.open(image_2_path).convert('RGB')
        original_width, original_height = input_image2.size
        input_image2 = transforms.ToTensor()(input_image2).unsqueeze(0)
        
        input_images = torch.cat((input_image1, input_image2), 2)
        
        # PREDICTION
        input_images = input_images.to(device)
        features = encoder(input_images)
        outputs = pose_decoder(features)

        disp = outputs[("disp", 0)]
        print(disp)
        """
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
        scaled_disp, _ = disp_to_depth(disp, 0.1, 100)


        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        
        name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))

        if no_filesave == False:
            np.save(name_dest_npy, scaled_disp.cpu().numpy())
            im.save(name_dest_im)
            print("Processed {:d} of {:d} images - saved prediction to {}".format(idx + 1, len(paths), name_dest_im))

        if display_result == True:
            f, axarr = plt.subplots(2,1)
            axarr[0].imshow(mpimg.imread(image_path))
            axarr[1].imshow(colormapped_im)
        """
    
    print('-> Done!')    
test_pose_model('data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_00/data/000000000.png','data/kitti_example/2011_09_26/2011_09_26_drive_0048_sync/image_00/data/000000001.png', 'mono_640x192')