import torch
import numpy as np


def crop_batch(batch, xywh):
    """
    Given a batch of images, crops each image differently based on the xywh array given, and reassembles the batch
    :param [torch.Tensor] batch: Tensor of shape [batch_size, channels, H, W] to be cropped
    :param [torch.Tensor] xywh: Tensor of shape [batch_size, 4], representing the leftmost and topmost indices of the
    crop, and specifying the width and height of the batch.
    :return [torch.Tensor]: Cropped batch of shape [batch_size, channels, new_H, new_W]
    """
    if torch.is_tensor(batch):
        out_batch = torch.empty_like(batch)
    else:
        out_batch = np.empty_like(batch)

    out_batch = out_batch[:, :, :xywh[0, 3], :xywh[0, 2]]
    endxy = xywh[:, :2] + xywh[:, 2:]
    for i in range(len(batch)):
        out_batch[i] = batch[i, :, xywh[i, 1]:endxy[i, 1], xywh[i, 0]:endxy[i, 0]]
    return out_batch
