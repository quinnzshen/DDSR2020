import torch

from math import pi


def convert_rgb(rgb_images: torch.Tensor, color: str = "RGB") -> torch.Tensor:
    """
    Performs all necessary calculations to change a batch of RGB images to a chosen color model
    :param rgb_images: Shape [batch_size, 3, H, W] of RGB images
    :param color: The color model to be transformed into (only RGB and HSV are implemented currently)
    :return: The converted batch of images into the given color model, with shape [batch_size, num_channels, H, W]
    """
    if color == "HSV":
        # Converts to HSV, then replaces the Hue channel with cos(Hue) and sin(Hue), so num_channels is increased by 1
        hsv_images = rgb_to_hsv(rgb_images)
        hsv_images[:, 0] *= pi / 180
        hsv_images = torch.cat((torch.cos(hsv_images[:, 0].unsqueeze(1)), hsv_images), dim=1)
        hsv_images[:, 1] = torch.sin(hsv_images[:, 0])
        return hsv_images

    return rgb_images


def rgb_to_hsv(rgb_images: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of RGB images into HSV
    :param rgb_images: Shape [batch_size, 3, H, W] of RGB images
    :return: The converted to HSV images in shape [batch_size, 3, H, W]
    """
    out = torch.empty_like(rgb_images)
    if len(rgb_images.shape) == 4:
        v = rgb_images.max(dim=1)
        mins = rgb_images.min(dim=1)
        diff = 60 / (v[0] - mins[0])

        out[:, 2] = v[0]

        out[:, 1] = 1 - mins[0] / v[0]
        out[:, 1][torch.isnan(out[:, 1])] = 0

        rgb_max_mask = (v[1] == 0, v[1] == 1, v[1] == 2)
        out[:, 0][rgb_max_mask[0]] = rgb_images[:, 1][rgb_max_mask[0]] - rgb_images[:, 2][rgb_max_mask[0]]
        out[:, 0][rgb_max_mask[1]] = rgb_images[:, 2][rgb_max_mask[1]] - rgb_images[:, 0][rgb_max_mask[1]]
        out[:, 0][rgb_max_mask[2]] = rgb_images[:, 0][rgb_max_mask[2]] - rgb_images[:, 1][rgb_max_mask[2]]
        out[:, 0] = out[:, 0] * diff + 120 * v[1]
        out[:, 0][torch.isnan(out[:, 0])] = 0
        out[:, 0][out[:, 0] < 0] += 360

    return out


def color_difference(batch1: torch.Tensor, batch2: torch.Tensor, color: str = "RGB") -> torch.Tensor:
    """
    Calculates L1 loss between two given batches of images in the respective color space given
    :param batch1: A batch of images, with shape [batch_size, num_channels, H, W]
    :param batch2: A batch of images to be compared to, with shape [batch_size, num_channels, H, W]
    :param color: The color model to be used in the comparison
    :return: Shape [batch_size, H, W], representing the L1 color difference at each pixel between the two input batches
    """
    if color == "HSV":
        return (
            torch.abs(batch1[:, 0] * batch1[:, 2] - batch2[:, 0] * batch2[:, 2]) +
            torch.abs(batch1[:, 1] * batch1[:, 2] - batch2[:, 1] * batch2[:, 2]) +
            torch.abs(batch1[:, 3] - batch2[:, 3])
        ) / 3

    return torch.mean(torch.abs(batch1 - batch2), 1)

