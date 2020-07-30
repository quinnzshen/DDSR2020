import torch
import torch.nn.functional as F
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class TrainerCollator(object):
    def __init__(self, height, width):
        """
        Creates an instance of TrainerCollator
        :param [int] height: image height used in training
        :param [int] width: image width used in training
        """
        self.height = height
        self.width = width

    def __call__(self, batch):
        """
        Puts each data field into a tensor with outer dimension batch size
        :param [list] batch: List of information for each batch
        """
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return self([torch.tensor(b, dtype=torch.float32) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float32)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch, dtype=torch.float32)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            if 'lidar_point_coord_velodyne' in elem.keys():
                rem_list = ['lidar_point_coord_velodyne', 'lidar_point_reflectivity']
                for d in batch:
                    [d.pop(key) for key in rem_list]
            if 'stereo_right_image' in elem.keys():
                resize_list = ['stereo_right_image', 'stereo_left_image']
                # for key in resize_list:
                #     for d in batch:
                #         d[key] = F.interpolate(
                #             (d[key].permute(2, 0, 1).float().unsqueeze(0)),
                #             [self.height, self.width], mode="bilinear", align_corners=False).squeeze(0)
            return {key: self([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [self(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))
