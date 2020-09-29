import torch
import pytest

from utils import crop_batch


def test_crop_batch():
    test_batch = torch.arange(32).reshape(4, 8).repeat(5, 1, 1).unsqueeze(1)
    crops = torch.tensor([
        [0, 0, 4, 2],
        [1, 1, 4, 2],
        [3, 2, 4, 2],
        [0, 0, 4, 2],
        [0, 1, 4, 2],
    ])
    ans = torch.tensor([[[[0, 1, 2, 3],
                          [8, 9, 10, 11]]],
                        [[[9, 10, 11, 12],
                          [17, 18, 19, 20]]],
                        [[[19, 20, 21, 22],
                          [27, 28, 29, 30]]],
                        [[[0, 1, 2, 3],
                          [8, 9, 10, 11]]],
                        [[[8, 9, 10, 11],
                          [16, 17, 18, 19]]]])

    torch.testing.assert_allclose(crop_batch(test_batch, crops), ans)
