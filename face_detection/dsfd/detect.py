import torch
import numpy as np
import typing
from .face_ssd import SSD
from .config import resnet152_model_config
from .. import torch_utils
from torch.hub import load_state_dict_from_url
from ..base import Detector
from ..build import DETECTOR_REGISTRY

# model_url = "http://folk.ntnu.no/haakohu/WIDERFace_DSFD_RES152.pth"


@DETECTOR_REGISTRY.register_module
class DSFDDetector(Detector):

    def __init__(
            self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dict = torch.load('WIDERFace_DSFD_RES152.pth')
        self.net = SSD(resnet152_model_config)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net = self.net.to(self.device)

    @torch.no_grad()
    def _detect(self, x: torch.Tensor,) -> typing.List[np.ndarray]:
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        # Expects BGR
        x = x[:, [2, 1, 0], :, :]
        boxes = self.net(
            x, self.confidence_threshold, self.nms_iou_threshold
        )
        return boxes
