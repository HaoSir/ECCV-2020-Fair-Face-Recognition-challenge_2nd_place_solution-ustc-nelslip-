import numpy as np
import torch
import typing
from abc import ABC, abstractmethod
# from torchvision.ops import nms
# from torch.ops.torchvision import nms
from .box_utils import scale_boxes


def check_image(im: np.ndarray):
    assert im.dtype == np.uint8,\
        f"Expect image to have dtype np.uint8. Was: {im.dtype}"
    assert len(im.shape) == 4,\
        f"Expected image to have 4 dimensions. got: {im.shape}"
    assert im.shape[-1] == 3,\
        f"Expected image to be RGB, got: {im.shape[-1]} color channels"
def nms(bboxes, scores, threshold=0.3):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor

class Detector(ABC):

    def __init__(
            self,
            confidence_threshold: float,
            nms_iou_threshold: float,
            device: torch.device,
            max_resolution: int):
        """
        Args:
            confidence_threshold (float): Threshold to filter out bounding boxes
            nms_iou_threshold (float): Intersection over union threshold for non-maxima threshold
            device ([type], optional): Defaults to cuda if cuda capable device is available.
            max_resolution (int, optional): Max image resolution to do inference to.
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.max_resolution = max_resolution
        self.mean = np.array(
            [123, 117, 104], dtype=np.float32).reshape(1, 1, 1, 3)

    def detect(
            self, image: np.ndarray, shrink=1.0) -> np.ndarray:
        """Takes an RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [height, width, 3]
        Returns:
            np.ndarray: shape [N, 5] with (xmin, ymin, xmax, ymax, score)
        """
        image = image[None]
        boxes = self.batched_detect(image, shrink)
        return boxes[0]

    @abstractmethod
    def _detect(self, image: torch.Tensor) -> torch.Tensor:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (torch.Tensor): shape [N, 3, height, width]
        Returns:
            torch.Tensor: of shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        raise NotImplementedError

    def filter_boxes(self, boxes: torch.Tensor) -> typing.List[np.ndarray]:
        """Performs NMS and score thresholding

        Args:
            boxes (torch.Tensor): shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        Returns:
            list: N np.ndarray of shape [B, 5]
        """
        final_output = []
        for i in range(len(boxes)):
            scores = boxes[i, :,  4]
            keep_idx = scores >= self.confidence_threshold
            boxes_ = boxes[i, keep_idx, :-1]
            scores = scores[keep_idx]
            if scores.dim() == 0:
                final_output.append(torch.empty(0, 5))
                continue
            keep_idx = nms(boxes_, scores, self.nms_iou_threshold)
            scores = scores[keep_idx].view(-1, 1)
            boxes_ = boxes_[keep_idx].view(-1, 4)
            output = torch.cat((boxes_, scores), dim=-1)
            final_output.append(output)
        return final_output

    def _pre_process(self, image: np.ndarray, shrink: float) -> torch.Tensor:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            torch.Tensor: shape [N, 3, height, width]
        """
        assert image.dtype == np.uint8
        height, width = image.shape[1:3]
        image = image.astype(np.float32) - self.mean
        image = np.moveaxis(image, -1, 1)
        image = torch.from_numpy(image)
        if self.max_resolution is not None:
            shrink_factor = self.max_resolution / max((height, width))
            if shrink_factor <= shrink:
                shrink = shrink_factor
        image = torch.nn.functional.interpolate(image, scale_factor=shrink)
        image = image.to(self.device)
        return image

    def _batched_detect(self, image: np.ndarray) -> typing.List[np.ndarray]:
        boxes = self._detect(image)
        boxes = self.filter_boxes(boxes)
        return boxes

    @torch.no_grad()
    def batched_detect(
            self, image: np.ndarray, shrink=1.0) -> typing.List[np.ndarray]:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            np.ndarray: a list with N set of bounding boxes of
                shape [B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        check_image(image)
        height, width = image.shape[1:3]
        image = self._pre_process(image, shrink)
        boxes = self._batched_detect(image)
        boxes = [scale_boxes((height, width), box).cpu().numpy() for box in boxes]
        self.validate_detections(boxes)
        return boxes

    def validate_detections(self, boxes: typing.List[np.ndarray]):
        for box in boxes:
            assert np.all(box[:, 4] <= 1) and np.all(box[:, 4] >= 0),\
                f"Confidence values not valid: {box}"
