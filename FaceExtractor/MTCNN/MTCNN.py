import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torchvision.transforms.functional import to_pil_image

from .get_nets import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .first_stage import _generate_bboxes


class MTCNN(nn.Module):
    def __init__(self, min_face_size=20.0, thresholds=(0.6, 0.7, 0.8), nms_thresholds=(0.7, 0.7, 0.7)):
        super().__init__()
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.onet.eval()

    def run_first_stage(self, image_01, scale, threshold):
        width, height = image_01.shape[2:]
        sw, sh = math.ceil(width*scale), math.ceil(height*scale)
        image_01 = F.interpolate(image_01, (sw, sh), mode='bilinear')
        image = self.normalize(image_01)
        outputs = self.pnet(image)
        probs = outputs[1].data.numpy()[0, 1, :, :]
        offsets = outputs[0].data.numpy()

        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]

    def normalize(self, image_01):
        return ((image_01 * 255) - 127.5)*0.0078125

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, image_01):  # the input is actually a PIL image
        if image_01.ndim == 3:
            image_01 = image_01.unsqueeze(0)
        assert image_01.size(0) == 1, 'The forward function only accepts a batch size of 1'
        image_01 = image_01.to(self.device)

        width, height = image_01.shape[2:]
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size/self.min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = self.run_first_stage(image_01, scale=s, threshold=self.thresholds[0])
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], self.nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        image_pil = to_pil_image(image_01.cpu())

        img_boxes = get_image_boxes(bounding_boxes, image_pil, size=24)
        #img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        with torch.no_grad():
            img_boxes = torch.FloatTensor(img_boxes)
        output = self.rnet(img_boxes.to(self.device))
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, self.nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image_pil, size=48)
        if len(img_boxes) == 0:
            return [], []
        #img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        with torch.no_grad():
            img_boxes = torch.FloatTensor(img_boxes)
        output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, self.nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks




