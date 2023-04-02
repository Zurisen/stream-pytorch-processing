import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageColor
import numpy as np
import cv2
plt.rcParams["savefig.bbox"] = 'tight'


def detect_image(img, img_size=416):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0).to(device)
    #input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(image_tensor)
        #detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections
