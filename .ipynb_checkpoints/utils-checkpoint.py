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


def load_maskrcnn():
    # Define the COCO dataset and data loader
    coco_dataset = torchvision.datasets.CocoDetection(root='etc/val2017/', annFile='etc/annotations/captions_val2017.json',
                                                      transform=transforms.ToTensor())
    coco_loader = DataLoader(coco_dataset, batch_size=1, shuffle=False)

    # Load the pre-trained Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Put the model in evaluation mode
    model.eval()
    return model

def feedforward(model, image):
    # Convert the frame to a NumPy array
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # change color space if necessary
    image = Image.fromarray(image)
    new_size = (int(image.size[0]/2), int(image.size[1]/2))
    image = image.resize(new_size)
    prediction = model([transforms.ToTensor()(image)])
    indices = torch.nonzero(prediction[0]['scores'] > 0.90, as_tuple=False).squeeze(1)
    
    
    
    # Convert the tensor to a numpy array
    image_np = torch.Tensor.cpu(transforms.ToTensor()(image)).numpy()
    # Convert the numpy array to uint8
    image_np_uint8 = (image_np * 255).astype('uint8')
    result = draw_segmentation_masks(torch.from_numpy(image_np_uint8),
                                     masks = prediction[0]['masks'][indices].squeeze(1) > 0.5,
                                     alpha=0.9)

    result = result.detach().cpu().numpy()
    result = np.rollaxis(result, 0, 3)
    #result = F.to_pil_image(result)

    return result
    
    
def PIL_to_cv2(image: np.array):    
    # Convert the PIL image to a NumPy array in BGR format
    output_np = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return output_np