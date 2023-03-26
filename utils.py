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
    coco_dataset = torchvision.datasets.CocoDetection(root='samples/val2017/', annFile='samples/annotations/captions_val2017.json',
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
        image = image.resize((720, 480))
        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy([transforms.ToTensor()(image)]).unsqueeze(0)
        prediction = model(image_tensor.float() / 255.0)
        
        # Convert the numpy array to uint8
        image_np = torch.Tensor.cpu(transforms.ToTensor()(image)).numpy()
        # Convert the numpy array to uint8
        image_np_uint8 = (image_np * 255).astype('uint8')
        result = draw_segmentation_masks(torch.from_numpy(image_np_uint8),
                                         masks = prediction[0]['masks'].squeeze(1) > 0.5,
                                         alpha=0.9)
        
        result = result.detach()
        result = F.to_pil_image(result)
        
        return result
    
    
def PIL_to_cv2(image):    
        # Convert the PIL image to a NumPy array in BGR format
        output_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return output_np
