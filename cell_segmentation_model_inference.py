import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
from PIL import ImageFilter
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np

import os

# Global Constants
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device: ", DEVICE)

RESNET_MEAN = (0.485, 0.456, 0.406)  # change to mean of training images
RESNET_STD = (0.229, 0.224, 0.225)  # change to std dev of training images
NORMALIZE = True
BOX_DETECTIONS_PER_IMG = 540
cell_type_dict = {"astro": 1, "cort": 2, "shsy5y": 3}
mask_threshold_dict = {1: 0.55, 2: 0.75, 3: 0.6}
min_score_dict = {1: 0.55, 2: 0.75, 3: 0.5}


# Function to get model
def get_model(num_classes, model_chkpt=None):
    if NORMALIZE:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   box_detections_per_img=BOX_DETECTIONS_PER_IMG,
                                                                   image_mean=RESNET_MEAN,
                                                                   image_std=RESNET_STD)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                   box_detections_per_img=BOX_DETECTIONS_PER_IMG)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes + 1)

    if model_chkpt:
        model.load_state_dict(torch.load(model_chkpt, map_location=DEVICE))
    return model


# Function to convert image to pytorch tensor
class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


# Function to convert normalize image
class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, list(RESNET_MEAN), list(RESNET_STD))
        return image, target


# Function to combine multiple transforms
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# Function to get transformation of image
def get_transform():
    # transforms = [EnhanceEdges()]
    transforms = [ToTensor()]
    # transforms.append(ToTensor())
    if NORMALIZE:
        transforms.append(Normalize())

    return Compose(transforms)


# Function to remove overlapping masks
def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask


# Function to mask image with cell segments
def mask_image_cell_segments(image_path, model_chkpt_path):
    model = get_model(len(cell_type_dict), model_chkpt='MaskRCNN_FineTuned/pytorch_model-e26.bin')
    model.eval()

    transforms = get_transform()

    image_orig = Image.open(image_path).convert("RGB")
    image_transformed, _ = transforms(image=image_orig, target=None)

    with torch.no_grad():
        result = model([image_transformed.to(DEVICE)])[0]

    previous_masks = []
    for i, mask in enumerate(result["masks"]):

        # Filter-out low-scoring results. Not tried yet.
        score = result["scores"][i].cpu().item()
        label = result["labels"][i].cpu().item()
        if score > min_score_dict[label]:
            mask = mask.cpu().numpy()
            # Keep only highly likely pixels
            binary_mask = mask > mask_threshold_dict[label]
            binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
            previous_masks.append(binary_mask)

    image_orig_masked = np.array(image_orig)
    for binary_mask in previous_masks:
        binary_mask = np.reshape(binary_mask, (binary_mask.shape[1], binary_mask.shape[2]))

        image_orig_masked[binary_mask != 0] = (255, 255, 0)
    image_orig_masked = Image.fromarray(image_orig_masked)
    return image_orig, image_orig_masked


# os.system(r'mkdir -p  C:\Users\gj7qc9\.cache\torch\hub\checkpoints')
# os.system(r'copy MaskRCNN_PreTrained\maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth C:\Users\gj7qc9\.cache\torch\hub\checkpoints\maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth')


if __name__ == '__main__':
    image_orig, image_orig_masked = mask_image_cell_segments(image_path='staticFiles/uploads/7ae19de7bc2a.png',
                                                             model_chkpt_path='MaskRCNN_FineTuned/pytorch_model-e26.bin')
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].imshow(image_orig)
    ax[1].imshow(image_orig_masked)
    plt.show()

    # print(len(previous_masks), ' ', previous_masks[0].shape, ' ', len(result["masks"]))
