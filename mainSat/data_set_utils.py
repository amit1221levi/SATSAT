# An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0 2 imaje then the masks of the image

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from pycocotools.mask import decode as rle_decode
from pycocotools import mask as mask_utils

#==============================================================================
#                          Load the dataset
#==============================================================================
def get_image_and_mask_paths_from_dir(dir_name):
    if not os.path.exists(dir_name):
        raise FileNotFoundError("Directory does not exist")
    
    # Get all files in the directory
    files = os.listdir(dir_name)

    # Pair each image file with its corresponding mask file
    image_mask_pairs = [(files[i], files[i+1]) for i in range(0, len(files), 2)]

    return image_mask_pairs


def read_masks_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    masks = []
    for ann in data["annotations"]:
        mask_data = {}
        mask_data["bbox"] = ann["bbox"]
        mask_data["area"] = ann["area"]
        mask_data["predicted_iou"] = ann.get("predicted_iou", None)
        mask_data["point_coords"] = ann.get("point_coords", None)
        mask_data["stability_score"] = ann.get("stability_score", None)
        mask_data["crop_box"] = ann.get("crop_box", None)

        # Decode RLE
        if isinstance(ann["segmentation"], dict):
            mask_data["segmentation"] = mask_utils.decode(ann["segmentation"])
        else:
            # If not in RLE, assuming it's already a binary mask or another format
            mask_data["segmentation"] = ann["segmentation"]

        masks.append(mask_data)

    return masks
    


def load_image_and_masks(image_path: str, mask_path: str, dir: str):
    """
    Load an image and its corresponding masks from specified paths and display the image with masks overlaid.

    Args:
    - image_path (str): Path to the image file.
    - mask_path (str): Path to the JSON file containing mask data.
    - dir (str): Directory where the image and masks are stored.

    Returns:
    - image (numpy.ndarray): The loaded image in RGB format.
    - masks (list[dict]): List of mask data dictionaries.
    """
    # Load the image jpg file
    image = cv2.imread(os.path.join(dir, image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {os.path.join(dir, image_path)}")

    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    path = os.path.join(dir, mask_path)

    # Load the masks data from the modified function appropriate for your data format
    masks = read_masks_from_json(path)

    return image, masks


#==============================================================================
#                           Visualizations
#==============================================================================

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



def show_image_and_mask(image, masks):
    # Display the image and the mask
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    ax[1].imshow(image)
    show_anns(masks)
    ax[1].axis('off')
    ax[1].set_title('Image with Masks')
    plt.show()







