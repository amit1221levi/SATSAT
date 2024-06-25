import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

#==================================================================================================
#                                  Image to set of mini images
#==================================================================================================
def image2segVec(image, 
                 sam_checkpoint="/home/benjaminc/project_transformers/mainSat/segment-anything/sam_vit_h_4b8939.pth",
                 device = "cpu",
                 sort_algorithm = "clustering",
                 size=(600, 600),
                 print_resualts = True):
    
    # Get masks from image
    masks = get_masks_from_image(image, sam_checkpoint=sam_checkpoint,device = device)

    # Sort masks
    masks = sort_algorithms(masks, image, sort_algorithm = sort_algorithm)

    # Create mini images
    return  masks

#--------------------------------------------------------------------------------------------------
def sort_algorithms(masks, image, sort_algorithm = "clustering"):
    # Sort masks
    masks_sorted = masks
    if sort_algorithm == "depth":
        masks_sorted = depth_sort_masks(image, masks)
    elif sort_algorithm == "centroid":
        masks_sorted = sort_by_centroid(masks)
    elif sort_algorithm == "size_and_position":
        masks_sorted = sort_by_size_and_position(masks)
    elif sort_algorithm == "clustering":
        masks_sorted = sort_by_clustering(masks)
    return masks_sorted

    

#==================================================================================================
#                                  Image Segment everything
#==================================================================================================

def get_masks_from_image(image, sam_checkpoint="/home/benjaminc/project_transformers/SAT/STViT-main/segment-anything/sam_vit_h_4b8939.pth",device = "cpu"):
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)
    return masks

#==================================================================================================
#                                  Masks Sort algorithms 
#==================================================================================================
def depth_sort_masks(image, masks):
    # Load a depth estimation model from PyTorch Hub
    # Replace 'repo_owner/repo_name' and 'model' with the actual repository and model name
    model = torch.hub.load('repo_owner/repo_name', 'model', pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define transformations (adjust based on the model's requirements)
    transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess the image
    img = image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = transform(img_rgb).unsqueeze(0).to(device)

    # Predict the depth map
    with torch.no_grad():
        depth_map = model(img_rgb)
    
    # Adjust the depth map post-processing based on the model's output format
    depth_map = depth_map.squeeze().cpu().numpy()  # Assuming the model outputs a single channel depth map

    # Compute average depth for each mask and sort
    sorted_masks = []
    for mask in masks:
        # Resize the mask to match the depth map size if necessary
        resized_mask = cv2.resize(mask['segmentation'].astype(np.float32), depth_map.shape[::-1])
        # Calculate average depth for the mask
        avg_depth = np.mean(depth_map[resized_mask > 0])
        sorted_masks.append((mask, avg_depth))

    # Sort the masks based on average depth
    sorted_masks.sort(key=lambda x: x[1])

    return [mask for mask, _ in sorted_masks]

#--------------------------------------------------------------------------------------------------
def centroid(bbox):
    """Calculate the centroid of a bounding box."""
    x_center = bbox[0] + bbox[2] / 2
    y_center = bbox[1] + bbox[3] / 2
    return (x_center, y_center)

#--------------------------------------------------------------------------------------------------
def sort_by_centroid(masks):
    """Sort masks by their centroids (left-to-right, then top-to-bottom)."""
    return sorted(masks, key=lambda x: (centroid(x['bbox'])[1], centroid(x['bbox'])[0]))

#--------------------------------------------------------------------------------------------------
def sort_by_size_and_position(masks):
    """Sort masks by area (larger to smaller), then top-to-bottom for each size category."""
    return sorted(masks, key=lambda x: (-x['area'], centroid(x['bbox'])[1]))

#--------------------------------------------------------------------------------------------------
def sort_by_clustering(masks, n_clusters=6):
    """Sort masks based on clustering their centroids."""
    if len(masks) <= n_clusters:
        return masks  # Return early if not enough masks for clustering
    
    # Extract centroids and perform clustering
    centroids = np.array([centroid(mask['bbox']) for mask in masks])
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(centroids)
    
    # Sort masks based on clusters, then within clusters based on area
    clustered_masks = sorted(zip(clusters, masks), key=lambda x: (x[0], -x[1]['area']))
    return [mask for _, mask in clustered_masks]

#--------------------------------------------------------------------------------------------------

def sort_by_clustering_extreme_points(masks, n_clusters=4, center_wight = 2):
    """Sort masks based on clustering their centroids and the 4 most extreme points."""
    if len(masks) <= n_clusters:
        return masks  # Return early if not enough masks for clustering
    
    # Extract centroids and perform clustering
    centroids = np.array([centroid(mask['bbox']) for mask in masks])
    extreme_points = np.array([[mask['bbox'][0], mask['bbox'][1], mask['bbox'][0] + mask['bbox'][2], mask['bbox'][1] + mask['bbox'][3]] for mask in masks])
    extreme_points = np.concatenate((extreme_points, centroids * center_wight), axis=1)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(extreme_points)
    
    # Sort masks based on clusters, then within clusters based on area
    clustered_masks = sorted(zip(clusters, masks), key=lambda x: (x[0], -x[1]['area']))
    
    return [mask for _, mask in clustered_masks]




#==================================================================================================
#                                     Create Mini Images
#==================================================================================================
# Expand the mask to be a square that his side is the longest side of the mask(the smallest square that contains the mask)
def mini_image_bbox_to_square(bbox):
    if bbox[2] > bbox[3]:
        bbox[1] -= (bbox[2] - bbox[3]) // 2
        bbox[3] = bbox[2]
    else:
        bbox[0] -= (bbox[3] - bbox[2]) // 2
        bbox[2] = bbox[3]
    return bbox


# Calculate the mean color within the mask and find the farthest color
def get_mean_and_farthest_color(image, mask):
    mean_color = np.mean(image[mask], axis=0)
    masked_pixels = image[mask]
    if masked_pixels.size > 0:
        mean_color = np.mean(masked_pixels, axis=0)
        distances = np.linalg.norm(masked_pixels - mean_color, axis=1)
        farthest_color = masked_pixels[np.argmax(distances)]
    else:
        farthest_color = [0, 0, 0]  # Default color if no masked pixels
    return mean_color, farthest_color


# Resize the clean cropped image to maintain aspect ratio
def resize_image(image, size):
    # image n*n*3 to size*size*3
    new_height, new_width = size
    height, width = image.shape[:2]
    if height > width:
        new_height = size[0]
        new_width = int(width * new_height / height)
    else:
        new_width = size[1]
        new_height = int(height * new_width / width)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image
    




def create_mini_images(image, masks, size=(600, 600),print_resualts = False):
    mini_images_clean = []
    mini_images_original = []   
    original_masks = masks
    for mask in masks:
        # Extract the bounding box and the mask
        bbox = mini_image_bbox_to_square(mask['bbox'])
        segmentation_mask = mask['segmentation']

        # Crop the image and mask to the square bounding box(the smallest square that contains the mask)
        cropped_image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        cropped_mask = segmentation_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

      
        # Calculate the mean color within the mask and find the farthest color
        #masked_pixels = cropped_image[cropped_mask]
        #mean_color, farthest_color = get_mean_and_farthest_color(cropped_image, cropped_mask)

        # outside of the mask *0.2 inside original color
        clean_cropped_image = np.zeros_like(cropped_image)
        clean_cropped_image[cropped_mask] = cropped_image[cropped_mask]
        clean_cropped_image[~cropped_mask] = cropped_image[~cropped_mask] * 0.5

        if clean_cropped_image.size == 0 or clean_cropped_image.shape[0] == 0 or clean_cropped_image.shape[1] == 0:
            #print("Invalid image dimensions for resizing. Skipping this mask.")
            continue  # Skip this mask

        if cropped_image.size == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            #print("Invalid image dimensions for resizing. Skipping this mask.")
            continue

        # Resize the clean cropped image to maintain aspect ratio
        clean_cropped_image = resize_image(clean_cropped_image, size)
        cropped_image = resize_image(cropped_image, size)



        # Place the resized image in the center of the canvas
   
        """
        # get the mask segmentation on the object in the  cropped_image
        mini_mask = np.zeros((size[0], size[1]), dtype=np.uint8)
        mini_mask[start_y:start_y + new_height, start_x:start_x + new_width] = cropped_mask
        """

        mini_images_original.append(cropped_image)

        mini_images_clean.append(clean_cropped_image)

    if print_resualts:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].set_title('Original Image')
        ax[1].imshow(image)
        if original_masks is not None:
            show_anns(original_masks)
        ax[1].axis('off')
        ax[1].set_title('Image with Masks')
        plt.show()
        
    return mini_images_clean, mini_images_original


#--------------------------------------------------------------------------------------------------
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


#--------------------------------------------------------------------------------------------------
def print_grid(images, n_cols=18):
    n_rows = len(images) // n_cols + (len(images) % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5, n_rows * (5 / n_cols)))
    for i, ax in enumerate(axs.flatten()):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


#--------------------------------------------------------------------------------------------------
import numpy as np
from math import ceil, sqrt

def create_square_grid_image(mini_images,print_en=False):
    # Calculate the dimensions of the grid
    l = len(mini_images)
    n = ceil(sqrt(l))  # The size of the grid (n x n)
    
    if l == 0:
        return None  # Return None if the list of images is empty

    # Clean the mini images that have the wrong shape
    mini_images_clean = [mini_images[i] for i in range(l) if mini_images[i].shape[0] == mini_images[i].shape[1]]
    mini_images = mini_images_clean

    channels = mini_images_clean[0].shape[2] if len(mini_images_clean[0].shape) == 3 else 1
    mini_image_size = mini_images_clean[0].shape[0]

    # Create an empty image for the final image the image is n*mini_image_size x n*mini_image_size
    img = np.zeros((n * mini_image_size, n * mini_image_size, channels), dtype=np.uint8)

    if print_en:
        # Print the grid
        print(f"Grid size: {n}x{n}")
        print(f"Mini image size: {mini_image_size}x{mini_image_size}")
        print(f"Final image size: {n * mini_image_size}x{n * mini_image_size}")

    # Fill the grid
    # Make  set of mini images twice
    mini_image_runner = []
    for i in range(2):
        mini_image_runner += mini_images_clean
    mini_images_clean = mini_image_runner

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # Check if the mini image have the same shape as the mini_image_size
            if mini_images_clean[idx].shape[0] != mini_image_size or mini_images_clean[idx].shape[1] != mini_image_size:
                print(f"Mini image {idx} has the wrong shape")
            else:
                img[i * mini_image_size:(i + 1) * mini_image_size, j * mini_image_size:(j + 1) * mini_image_size] = mini_images_clean[idx]

    print("The Length of the mini images: ", len(mini_images_clean))
    return img


    


def img2gridImage(img,original_img=True):
    masks = image2segVec(image=img,size=(256,256))
    mini_images_clean, mini_images_original= create_mini_images(img, masks,size=(40,40))
    #img = create_square_grid_image(mini_images_clean,print_en=True)
    img_in = create_square_grid_image(mini_images_original)
    if original_img:
        return img_in
    else:
     return img
    



