import data_set_utils as data_utils
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
import torch
import cv2
import image2segVec as image2Vec
import torch

import os
import torch
import sys

# Ensure you can import from the parent directory
sys.path.append("..")

# Get current directory until the SAT directory
cwd = os.getcwd()

sam_checkpoint = os.path.join(cwd, "segment-anything", "sam_vit_h_4b8939.pth")

model_type = "vit_h"

# Set CUDA_VISIBLE_DEVICES to specify GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Ensure CUDA is available and set the device to cuda:0 (which should map to the real GPU 2)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Print the current device being used
print(f"Using device: {device}")

# Load the model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)




#import convolve2d
from scipy.signal import convolve, convolve2d



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
    

def apply_convolution_with_scipy(image, mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)
    result = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = convolve2d(image[:, :, c] * mask, kernel, mode='same', boundary='fill', fillvalue=0)
    return result



def create_mini_imagess(image, masks, size=(600, 600),print_resualts = False):
    mini_images_clean = []
    mini_images_original = []   
    original_masks = masks
    for mask in masks:
        # find the center of the mask and devide bythe width of the image to get the position of the mask in the image
        center = mask['bbox'][0] + mask['bbox'][2] // 2, mask['bbox'][1] + mask['bbox'][3] // 2
        center = center[0] / image.shape[1], center[1] / image.shape[0]
        

        # Extract the bounding box and the mask
        bbox = mini_image_bbox_to_square(mask['bbox'])
        #check if the mask is in the image and a box
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
            continue
        if bbox[0] + bbox[2] > image.shape[1] or bbox[1] + bbox[3] > image.shape[0]:
            continue

        # check if the mask is a square if not  set the mask to be a the smallest square that contains the mask and the image
        if bbox[2] != bbox[3]:
            bbox = mini_image_bbox_to_square(bbox)
            
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
        #we have binary mask size of image 1 where is the segment ans 0 where not we want to make positional encoding by take kernel path_sizepatch_sizeimage[wher ethe colors] and multiple it like conv over the mask (first make it like image from binary ) so the return is image//number of patch like mini image
        """
        #
        # multiple the ccentroids of the mask by the size of the image to get the position of the mask in the image
       # center = center[0] * size[0], center[1] * size[1]+size[1]//2
        # clean_cropped_image =  clean_cropped_image + center[0]//size[0] add with clip

       # clean_cropped_image = np.clip(clean_cropped_image + center[0]//size[0],0,255)



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



# patch the image into 8x8 patches according the size of the image
def patches_image(image, patch_num=8):
    #fic the size of the image to be a multiple of patch_num
    height, width = image.shape[:2]
    height = height - height % patch_num
    width = width - width % patch_num
    image = image[:height, :width]
    #split the image to patch_num*patch_num patches
    patch_height = height // patch_num
    patch_width = width // patch_num
    patches = []
    for i in range(patch_num):
        for j in range(patch_num):
            patch = image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
            patches.append(patch)

    return patches

#create grid image of the mini images,actual image


def create_square_grid_image(mini_images,image,print_en=False):

    #double the mini images to be able to fill the grid, mini_images_Double = mini_images + mini_images
    mini_images_runner = []

    # Calculate the dimensions of the grid
    l = len(mini_images)
    n = 16 # The size of the grid (n x n)
    
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

    mini_image_runner += mini_images_clean + mini_images_clean  + mini_images_clean + mini_images_clean
    # Patch the original image 8 and 4
    image_patches = patches_image(image)   
    #resize to quares: image_patches to miniimsge size
    mini_image_size = mini_images_clean[0].shape[0]
    image_patches = [cv2.resize(patch, (mini_image_size, mini_image_size)) for patch in image_patches]
     
    mini_images_clean = mini_images_clean + image_patches  + mini_image_runner  

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

import matplotlib.pyplot as plt
# Create the grid image
#now we will do function that ger patch_num and image and create grid with the seg and the patches 
def final_pre_image(image, patch_num=16):
    len_of_mini_images = patch_num*patch_num
    #get masks from the image
    masks = image2Vec.image2segVec(image=image,size=(256,256))
    #get the mini images clean and original
    mini_images_clean, mini_images_original = create_mini_imagess(image, masks,size=(40,40))
    #create the patches
    patches4 = patches_image(image, patch_num=4)
    patches8 = patches_image(image, patch_num=8)
    patches = patches4 + patches8
    #add the patches to the mini images
    mini_images_clean.extend(patches)
    #add the mini images to the original mini images until 256 according the order
    while len(mini_images_clean) < patch_num*patch_num:
        mini_images_clean.extend(mini_images_clean)
        mini_images_original.extend(mini_images_original)

    #cut the mini images to be 256
    mini_images_clean = mini_images_clean[:patch_num*patch_num]

    #create the grid image 
    grid_image = create_square_grid_image(mini_images_clean,image,print_en=True)
    # resize into 512
    grid_image = cv2.resize(grid_image, (512, 512))
    return grid_image

image = cv2.imread("/home/benjaminc/project_transformers/mainSat/segment-anything/dataset_seg_ev/plage.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

grid_image = final_pre_image(image, patch_num=16)
from transformers.utils import send_example_telemetry

send_example_telemetry("semantic_segmentation_notebook", framework="pytorch")
from datasets import load_dataset

hf_dataset_identifier = "segments/sidewalk-semantic"
ds = load_dataset(hf_dataset_identifier)

  

import evaluate

metric = evaluate.load("mean_iou")
from huggingface_hub import hf_hub_download
import json

filename = "id2label.json"
id2label = json.load(
    open(hf_hub_download(hf_dataset_identifier, filename, repo_type="dataset"), "r")
)
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
num_labels, list(label2id.keys())
num_labels, list(label2id.keys())  


ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]
#example 
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

feature_extractor
import os
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, TrainingArguments, Trainer
import torch
from torch import nn
import evaluate
import json
from huggingface_hub import hf_hub_download
from torchvision.transforms import ColorJitter

# Function to apply segTokenaizer and revert format
def segTokenaizer(image, patch_num=16):
    len_of_mini_images = patch_num * patch_num
    masks = image2Vec.image2segVec(image=image, size=(256, 256))
    mini_images_clean, mini_images_original = create_mini_imagess(image, masks, size=(40, 40))
    patches4 = patches_image(image, patch_num=4)
    patches8 = patches_image(image, patch_num=8)
    patches = patches4 + patches8
    mini_images_clean.extend(patches)
    while len(mini_images_clean) < patch_num * patch_num:
        mini_images_clean.extend(mini_images_clean)
        mini_images_original.extend(mini_images_original)
    mini_images_clean = mini_images_clean[:patch_num * patch_num]
    grid_image = create_square_grid_image(mini_images_clean, image, print_en=True)
    grid_image = cv2.resize(grid_image, (512, 512))
    return grid_image

# Load and preprocess dataset
hf_dataset_identifier = "segments/sidewalk-semantic"
ds = load_dataset(hf_dataset_identifier)
ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

# Load id2label mapping
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

# Preprocess and save all images in a single directory
def preprocess_and_save(ds, split):
    dir_path = f"/home/benjaminc/project_transformers/mainSat/tokenizer_data/{split}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    for i in range(len(ds)):
        example = ds[i]
        image_path = os.path.join(dir_path, f"pixel_values_{i}.npy")

        if not os.path.exists(image_path):
            # Convert image to RGB format
            image_cv2 = cv2.cvtColor(np.array(example["pixel_values"]), cv2.COLOR_BGR2RGB)
            # Apply final_pre_image
            processed_image_pil = segTokenaizer(image_cv2)
            # Convert back to BGR format
            processed_image_cv2 = cv2.cvtColor(np.array(processed_image_pil), cv2.COLOR_RGB2BGR)
            # Save the processed image to disk
            np.save(image_path, processed_image_cv2)
        else:
            # Load the processed image from disk
            processed_image_cv2 = np.load(image_path)

        # Update the example with the processed image
        ds[i]["pixel_values"] = processed_image_cv2

        if i % 10 == 0:
            print(f"Processed {i} images for {split}")

preprocess_and_save(train_ds, "train")
preprocess_and_save(test_ds, "test")

# Ensure transformations are removed before saving the dataset
train_ds.set_transform(None)
test_ds.set_transform(None)

# Save the dataset
train_ds.save_to_disk("/home/benjaminc/project_transformers/mainSat/train_dataset")
test_ds.save_to_disk("/home/benjaminc/project_transformers/mainSat/test_dataset")

# Set up the processor and transformations
processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.002)  # Reduced hue to avoid overflow

def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

# Initialize the model
pretrained_model_name = "nvidia/mit-b0"
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

# Training arguments
epochs = 50
lr = 0.00006
batch_size = 2

hub_model_id = "segformer-b0-finetuned-segments-sidewalk-oct-22"

training_args = TrainingArguments(
    "segformer-b0-finetuned-segments-sidewalk-outputs",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_model_id=hub_model_id,
    hub_strategy="end",
)

# Evaluation metric
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # Scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=processor.do_reduce_labels,
        )

        # Add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()


    

    