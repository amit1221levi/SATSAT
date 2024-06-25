import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from segment_anything import sam_model_registry
import data_set_utils as data_utils
import image2segVec as image2Vec
from datasets import load_dataset
from transformers import AutoFeatureExtractor, SegformerImageProcessor, SegformerForSemanticSegmentation, TrainingArguments, Trainer
from huggingface_hub import hf_hub_download
import evaluate
from torchvision.transforms import ColorJitter
import json

# Set environment variables and paths
cwd = os.getcwd()
model_checkpoint = "nvidia/mit-b0"
batch_size = 32
sam_checkpoint = os.path.join(cwd, "mainSat/segment-anything/", "sam_vit_h_4b8939.pth")
model_type = "vit_h"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(sam_checkpoint)

# Load SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

# Function definitions
def mini_image_bbox_to_square(bbox):
    if bbox[2] > bbox[3]:
        bbox[1] -= (bbox[2] - bbox[3]) // 2
        bbox[3] = bbox[2]
    else:
        bbox[0] -= (bbox[3] - bbox[2]) // 2
        bbox[2] = bbox[3]
    return bbox

def resize_image(image, size):
    new_height, new_width = size
    height, width = image.shape[:2]
    if height > width:
        new_height = size[0]
        new_width = int(width * new_height / height)
    else:
        new_width = size[1]
        new_height = int(height * new_width / width)
    return cv2.resize(image, (new_width, new_height))

def apply_convolution_with_scipy(image, mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size ** 2)
    result = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = convolve2d(image[:, :, c] * mask, kernel, mode='same', boundary='fill', fillvalue=0)
    return result

def create_mini_images(image, masks, size=(600, 600), print_results=False):
    mini_images_clean, mini_images_original = [], []
    original_masks = masks
    for mask in masks:
        bbox = mini_image_bbox_to_square(mask['bbox'])
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
            continue
        if bbox[0] + bbox[2] > image.shape[1] or bbox[1] + bbox[3] > image.shape[0]:
            continue
        segmentation_mask = mask['segmentation']
        cropped_image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        cropped_mask = segmentation_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        clean_cropped_image = np.zeros_like(cropped_image)
        clean_cropped_image[cropped_mask] = cropped_image[cropped_mask]
        clean_cropped_image[~cropped_mask] = cropped_image[~cropped_mask] * 0.5
        if clean_cropped_image.size == 0 or cropped_image.size == 0:
            continue
        clean_cropped_image = resize_image(clean_cropped_image, size)
        cropped_image = resize_image(cropped_image, size)
        mini_images_original.append(cropped_image)
        mini_images_clean.append(clean_cropped_image)
    if print_results:
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

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def patches_image(image, patch_num=8):
    height, width = image.shape[:2]
    height = height - height % patch_num
    width = width - width % patch_num
    image = image[:height, :width]
    patch_height = height // patch_num
    patch_width = width // patch_num
    patches = [image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] for i in range(patch_num) for j in range(patch_num)]
    return patches

def create_square_grid_image(mini_images, image, print_en=False):
    l = len(mini_images)
    n = 16
    if l == 0:
        return None
    mini_images_clean = [mini_images[i] for i in range(l) if mini_images[i].shape[0] == mini_images[i].shape[1]]
    mini_images = mini_images_clean
    channels = mini_images_clean[0].shape[2] if len(mini_images_clean[0].shape) == 3 else 1
    mini_image_size = mini_images_clean[0].shape[0]
    img = np.zeros((n * mini_image_size, n * mini_image_size, channels), dtype=np.uint8)
    if print_en:
        print(f"Grid size: {n}x{n}")
        print(f"Mini image size: {mini_image_size}x{mini_image_size}")
        print(f"Final image size: {n * mini_image_size}x{n * mini_image_size}")
    mini_image_runner = []
    mini_image_runner += mini_images_clean + mini_images_clean + mini_images_clean + mini_images_clean
    image_patches = patches_image(image)
    image_patches = [cv2.resize(patch, (mini_image_size, mini_image_size)) for patch in image_patches]
    mini_images_clean = mini_images_clean + image_patches + mini_image_runner
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if mini_images_clean[idx].shape[0] != mini_image_size or mini_images_clean[idx].shape[1] != mini_image_size:
                print(f"Mini image {idx} has the wrong shape")
            else:
                img[i * mini_image_size:(i + 1) * mini_image_size, j * mini_image_size:(j + 1) * mini_image_size] = mini_images_clean[idx]
    print("The Length of the mini images: ", len(mini_images_clean))
    return img

def final_pre_image(image, patch_num=16):
    masks = image2Vec.image2segVec(image=image, size=(256,256))
    mini_images_clean, mini_images_original = create_mini_images(image, masks, size=(40, 40))
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

image = cv2.imread("/home/benjaminc/project_transformers/mainSat/segment-anything/dataset_seg_ev/plage.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
grid_image = final_pre_image(image, patch_num=16)

# Load and preprocess dataset
hf_dataset_identifier = "segments/sidewalk-semantic"
ds = load_dataset(hf_dataset_identifier)

metric = evaluate.load("mean_iou")

filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

augmentation = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
processor = SegformerImageProcessor(do_reduce_labels=True)
model = SegformerForSemanticSegmentation.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)

# Define transformation functions
def train_transforms(example_batch):
    images = [augmentation(image) for image in example_batch["image"]]
    encode_inputs = processor(images, seg_map=example_batch["seg_map"], random_padding=True, return_tensors="pt")
    return encode_inputs

def val_transforms(example_batch):
    encode_inputs = processor(image=example_batch["image"], seg_map=example_batch["seg_map"], return_tensors="pt")
    return encode_inputs

train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        predicted_class = torch.argmax(logits_tensor, dim=1)
        metrics = metric.compute(predictions=predicted_class, references=labels, num_labels=model.config.num_labels, ignore_index=processor.ignore_index, reduce_labels=processor.do_reduce_labels)
    return metrics

# Define training arguments and trainer
training_args = TrainingArguments(
    output_dir="segformer-b0-finetuned-segments-sidewalk",
    learning_rate=0.00006,
    num_train_epochs=50,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

train_results = trainer.train()
metrics = trainer.evaluate()

# Save the model and metrics
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_metrics("eval", metrics)
trainer.save_state()

print("Training completed and model saved.")
