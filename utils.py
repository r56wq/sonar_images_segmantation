import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# For logging
import logging
import time

# Configurations for logging
train_log = "./logs/train"
test_log = "./logs/test"


class soft_dice_loss(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Args:
            smooth (float): A smoothing factor to avoid division by zero. Default: 1e-5.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, digits, y_true):
         """
        Args:
            digits (torch.Tensor): Raw logits of shape (N, C, H, W), it is before softmax
            y_true (torch.Tensor): Ground truth one-hot encoded tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The computed Soft Dice Loss (scalar).
        """
         y_pred = torch.softmax(digits, dim=1)
         intersection = torch.sum((y_pred*y_true), dim=(2, 3)) + self.smooth
         union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth
         dice = 2*intersection/union
         return 1 - dice[:, 1:].mean()
    

class hard_dice_score(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Args:
            smooth (float): A smoothing factor to avoid division by zero. Default: 1e-5.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, digits, y_true):
         """
        Args:
            digits (torch.Tensor): Raw logits of shape (N, C, H, W), it is not processed through argmax of softmax
            y_true (torch.Tensor): Ground truth one-hot encoded tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The computed Soft Dice Loss (scalar).
        """
         y_pred = digits.argmax(dim=1)
         # convert y_pred to one hot 
         y_pred_one_hot = F.one_hot(y_pred, num_classes=y_true.shape[1]).permute(0, 3, 1, 2)
         intersection = torch.sum((y_pred_one_hot*y_true), dim=(2, 3)) + self.smooth
         union = y_pred_one_hot.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth
         dice = 2*intersection/union
         return dice[:, 1:].mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanIoU(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, digits, y_true):
        y_pred = digits.argmax(dim=1)  # Shape: (N, H, W)
        y_pred_one_hot = F.one_hot(y_pred, num_classes=y_true.shape[1]).permute(0, 3, 1, 2)  # Shape: (N, C, H, W)

        intersection = torch.sum(y_pred_one_hot * y_true, dim=(2, 3)) + self.smooth  # Shape: (N, C)
        union = torch.sum(y_pred_one_hot, dim=(2, 3)) + torch.sum(y_true, dim=(2, 3)) - intersection + self.smooth  # Shape: (N, C)

        # Calculate IoU for each class
        iou = intersection / union  # Shape: (N, C)

        # Mask for valid classes (where union > 0)
        valid_classes = union > 0  # Shape: (N, C)

        # Compute mean IoU only over valid classes
        miou = iou[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)

        return miou


def get_unique(mask_dir):
    """
    Extracts all unique RGB values from mask images in a directory and saves them to colormap.txt.
    If colormap.txt already exists, then the function will return immediately 
    Args:
        mask_dir (str): Path to the directory containing mask images.
    
    Output:
        Writes unique RGB values to 'colormap.txt' in the format 'R, G, B' per line.
    """
    if (os.path.exists("./colormap.txt")):
        return

    # Set to store unique RGB values (using tuples for hashability)
    unique_colors = set()

    # Iterate over all files in the mask directory
    for filename in sorted(os.listdir(mask_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            # Open the image
            mask_path = os.path.join(mask_dir, filename)
            mask = Image.open(mask_path)

            # Convert to RGB if not already (in case of RGBA or other modes)
            if mask.mode != 'RGB':
                mask = mask.convert('RGB')

            # Convert to numpy array
            mask_array = np.array(mask)

            # Reshape to (num_pixels, 3) to get list of RGB values
            pixels = mask_array.reshape(-1, 3)

            # Add unique RGB tuples to the set
            for pixel in pixels:
                unique_colors.add(tuple(pixel))

    # Convert set to sorted list for consistent output
    unique_colors_list = sorted(list(unique_colors))

    # Write to colormap.txt
    with open('colormap.txt', 'w') as f:
        for r, g, b in unique_colors_list:
            f.write(f"{r}, {g}, {b}\n")

    print(f"Found {len(unique_colors_list)} unique RGB values. Saved to 'colormap.txt'.")



def tomasks(img: torch.Tensor, colormap: list) -> torch.Tensor:
    """
    Converts a tensor (C, H, W) to a mask (H, W) using a colormap.
    
    Args:
        img (torch.Tensor): Input tensor of shape (C, H, W), where C=1 (grayscale) or C=3 (RGB).
        colormap (list): List of RGB triplets, e.g., [(243, 231, 132), (211, 222, 111)].
    
    Returns:
        torch.Tensor: Mask of shape (H, W) with integer values corresponding to colormap indices.
    """
    # Ensure colormap is a tensor for efficient comparison
    colormap_tensor = torch.tensor(colormap, dtype=torch.uint8)  # Shape: (N, 3), N = num colors

    # Check input tensor shape
    if img.dim() != 3 or img.shape[0] not in [1, 3]:
        raise ValueError(f"Expected tensor of shape (C, H, W) with C=1 or 3, got {img.shape}")

    # Handle grayscale (C=1) or RGB (C=3)
    if img.shape[0] == 1:
        # Assume grayscale values are normalized [0, 1] or [0, 255]
        # Convert to RGB-like format by repeating the channel
        if img.max() <= 1.0:  # If normalized to [0, 1], scale to [0, 255]
            img_rgb = (img * 255).byte().expand(3, img.shape[1], img.shape[2])
        else:  # Assume [0, 255] range
            img_rgb = img.byte().expand(3, img.shape[1], img.shape[2])
    else:  # C=3, RGB case
        if img.max() <= 1.0:  # If normalized to [0, 1], scale to [0, 255]
            img_rgb = (img * 255).byte()
        else:  # Assume [0, 255] range
            img_rgb = img.byte()

    # Reshape img_rgb to (H*W, 3) for comparison
    H, W = img_rgb.shape[1], img_rgb.shape[2]
    img_flat = img_rgb.permute(1, 2, 0).reshape(-1, 3)  # Shape: (H*W, 3)

    # Initialize mask with -1 (unmatched pixels)
    mask_flat = torch.full((H * W,), -1, dtype=torch.long)

    # Compare each pixel with colormap entries
    for idx, color in enumerate(colormap_tensor):
        # Find pixels matching this color
        matches = (img_flat == color).all(dim=1)
        mask_flat[matches] = idx

    # Reshape to (H, W)
    mask = mask_flat.reshape(H, W)

    # Replace unmatched pixels (-1) with 0 (or another default value)
    mask[mask == -1] = 0

    return mask



def readColormap(path: str) -> list:
    """
    Reads RGB values from a colormap file and returns them as a list of tuples.
    
    Args:
        path (str): Path to the colormap file (e.g., 'colormap.txt').
    
    Returns:
        list: List of RGB tuples, e.g., [(0, 0, 0), (51, 221, 255), (250, 50, 83)].
    """
    colormap = []
    
    # Open and read the file
    with open(path, 'r') as f:
        for line in f:
            # Strip whitespace and split by comma
            r, g, b = map(int, line.strip().split(','))
            # Add the RGB triplet as a tuple
            colormap.append((r, g, b))
    
    return colormap



def visualize_tensor(T: torch.Tensor, title: str = None):
    """
    Plot a tensor with shape (C, H, W) with matplotlib.
    
    Args:
        T (torch.Tensor): Tensor representing an image with shape (C, H, W).
        title (str, optional): Title of the plot. Defaults to None.
    """
    # Ensure the tensor is on CPU and converted to numpy
    if T.is_cuda:
        T = T.cpu()
    T_np = T.detach().numpy()

    # Check tensor shape
    if len(T.shape) != 3:
        raise ValueError(f"Expected tensor of shape (C, H, W), got {T.shape}")

    channels, height, width = T.shape

    # Handle different channel cases
    if channels == 1:  # Grayscale image
        img = T_np[0]  # (H, W)
        plt.imshow(img, cmap='gray')
    elif channels == 3:  # RGB image
        img = np.transpose(T_np, (1, 2, 0))  # (H, W, C)
        plt.imshow(img)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}. Expected 1 or 3.")

    # Set title if provided
    if title is not None:
        plt.title(title)

    # Remove axis ticks
    plt.axis('off')
    plt.show()


def classes_to_colors(classes: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor representing classes into an RGB image using a colormap.

    Args:
        classes (torch.Tensor): Tensor representing classes. Can be:
            - Shape (C, H, W): One-hot encoded, where C is the number of classes.
            - Shape (H, W): Class indices (integers from 0 to C-1).

    Returns:
        torch.Tensor: RGB image tensor of shape (3, H, W) with values in [0, 255].

    Raises:
        ValueError: If input shapes or colormap length are incompatible.
    """
    # Example colormap with tuples
    colormap = readColormap("./colormap.txt") 
    
    # Ensure tensor is on CPU and detached
    if classes.is_cuda:
        classes = classes.cpu()
    classes = classes.detach()

    # Validate colormap
    if not isinstance(colormap, (list, tuple)) or not all(isinstance(c, (list, tuple)) and len(c) == 3 for c in colormap):
        raise ValueError("colormap must be a list or tuple of [R, G, B] or (R, G, B) elements")

    # Handle input tensor shape
    if len(classes.shape) == 3:  # (C, H, W) - one-hot encoded
        num_classes, height, width = classes.shape
        class_indices = torch.argmax(classes, dim=0)  # (H, W)
    elif len(classes.shape) == 2:  # (H, W) - class indices
        height, width = classes.shape
        class_indices = classes
        num_classes = int(class_indices.max().item()) + 1  # Infer number of classes
    else:
        raise ValueError(f"Expected classes tensor of shape (C, H, W) or (H, W), got {classes.shape}")

    # Validate colormap length
    if len(colormap) < num_classes:
        raise ValueError(f"colormap must have at least {num_classes} colors, got {len(colormap)}")

    # Convert colormap to tensor (handles both lists and tuples)
    colormap_tensor = torch.tensor(colormap, dtype=torch.uint8)  # Shape: (num_colors, 3)

    # Map class indices to colors
    # Shape of class_indices is (H, W), output should be (H, W, 3)
    colored_image = colormap_tensor[class_indices]  # (H, W, 3)

    # Transpose to (3, H, W) for consistency with image conventions
    colored_image = colored_image.permute(2, 0, 1)  # (3, H, W)

    return colored_image


def show_results(dir, img_name, model, tx, device="cuda"):
    """
    Display the original image, ground truth mask, and predicted mask side by side.
    
    Args:
        dir (str): Directory containing images and masks.
        img_name (str): Name of the image file.
        model (torch.nn.Module): Trained model for prediction.
    """
    # Load the image and mask
    img_path = os.path.join(dir, "JPEGImages", img_name+".jpg")
    mask_path = os.path.join(dir, "Annotations", img_name+".png")
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    transform = tx
    
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    to_tensor = torchvision.transforms.ToTensor()
    mask_tensor = to_tensor(mask).to(device)
    # Predict the mask
    model.eval()
    with torch.no_grad():
        raw_digits = model(img_tensor)
        pred_mask = raw_digits.argmax(dim=1).squeeze(dim=0) # Predict and get class indices
        colors = classes_to_colors(pred_mask)  # Convert to RGB image
    
    # Visualize the images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.axis('off')
    plt.title("Ground Truth Mask")
    plt.subplot(1, 3, 3)
    visualize_tensor(colors, title="Predicted Mask")
    plt.show()

    
    
def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    #sHandler = logging.StreamHandler()
    #sHandler.setFormatter(formatter)
    #logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join(train_log,
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='a')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger


