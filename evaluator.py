from Data_Loader import Images_Dataset_folder
import os
import torch
from torch.utils.data import DataLoader
from utils import hard_dice_score
import torch.nn.functional as F  # Use F.interpolate for resizing

def evaluate(test_dir, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the test dataset
    test_feature = os.path.join(test_dir, "JPEGImages")
    test_label = os.path.join(test_dir, "Annotations")
    test_dataset = Images_Dataset_folder(test_feature, test_label)
    
    # Create a DataLoader with batch_size=1
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Initialize Dice score calculator
    dice_calculator = hard_dice_score()
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            # Move data to the appropriate device
            images = images.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            outputs = model(images)
            
            # Resize the model's output to match the label size
            outputs_resized = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            
            # Convert model output to binary format
            if outputs_resized.shape[1] > 1:  # Multi-class segmentation
                preds = torch.argmax(outputs_resized, dim=1)  # Get the class with the highest probability
                preds = torch.nn.functional.one_hot(preds, num_classes=outputs_resized.shape[1]).permute(0, 3, 1, 2)  # Convert to one-hot
            else:  # Binary segmentation
                preds = (torch.sigmoid(outputs_resized) > 0.5).float()  # Apply threshold to get binary mask
            
            # Calculate Dice score
            dice_score = dice_calculator(preds, labels)
            total_dice += dice_score.item()
            num_samples += 1
    
    # Calculate average Dice score
    avg_dice = total_dice / num_samples
    print(f"Average Dice Score: {avg_dice:.4f}")