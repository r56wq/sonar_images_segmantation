import torch
import logging
import matplotlib.pyplot as plt
from utils import getLogger
import os

def train(
        model,
        device, 
        train_dataloader, 
        val_dataloader,
        loss_fn,
        evaluator,
        optimizer,
        epochs,
        model_name,
        save_model=False,
        extra_message=None
    ):
    # Get the logger
    logger = getLogger()
    if extra_message:
        logger.info(extra_message)
    # Log the start of training
    logger.info(f"Starting training {model_name}")

    # Lists to store dice scores for plotting
    train_dice_losses = []
    val_dice_scores = []

    # Move model to device
    model = model.to(device)

    # Create a directory to save checkpoints
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_train_dice_score = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            images, true_masks = batch[0], batch[1]
            assert images.shape[1] == 1, "The input image is expected to be one channel"
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # Forward pass
            digits = model(images)
            loss = loss_fn(digits, true_masks)
            dice_score = evaluator(digits, true_masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training dice loss
            epoch_train_dice_score += dice_score.item()

            # Log batch-level metrics
            logger.info(f"In epoch {epoch}, batch {batch_idx}, the dice loss is {loss.item()}, the dice score is {dice_score.item()}")

        # Compute average training dice loss for the epoch
        epoch_train_dice_score /= len(train_dataloader)
        train_dice_losses.append(epoch_train_dice_score)

        # Evaluate at the end of the epoch
        model.eval()
        val_dice_score = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                images, true_masks = batch[0], batch[1]
                assert images.shape[1] == 1, "The input image is expected to be one channel"
                images = images.to(device, dtype=torch.float32)
                true_masks = true_masks.to(device, dtype=torch.long)

                # Forward pass
                digits = model(images)
                dice_score = evaluator(digits, true_masks)
                val_dice_score += dice_score.item()

        # Compute average validation dice score for the epoch
        val_dice_score /= len(val_dataloader)
        val_dice_scores.append(val_dice_score)

        # Log epoch-level metrics
        logger.info(f"In epoch {epoch}, the training dice loss is {epoch_train_dice_score}, the validation dice score is {val_dice_score}")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_dice_loss': epoch_train_dice_score,
                'val_dice_score': val_dice_score,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")

    # Save the final model after training is done
    if save_model:
        final_model_path = os.path.join(checkpoint_dir, f"{model_name}_final.pt")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_dice_loss': epoch_train_dice_score,
            'val_dice_score': val_dice_score,
        }, final_model_path)
        logger.info(f"Final model saved at {final_model_path}")

    # Log the end of training
    logger.info(f"Finishing training {model_name}")

    # Plot training and validation dice scores
    plt.figure()
    plt.plot(train_dice_losses, label="Training Dice Loss")
    plt.plot(val_dice_scores, label="Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score/Loss")
    plt.title(f"Training and Validation Dice Scores for {model_name}")
    plt.legend()
    plt.savefig(f"{model_name}.png")  # Save the plot as a PNG file
    plt.close()

    # Log the plot creation
    logger.info(f"Saved training and validation dice scores plot to {model_name}.png")