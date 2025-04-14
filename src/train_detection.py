from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import time
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic Image Dataset
# ----------------------------------------------------------------------------------------------------------------------

class SyntheticVesselDataset(VisionDataset):
    def __init__(self, image_paths, annotations, transforms=None):
        super().__init__(image_paths, transforms=transforms)
        self.image_paths = image_paths
        self.annotations = annotations  # List of dicts with "boxes", no "labels"
        self.transforms = transforms

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize((224, 224))

        boxes = self.annotations[idx]["boxes"]  # List of [x1, y1, x2, y2]
        boxes = [[max(0, b[0]), max(0, b[1]), max(0, b[2]), max(0, b[3])] for b in boxes]  # Ensure non-negative values
        num_objs = len(boxes)

        # Since all objects are vessels, assign label 1 to all
        labels = [1] * num_objs

        # Initialize target dictionary
        target = {}

        if num_objs > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        else:
            # Handle empty boxes
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)

        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_paths)


# ----------------------------------------------------------------------------------------------------------------------
# Load Pretrained Faster R-CNN & Freeze Feature Extractor
# ----------------------------------------------------------------------------------------------------------------------

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Freeze the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Replace the classifier head (class 0 is background, class 1 is 'vessel')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ----------------------------------------------------------------------------------------------------------------------
# Utilities for Training and Evaluation
# ----------------------------------------------------------------------------------------------------------------------

def collate_fn(batch):
    """Custom collate function for detection batches"""
    return tuple(zip(*batch))


def plot_training_metrics(metrics, save_path='training_metrics.png'):
    """Plot training and validation metrics over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(metrics['train_loss'], label='Train Loss')
    ax1.plot(metrics['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot mAP
    ax2.plot(metrics['val_map'], label='mAP@0.5:0.95')
    ax2.plot(metrics['val_map50'], label='mAP@0.5')
    ax2.set_title('Mean Average Precision')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    metric = MeanAveragePrecision()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get model predictions
            outputs = model(images)

            # Update metrics - needs formatted inputs
            preds = []
            for output in outputs:
                pred = {
                    'boxes': output['boxes'],
                    'scores': output['scores'],
                    'labels': output['labels']
                }
                preds.append(pred)

            metric.update(preds, targets)

            # For getting validation lowhere pythonss, we need to run in training mode temporarily
            model.train()
            loss_dict = model(images, targets)
            model.eval()

            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    # Calculate metrics
    metrics = metric.compute()
    avg_loss = total_loss / len(data_loader)

    return {
        'loss': avg_loss,
        'map': metrics['map'].item(),
        'map50': metrics['map_50'].item(),
    }


# ----------------------------------------------------------------------------------------------------------------------
# Main Training Script
# ----------------------------------------------------------------------------------------------------------------------

def train_model(image_paths, annotations, output_dir, num_epochs=25, batch_size=4, val_split=0.2, lr=0.001):
    """Train object detection model with monitoring and validation"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize transforms
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and split into train/val
    full_dataset = SyntheticVesselDataset(image_paths, annotations, transforms=transform)

    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Perform the split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = get_model(num_classes=2)  # Background + vessel
    model.to(device)

    # Only update the parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )

    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_map': [],
        'val_map50': []
    }

    best_map = 0.0

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True)
        for images, targets in progress_bar:
            # Move data to device
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Update metrics
            epoch_loss += losses.item()
            progress_bar.set_postfix({"Loss": f"{losses.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)

        # Validation phase
        print(f"\nValidating model after epoch {epoch + 1}...")
        val_metrics = evaluate_model(model, val_loader, device)

        # Update metrics
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_map'].append(val_metrics['map'])
        metrics['val_map50'].append(val_metrics['map50'])

        # Update learning rate based on validation loss
        lr_scheduler.step(val_metrics['loss'])

        # Check if this is the best model
        if val_metrics['map'] > best_map:
            best_map = val_metrics['map']
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"Saved new best model with mAP: {best_map:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        # Plot and save metrics
        plot_training_metrics(metrics, save_path=os.path.join(output_dir, 'metrics.png'))

        # Print epoch summary
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {time_elapsed:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"mAP@0.5:0.95: {val_metrics['map']:.4f}, mAP@0.5: {val_metrics['map50']:.4f}")
        print("-" * 80)

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, os.path.join(output_dir, 'final_model.pth'))

    return model, metrics


# ----------------------------------------------------------------------------------------------------------------------
# Script Execution
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace these with your actual paths
    image_dir = "E:/Datasets/masati-thesis/augmented-images"
    annotation_dir = "E:/Datasets/masati-thesis/augmented-annotations"

    # Create list of full image paths
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]

    annotations = list()
    for file in tqdm(os.listdir(annotation_dir)):
        json_path = os.path.join(os.path.join("E:/Datasets/masati-thesis/augmented-annotations", file))
        with open(json_path, "r") as f:
            data = json.load(f)
            annotations.append(data)

    # Create output directory for saving models and metrics
    output_dir = "E:/Datasets/masati-thesis/vessel-detection-output"

    # Train the model
    model, metrics = train_model(
        image_paths=image_paths,
        annotations=annotations,
        output_dir=output_dir,
        num_epochs=25,
        batch_size=32,
        val_split=0.2,
        lr=0.001
    )

    print("Training completed successfully!")
