import os
import json
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import time

# ----------------------------------------------------------------------------------------------------------------------
# Synthetic Image Dataset
# ----------------------------------------------------------------------------------------------------------------------

class SyntheticVesselDataset(VisionDataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms

        # Assumes matching filenames like img001.jpg ↔ img001.json
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', 'bmp', '.tif', '.tiff'))
        ])
        self.annotation_paths = sorted([
            os.path.join(annotation_dir, fname)
            for fname in os.listdir(annotation_dir)
            if fname.lower().endswith('.json')
        ])

        assert len(self.image_paths) == len(self.annotation_paths), \
            "Mismatch between image and annotation count"

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = Image.open(image_path).convert("RGB")
        with open(annotation_path, "r") as f:
            data = json.load(f)

        boxes = data.get("boxes", [])
        num_objs = len(boxes)

        # Since all objects are vessels, assign label 1 to all
        labels = [1] * num_objs

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_paths)


# ----------------------------------------------------------------------------------------------------------------------
# Load Pretrained Faster R-CNN & Freeze Feature Extractor
# ----------------------------------------------------------------------------------------------------------------------

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Freeze the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Replace the classifier head (we assume class 1 is 'vessel')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    return model

# ----------------------------------------------------------------------------------------------------------------------
# Training Script
# ----------------------------------------------------------------------------------------------------------------------

# Define the directories here:
image_path = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/masati-thesis/images"
annotations = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/masati-thesis/annotations"

# Dummy loader data for example
train_dataset = SyntheticVesselDataset(image_path, annotations, transforms=T.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=2)  # background + vessel
model.to(device)

# Only update the detection head (since backbone is frozen)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    start = time.time()

    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    end = time.time()
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f} - Time: {end - start:.2f}s")