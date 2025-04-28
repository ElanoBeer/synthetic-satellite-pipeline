import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import random
from PIL import Image
import yaml


class VesselDetection:
    def __init__(self,
                 images_dir,
                 annotations_dir,
                 model_dir,
                 output_dir='vessel_detection',
                 img_size=(640,640),
                 n_folds=5,
                 epochs=50,
                 batch_size=16):
        """
        Initialize vessel detection pipeline with YOLOv8

        Args:
            images_dir: Directory containing satellite images
            annotations_dir: Directory to JSON annotation files
            output_dir: Directory to store processed data and results
            img_size: Input image size for YOLOv8
            n_folds: Number of folds for cross-validation
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.images_dir = images_dir
        self.annotations_file = annotations_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.n_folds = n_folds
        self.epochs = epochs
        self.batch_size = batch_size

        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        self.data_dir = os.path.join(output_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Class names (assuming vessel is the only class)
        self.classes = ['vessel']

    def load_annotations(self):
        """Load and parse individual JSON annotation files"""
        print("Loading annotations from individual JSON files...")

        annotation_dir = self.annotations_file  # Repurpose this parameter as directory path
        self.annotations = {}
        self.image_list = []

        # Find all JSON files in the annotation directory
        json_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
        print(f"Found {len(json_files)} annotation files")

        # Process each annotation file
        for json_file in tqdm(json_files):
            # Extract corresponding image filename (assuming same basename)
            base_name = os.path.splitext(json_file)[0]
            possible_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

            # Find matching image file
            img_file = None
            for ext in possible_extensions:
                potential_img = base_name + ext
                if os.path.exists(os.path.join(self.images_dir, potential_img)):
                    img_file = potential_img
                    break

            if img_file is None:
                print(f"Warning: No matching image found for annotation {json_file}")
                continue

            # Load the JSON annotation
            with open(os.path.join(annotation_dir, json_file), 'r') as f:
                try:
                    annotation_data = json.load(f)

                    # Store the annotation with image filename as key
                    self.annotations[img_file] = annotation_data
                    self.image_list.append(img_file)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON file {json_file}")

        self.annotation_format = 'per_file'  # Mark that we're using the per-file format
        print(f"Successfully loaded annotations for {len(self.image_list)} images")
        return self.image_list

    def prepare_yolo_dataset(self):
        """Convert annotations to YOLO format and organize dataset"""
        print("Converting annotations to YOLO format...")

        # Create directories for YOLO format
        labels_dir = os.path.join(self.data_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        images_output_dir = os.path.join(self.data_dir, 'images')
        os.makedirs(images_output_dir, exist_ok=True)

        # Process each image
        for img_name in tqdm(self.image_list):
            # Copy image to dataset directory
            img_src = os.path.join(self.images_dir, img_name)
            img_dst = os.path.join(images_output_dir, img_name)

            if os.path.exists(img_src):
                # Open image
                img = Image.open(img_src)

                # Resize image
                img_resized = img.resize(self.img_size)

                # Save resized image to output directory
                img_resized.save(img_dst)

                # Get resized image dimensions (should now be 640, 640)
                img_width, img_height = img_resized.size

                # Create label file (txt format for YOLO)
                label_filename = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_filename)

                # Extract annotations for this image
                bboxes = self._extract_bboxes(img_name, img_width, img_height)

                # Write YOLO format labels
                with open(label_path, 'w') as f:
                    for cls_id, x_center, y_center, width, height in bboxes:
                        f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
            else:
                print(f"Warning: Image {img_name} not found in {self.images_dir}")

        return images_output_dir, labels_dir

    def _extract_bboxes(self, img_name, img_width, img_height):
        """Extract bounding boxes from annotations for a specific image"""
        bboxes = []

        if self.annotation_format == 'per_file':
            if img_name in self.annotations:
                annotation_data = self.annotations[img_name]
                bbox_list = annotation_data['boxes']

                if bbox_list:
                    for bbox_data in bbox_list:
                        # Extract coordinates based on available format
                        if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                            # If it's a simple list of coordinates
                            x, y, w, h = bbox_data[:4]
                        elif isinstance(bbox_data, dict):
                            if 'bbox' in bbox_data:
                                x, y, w, h = bbox_data['bbox']
                            elif all(k in bbox_data for k in ['x', 'y', 'width', 'height']):
                                x, y, w, h = bbox_data['x'], bbox_data['y'], bbox_data['width'], bbox_data[
                                    'height']
                            else:
                                # Try other common formats
                                if all(k in bbox_data for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                                    xmin, ymin, xmax, ymax = bbox_data['xmin'], bbox_data['ymin'], bbox_data[
                                        'xmax'], bbox_data['ymax']
                                    x, y = xmin, ymin
                                    w, h = xmax - xmin, ymax - ymin
                                else:
                                    print(f"Warning: Unrecognized bbox format in {img_name}")
                                    continue
                        else:
                            print(f"Warning: Could not parse bbox data in {img_name}")
                            continue

                        # Convert to YOLO format
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width = w / img_width
                        height = h / img_height

                        cls_id = 0  # Assuming 'vessel' is the only class
                        bboxes.append((cls_id, x_center, y_center, width, height))

        return bboxes

    def create_dataset_yaml(self, train_files, val_files, fold):
        """Create YAML configuration file for YOLOv8 training"""
        data_yaml_path = os.path.join(self.data_dir, f'fold_{fold}_dataset.yaml')

        # Write train and val image paths to text files
        train_list_file = os.path.join(self.data_dir, f'fold_{fold}_train.txt')
        with open(train_list_file, 'w') as f:
            for img_path in train_files:
                f.write(f"{img_path}\n")

        val_list_file = os.path.join(self.data_dir, f'fold_{fold}_val.txt')
        with open(val_list_file, 'w') as f:
            for img_path in val_files:
                f.write(f"{img_path}\n")

        # Create YAML content
        yaml_content = {
            'path': os.path.abspath(self.data_dir),
            'train': train_list_file,
            'val': val_list_file,
            'names': {i: name for i, name in enumerate(self.classes)}
        }

        # Write YAML file
        with open(data_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        return data_yaml_path

    def perform_cross_validation(self, images_dir, labels_dir):
        """Perform k-fold cross-validation for YOLOv8 training"""
        print(f"Starting {self.n_folds}-fold cross-validation...")

        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Setup k-fold cross validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Store results for each fold
        all_results = []

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
            print(f"\nTraining Fold {fold + 1}/{self.n_folds}")

            # Get train and validation splits
            train_images = [image_files[i] for i in train_idx]
            val_images = [image_files[i] for i in val_idx]

            # Get full paths for images
            train_paths = [os.path.join(images_dir, img) for img in train_images]
            val_paths = [os.path.join(images_dir, img) for img in val_images]

            # Create dataset YAML
            data_yaml = self.create_dataset_yaml(train_paths, val_paths, fold)

            # Define model output directory
            model_dir = os.path.join(self.results_dir, f'fold_{fold}')
            os.makedirs(model_dir, exist_ok=True)

            # Initialize and train YOLOv8 model
            model = YOLO(self.model_dir)  # You can change to m, l, x for larger models

            results = model.train(
                data=data_yaml,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                project=self.results_dir,
                name=f'fold_{fold}',
                exist_ok=True
            )

            # Validate the model
            val_results = model.val(data=data_yaml)

            # Store results
            metrics = {
                'fold': fold,
                'map50': val_results.box.map50,
                'map50-95': val_results.box.map,
                'precision': val_results.box.precision,
                'recall': val_results.box.recall
            }
            all_results.append(metrics)

            print(
                f"Fold {fold + 1} results: mAP@0.5 = {metrics['map50']:.4f}, mAP@0.5:0.95 = {metrics['map50-95']:.4f}")

        return all_results

    def summarize_results(self, results):
        """Summarize cross-validation results"""
        print("\n===== Cross-Validation Results =====")

        # Calculate mean and std for metrics
        metrics = ['map50', 'map50-95', 'precision', 'recall']
        summary = {}

        for metric in metrics:
            values = [r[metric] for r in results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[metric] = (mean_val, std_val)

            print(f"Mean {metric}: {mean_val:.4f} ± {std_val:.4f}")

        # Plot results
        self.plot_cv_results(results)

        return summary

    def plot_cv_results(self, results):
        """Plot cross-validation results"""
        metrics = ['map50', 'map50-95', 'precision', 'recall']
        plt.figure(figsize=(12, 8))

        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            values = [r[metric] for r in results]
            folds = [r['fold'] + 1 for r in results]

            plt.bar(folds, values)
            plt.axhline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
            plt.title(f'{metric.upper()} by Fold')
            plt.xlabel('Fold')
            plt.ylabel(metric)
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'cv_results.png'))
        plt.close()

    def run_pipeline(self):
        """Run the complete vessel detection pipeline"""
        # Load annotations
        self.load_annotations()

        # Prepare dataset in YOLO format
        images_dir, labels_dir = self.prepare_yolo_dataset()

        # Perform cross-validation
        results = self.perform_cross_validation(images_dir, labels_dir)

        # Summarize results
        summary = self.summarize_results(results)

        # Save results to JSON
        with open(os.path.join(self.results_dir, 'cv_results.json'), 'w') as f:
            json.dump({
                'individual_folds': results,
                'summary': {k: {'mean': float(v[0]), 'std': float(v[1])} for k, v in summary.items()}
            }, f, indent=4)

        # Train final model using all data (optional)
        print("\nTraining final model on all data...")
        all_images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                      if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Split into train and val (small validation set just for metrics)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.9)
        final_train = all_images[:split_idx]
        final_val = all_images[split_idx:]

        # Create dataset YAML
        final_yaml = self.create_dataset_yaml(final_train, final_val, 'final')

        # Train final model
        final_model = YOLO(self.model_dir)
        final_model.train(
            data=final_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            project=self.results_dir,
            name='final_model',
            exist_ok=True
        )

        # Validate final model
        final_results = final_model.val(data=final_yaml)

        print("\n===== Final Model Results =====")
        print(f"mAP@0.5: {final_results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {final_results.box.map:.4f}")

        print("\nVessel detection pipeline complete!")
        return final_model

    def visualize_predictions(self, num_samples=5):
        """Visualize model predictions on sample images"""
        print("Visualizing model predictions...")

        # # Use final model or specified model
        # if model_path is None:
        #     model_path = os.path.join(self.results_dir, 'final_model', 'weights', 'best.pt')
        #
        # if not os.path.exists(model_path):
        #     print(f"Model not found at {model_path}")
        #     return

        # Load the model
        model = YOLO(self.model_dir)

        # Get sample images
        images_dir = os.path.join(self.data_dir, 'images')
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if num_samples > len(image_files):
            num_samples = len(image_files)

        random.shuffle(image_files)
        sample_images = image_files[:num_samples]

        # Create visualization directory
        vis_dir = os.path.join(self.results_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Process each sample image
        for img_file in sample_images:
            img_path = os.path.join(images_dir, img_file)

            # Run prediction
            results = model(img_path)

            # Save visualization
            vis_path = os.path.join(vis_dir, f"pred_{img_file}")

            # Get the result image with annotations
            result_img = results[0].plot()

            # Save the visualization
            cv2.imwrite(vis_path, result_img)

        print(f"Saved {num_samples} visualization images to {vis_dir}")


# Example usage
if __name__ == "__main__":
    # Parameters
    IMAGES_DIR = "E:\Datasets\masati-thesis\clones"  # Update with your image directory
    ANNOTATIONS_DIR = "E:\Datasets\masati-thesis\clone_annotations"  # Update with your annotations file
    OUTPUT_DIR = "E:/Datasets/masati-thesis/vessel_detection_output"
    MODEL_DIR = "/models/yolov8n.pt"
    N_FOLDS = 5  # Number of cross-validation folds
    EPOCHS = 50  # Training epochs per fold

    # Initialize and run pipeline
    detector = VesselDetection(
        images_dir=IMAGES_DIR,
        annotations_dir=ANNOTATIONS_DIR,
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        img_size=(480,480),
        n_folds=N_FOLDS,
        epochs=EPOCHS,
        batch_size=16
    )

    final_model = detector.run_pipeline()

    # Visualize some predictions
    detector.visualize_predictions(num_samples=5)
