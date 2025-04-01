import os
import io
import numpy as np
import json
#import matplotlib.pyplot as plt
import albumentations as A
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm


class BasicAugmentation:
    def __init__(self, img_dir, xml_dir, output_dir, size, p=0.5, height=224, width=224, n_augmentations=1):
        """
        Initialize the basic augmentation pipeline.
        This pipeline utilizes albumentations to perform geometric and
        color space transformations.

        Args:
            p (float): Probability of applying each augmentation
        """

        # Define the input variable to use
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.output_dir = output_dir
        self.size = size
        self.p = p
        self.height = height
        self.width = width
        self.n_augmentations = n_augmentations

        # Define information to store
        self.images = list()
        self.annotations = dict()

        # Use the albumentations library for basic transformations
        self.transform = A.Compose(transforms=[
            A.HorizontalFlip(p=p),
            A.RandomRotate90(p=p),
            A.VerticalFlip(p=p),
            A.RandomCrop(height=height, width=width, p=p),
            A.RandomBrightnessContrast(p=p),
            A.HueSaturationValue(p=p),
            A.GaussNoise(std_range=(0.1, 0.25), mean_range=(0,0), p=p)
        ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def calculate_size(self, augmentations_per_image=1,
                                         original_image_size_mb=None, sample_image=None):
        """
        Calculate the approximate dataset size after augmentation.

        Args:
            augmentations_per_image (int): Number of augmented versions per original image
            original_image_size_mb (float, optional): Average size of one image in MB
            sample_image (PIL.Image, optional): Sample image to calculate average size

        Returns:
            dict: Dictionary containing:
                - total_images: Total number of images after augmentation
                - original_size_mb: Approximate size of original dataset in MB
                - augmented_size_mb: Approximate size of augmented dataset in MB
        """
        # Calculate total number of images after augmentation
        total_images = len(self.images) * (augmentations_per_image + 1)  # +1 for original images

        # If image size not provided, try to calculate from sample image
        if original_image_size_mb is None and sample_image is not None:
            # Convert PIL image to bytes and calculate size in MB
            with io.BytesIO() as bio:
                sample_image.save(bio, format='PNG')
                original_image_size_mb = len(bio.getvalue()) / (1024 * 1024)

        # Calculate sizes if we have image size information
        if original_image_size_mb is not None:
            original_size_mb = len(self.images) * original_image_size_mb
            augmented_size_mb = total_images * original_image_size_mb
        else:
            original_size_mb = None
            augmented_size_mb = None

        return {
            'total_images': total_images,
            'original_size_mb': original_size_mb,
            'augmented_size_mb': augmented_size_mb
        }

    def preprocess(self, img_path):
        """
            Load and preprocess an image.

            Args:
                img_path (str): Path to the image file

            Returns:
                PIL.Image: Preprocessed image
            """
        # Load image
        image = Image.open(img_path)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image
        resized = image.resize((self.height, self.width), Image.Resampling.LANCZOS)

        return resized

    def load_data(self):
        """
        Load all images from a directory and preprocess them.

        Args:
            img_dir (str): Path to the directory containing images

        Returns:
            images: List of image preprocessed images
            annotations: Dictionary of annotations
        """

        # Define acceptable file formats
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        # Iterate over the image folder to load and process the images
        print(f"Loading images and annotations from {self.img_dir}...")
        for file in tqdm(os.listdir(self.img_dir)):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_path = os.path.join(self.img_dir, file)
                img = self.preprocess(image_path)
                self.images.append(img)

                xml_file = os.path.splitext(file)[0] + ".xml"
                xml_path = os.path.join(self.xml_dir, xml_file)

                # Save the annotations for images with objects
                if os.path.exists(xml_path):
                    boxes, class_labels = self.parse_voc_xml(xml_path)
                    self.annotations[os.path.splitext(file)[0]] = (boxes, class_labels)

                else:
                    self.annotations[os.path.splitext(file)[0]] = ([], [])

        return self.images, self.annotations

    def parse_voc_xml(self, xml_path):
        """
        Parse PASCAL VOC annotation XML file and adjust bounding boxes.

        Args:
            xml_path (str): Path to XML annotation file

        Returns:
            tuple: (list of scaled bounding boxes, list of class labels)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        class_labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Scale bounding boxes
            xmin = (xmin / self.size) * self.width
            ymin = (ymin / self.size) * self.height
            xmax = (xmax / self.size) * self.width
            ymax = (ymax / self.size) * self.height

            boxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(name)

        return boxes, class_labels

    def save_data(self, image, boxes, classes, image_id):
        """
        Save the images to a new directory.

        Args:
            self: class information
            image: PIL.Image
            boxes: List of bounding boxes
            classes: List of class labels
            image_id: Unique image ID representing the index

        Returns:
            None: directory
        """

        # Define subdirectories for images and annotations
        images_dir = os.path.join(self.output_dir, "augmented-images")
        annotations_dir = os.path.join(self.output_dir, "augmented-annotations")

        # Create directories if they do not exist
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        # Define file names
        image_filename = f"{image_id}.png"
        annotation_filename = f"{image_id}.json"

        # Save image
        image_path = os.path.join(images_dir, image_filename)
        image.save(image_path)

        if boxes:
            # Save annotation as JSON
            annotation_data = {"boxes": boxes, "classes": classes}
            annotation_path = os.path.join(annotations_dir, annotation_filename)
            with open(annotation_path, "w") as f:
                json.dump(annotation_data, f)


    def augment(self):
        """
        Apply augmentations to an image.

        Args:
            image (PIL.Image): Input image

        Returns:
            PIL.Image: Augmented image
        """

        # Load the images and annotations from the directories
        self.images, self.annotations = self.load_data()

        # Calculate final dataset size
        print(f"Calculating augmented dataset size...")
        print(f"{self.calculate_size(augmentations_per_image=3, sample_image=self.images[0])}")

        # Start performing the basic augmentation
        print(f"Start augmentation for {len(self.images)} images...")

        # Convert PIL images to numpy arrays
        for (filename, image) in tqdm(zip(self.annotations.keys(), self.images), total=len(self.images)):
            image_np = np.array(image)

            # Retrieve bounding boxes and class labels using the filename
            boxes, class_labels = self.annotations[filename]

            for i in range(self.n_augmentations):
                # Apply augmentations
                augmented = self.transform(
                    image=image_np,
                    bboxes=boxes,
                    class_labels=class_labels)

                # Convert back to PIL image
                augmented_image = Image.fromarray(augmented['image'])
                augmented_box = augmented['bboxes']
                augmented_classes = augmented['class_labels']

                # Save augmented items to directory
                self.save_data(augmented_image, augmented_box, augmented_classes, f"{filename}_{i}")


# Define the directories here:
img_dir = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/masati-thesis/images"
xml_dir = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/masati-thesis/annotations"
output_dir = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/masati-thesis/"

# Create a BasicAugmentation instance
augmenter = BasicAugmentation(
    img_dir=img_dir,
    xml_dir=xml_dir,
    output_dir=output_dir,
    size=512,
    p=0.5,
    height=224,
    width=224,
    n_augmentations=3,
)
augmenter.augment()
