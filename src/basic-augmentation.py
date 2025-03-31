import os
import io
import numpy as np
import json
import matplotlib.pyplot as plt
import albumentations as A
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm


class BasicAugmentation:
    def __init__(self, img_dir, xml_dir, output_dir, p=0.5, height=224, width=224):
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
        self.p = p
        self.height = height
        self.width = width

        # Define information to store
        self.images = list()
        self.annotations = dict()
        self.boxes = list()
        self.classes = list()

        # Use the albumentations library for basic transformations
        self.transform = A.Compose([
            A.HorizontalFlip(p=p),
            A.RandomRotate90(p=p),
            A.VerticalFlip(p=p),
            A.RandomCrop(height=height, width=width, p=p),
            A.RandomBrightnessContrast(p=p),
            A.HueSaturationValue(p=p),
            A.GaussNoise(p=p)
        ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

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
        for filename in tqdm(os.listdir(self.img_dir)):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                image_path = os.path.join(self.img_dir, filename)
                try:
                    img = self.preprocess(image_path)
                    self.images.append(img)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

        # Iterate over the xml folder to load and process the annotations
        for file in tqdm(os.listdir(self.xml_dir)):
            if file.endswith('.xml'):
                xml_file = os.path.join(self.xml_dir, file)
                self.annotations[file] = self.parse_voc_xml(xml_file)

        return self.images, self.annotations

    def parse_voc_xml(self, xml_path):
        """
        Parse PASCAL VOC annotation XML file.

        Args:
            xml_path (str): Path to XML annotation file

        Returns:
            tuple: (list of bounding boxes, list of class labels)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            self.boxes.append([xmin, ymin, xmax, ymax])
            self.classes.append(name)

        return self.boxes, self.classes

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

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Define file names
        image_filename = f"{image_id}.bmp"
        annotation_filename = f"{image_id}.json"

        # Save image
        image_path = os.path.join(self.output_dir, image_filename)
        image.save(image_path)

        # Save annotation as JSON
        annotation_data = {"boxes": boxes, "classes": classes}
        annotation_path = os.path.join(self.output_dir, annotation_filename)
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

        # Start performing the basic augmentation
        print(f"Start augmentation for {len(self.images)} images...")

        # Convert PIL images to numpy arrays
        for idx, image in tqdm(enumerate(self.images)):
            image_np = np.array(image)

            # Apply augmentations
            augmented = self.transform(
                image=image_np,
                bboxes=self.annotations["boxes"][idx],
                class_labels=self.annotations["boxes"][idx])

            # Convert back to PIL image
            augmented_image = Image.fromarray(augmented['image'])
            augmented_box = augmented['bboxes']
            augmented_classes = augmented['class_labels']

            # Save augmented items to directory
            self.save_data(augmented_image, augmented_box, augmented_classes, idx)


# Define the directories here:
img_dir = IMAGE_DIR
xml_dir = XML_DIR
output_dir = OUTPUT_DIR

# Create a BasicAugmentation instance
augmenter = BasicAugmentation(
    img_dir=img_dir,
    xml_dir=xml_dir,
    output_dir=output_dir,
    p=0.5,
    height=224,
    width=224,
)
augmenter.augment()
