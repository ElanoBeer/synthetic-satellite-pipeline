import os
import io
import json
import random
import xml.etree.ElementTree as ET
import albumentations as A
import cv2
import numpy as np
from PIL import Image
from skimage import metrics
from tqdm import tqdm

class ObjectInsertion:
    def __init__(self,
                 img_dir,
                 obj_dir,
                 xml_dir,
                 out_dir,
                 input_size,
                 target_size=(224, 224),
                 margin=20,
                 max_iter=100,
                 sample_method="without"):


        # Initialize input variables
        self.img_dir = img_dir
        self.obj_dir = obj_dir
        self.xml_dir = xml_dir
        self.out_dir = out_dir
        self.input_size = input_size
        self.target_size = target_size
        self.margin = margin
        self.max_iter = max_iter
        self.sample_method = sample_method

        # Initialize variables to store data
        self.objects = dict()
        self.images = list()
        self.annotations = dict()
        self.masks = list()


    def preprocess(self, img_path):
        """
            Load and preprocess an image.

            Args:
                img_path (str): Path to the image file

            Returns:
                PIL.Image: Preprocessed image
            """
        # Load image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image
        resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        return resized

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
        height, width = self.target_size

        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Scale bounding boxes
            xmin = (xmin / self.input_size[0]) * width
            ymin = (ymin / self.input_size[1]) * height
            xmax = (xmax / self.input_size[0]) * width
            ymax = (ymax / self.input_size[1]) * height

            boxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(name)

        return boxes, class_labels

    def load_images(self):
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

    # def save_data(self, image, boxes, classes, image_id, image_dir_name, annot_dir_name):
    #     """
    #     Save the images to a new directory.
    #
    #     Args:
    #         self: class information
    #         image: PIL.Image
    #         boxes: List of bounding boxes
    #         classes: List of class labels
    #         image_id: Unique image ID representing the index
    #
    #     Returns:
    #         None: directory
    #     """
    #
    #     # Define subdirectories for images and annotations
    #     images_dir = os.path.join(self.out_dir, image_dir_name)
    #     annotations_dir = os.path.join(self.out_dir, annot_dir_name)
    #
    #     # Create directories if they do not exist
    #     os.makedirs(images_dir, exist_ok=True)
    #     os.makedirs(annotations_dir, exist_ok=True)
    #
    #     # Define file names
    #     image_filename = f"{image_id}.png"
    #     annotation_filename = f"{image_id}.json"
    #
    #     # Save image
    #     image_path = os.path.join(images_dir, image_filename)
    #     image.save(image_path)
    #
    #     if boxes:
    #         # Save annotation as JSON
    #         annotation_data = {"boxes": boxes, "classes": classes}
    #         annotation_path = os.path.join(annotations_dir, annotation_filename)
    #         with open(annotation_path, "w") as f:
    #             json.dump(annotation_data, f)

    def save_data(self, img, idx):

        # Create the output directory if it doesn't exist
        os.makedirs(self.out_dir, exist_ok=True)

        # Construct the full path for saving the image
        out_path = os.path.join(self.out_dir, idx)

        # Save the image
        cv2.imwrite(out_path, img)
        print(f"Image saved at {out_path}")

    def load_objects(self):
        """
        Load all objects from a directory and preprocess them.

        Returns:
            self.objects: A dictionary of objects
        """

        # Iterate over the image folder to load and process the images
        print(f"Loading images and annotations from {self.obj_dir}...")
        for file in tqdm(os.listdir(self.obj_dir)):
            obj = self.preprocess(file)
            self.objects[file] = obj

        return self.objects

    def locate_anchor(self, obj_shape, dst_bbox):
        """
        Generates a new bounding box that is close to but does not overlap with the given bounding box.
        The new bounding box may have a different size.

        Args:
            src_bbox (list): [x_min, y_min, x_max, y_max] representing the original bounding box.
            min_distance (int): Minimum distance to move the new bounding box away.
            max_distance (int): Maximum distance to move the new bounding box away.
            size_variation (float): Maximum percentage by which the new box size can vary (e.g., 0.5 = ±50%).

        Returns:
            list: [x_min, y_min, x_max, y_max] for the new non-overlapping bounding box.
        """

        # Initialize minimum and maximum distance to move the new bounding box
        min_distance = 0.1 * self.target_size[0]
        max_distance = 0.5 * self.target_size[0]

        # Calculate scaling factors
        scale_x = self.target_size[0] / self.input_size[0]
        scale_y = self.target_size[1] / self.input_size[1]

        # Extract destination bounding box information
        x_min, y_min, x_max, y_max = dst_bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = self.target_size[0]
        height = self.target_size[1]

        for i in range(self.max_iter):
            # Compute diagonal distances (to ensure proper separation)
            orig_diag = np.sqrt(width ** 2 + height ** 2) / 2

            # Choose a random angle (0 to 360 degrees)
            angle = random.uniform(0, 2 * np.pi)

            # Ensure the distance is at least greater than the diagonal
            min_safe_distance = orig_diag + min_distance
            max_safe_distance = orig_diag + max_distance
            distance = random.uniform(min_safe_distance, max_safe_distance)
            # print(min_safe_distance, max_safe_distance, distance)

            # Compute new center position using polar coordinates
            new_center_x = (center_x + distance * np.cos(angle)) * scale_x
            new_center_y = (center_y + distance * np.sin(angle)) * scale_y
            # print(new_center_x, new_center_y)

            # Compute new bounding box coordinates
            new_x_min = int(new_center_x - obj_shape[0] / 2)
            new_y_min = int(new_center_y - obj_shape[1] / 2)
            new_x_max = int(new_center_x + obj_shape[0] / 2)
            new_y_max = int(new_center_y + obj_shape[1] / 2)

            new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]

            if min(new_bbox) > self.margin & max(new_bbox) < (self.target_size[0] - self.margin):
                final_bbox = new_bbox
                return final_bbox

        return None

    def create_masks(self):
        """
            Create a binary mask from bounding boxes found in the XML file,
            adjusting for resized coordinates.

            Returns:
                np.ndarray: Binary mask with bounding box areas marked.
        """

        # Initiate a list of masks
        masks = list()

        # Extract the bounding boxes
        boxes = next(iter(self.annotations.values()))

        print(boxes)
        # Loop through each bounding box
        for box in boxes[0]:
            # Initialize the mask with zeros
            mask = np.zeros(self.target_size, dtype=np.uint8)

            # Extract coordinates from the bounding box
            xmin, ymin, xmax, ymax = map(int, box)

            # Fill the corresponding area in the mask with 255 (white)
            mask[ymin:ymax, xmin:xmax] = 255

            # Soften the edges of the mask
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            # Add to the list of masks
            masks.append(mask)
        return masks

    def check_overlap(self, tabu_lst, candidate):
        """
        Checks if the candidate bounding box overlaps with any bounding box in the tabu list.

        Args:
            tabu_lst (list): List of bounding boxes, each in the format [x_min, y_min, x_max, y_max].
            candidate (list): The candidate bounding box in the format [x_min, y_min, x_max, y_max].

        Returns:
            bool: True if the candidate overlaps with any box in the tabu list, False otherwise.
        """
        # Convert both tabu_lst and candidate into NumPy arrays
        tabu_arr = np.array(tabu_lst)
        candidate_arr = np.array(candidate)

        # Extract candidate box coordinates
        c_xmin, c_ymin, c_xmax, c_ymax = candidate_arr

        # Create mask for overlaps: True if there is overlap, False otherwise
        overlaps = ~((tabu_arr[:, 2] < c_xmin) | (tabu_arr[:, 0] > c_xmax) |  # check x overlap
                     (tabu_arr[:, 3] < c_ymin) | (tabu_arr[:, 1] > c_ymax))  # check y overlap

        # Return True if any overlap is detected, else False
        return np.any(overlaps)

    def check_insertion(self, image1, image2):
        """

        """
        # Convert images to grayscale
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_score = metrics.structural_similarity(image1_gray, image2_gray)
        print(f"SSIM Score: ", round(ssim_score, 2))
        return ssim_score

    def object_insertion(self):
        """


        """
        # Load the background images and annotations from the directories
        self.load_images()

        # Load the objects
        self.load_objects()

        # Create masks for the object images
        self.create_masks()

        # Start performing the object insertion
        print(f"Start insertion objects for {len(self.images)} images...")
        score_lst = list()
        count = 0

        # Iterate of the background images
        for img in tqdm(self.annotations.keys()):
            dst_bbox = self.annotations[img]

            # Create a list to store the new bounding boxes
            new_bbox_lst = list()

            # Some images have multiple bounding boxes
            print(f"Calculating new bounding box for {len(dst_bbox)} images...")
            for bbox in dst_bbox:

                # Sample a random object from the set
                if self.sample_method == "without":
                    # Without replacement
                    obj = np.random.sample(self.objects[img])

                else:
                    # With replacement
                    obj = np.random.choice(self.objects[img])

                # Store the object image size
                obj_shape = obj.shape[:2]

                print(f"Finding new bounding box...")
                # Calculate a new bbox for insertion
                for i in range(self.max_iter):
                    # Search for a candidate solution
                    new_bbox = self.locate_anchor(obj_shape, bbox)
                    if not self.check_overlap(new_bbox, new_bbox_lst):
                        new_bbox_lst.append(new_bbox)
                        break
                    else:
                        continue

                # Calculate the center of the candidate bounding box
                center = ((new_bbox[0] + new_bbox[2]) / 2), ((new_bbox[1] + new_bbox[3]) / 2)

                # Store the old image for comparison
                old_img = img

                # Iteratively create a normal clone of the image with an inserted object
                img = cv2.seamlessClone(
                    obj, img, new_bbox, center,
                    flags=cv2.NORMAL_CLONE,
                )

                # Check if the new image has an additional object
                ssim_score = self.check_insertion(img, old_img)
                score_lst.append([ssim_score, img])
                print(f"Clone score: {ssim_score}")

                # Save object inserted images to directory
                self.save_data(img, f"{img}_{count}")
                count += 1


# Define the directories here:
img_dir = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/masati-thesis/images"
obj_dir = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/MasatiV2/MasatiV2Boats"
xml_dir = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/masati-thesis/annotations"
output_dir = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/masati-thesis/"

# Create a ObjectInsertion instance
augmenter = ObjectInsertion(
    img_dir=img_dir,
    obj_dir=obj_dir,
    xml_dir=xml_dir,
    out_dir=output_dir,
    input_size=(512,512),
    target_size=(224, 224),
    margin=20,
    max_iter=100,
    sample_method="without"
)
augmenter.object_insertion()














