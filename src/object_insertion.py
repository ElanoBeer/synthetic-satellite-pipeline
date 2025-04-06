import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        self.obj_size = tuple()
        self.objects = dict()
        self.images = dict()
        self.annotations = dict()
        self.masks = dict()


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
            self: Class information

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
                self.images[file] = img

                xml_file = os.path.splitext(file)[0] + ".xml"
                xml_path = os.path.join(self.xml_dir, xml_file)

                # Save the annotations for images with objects
                if os.path.exists(xml_path):
                    boxes, class_labels = self.parse_voc_xml(xml_path)
                    self.annotations[file] = boxes

                else:
                    #self.annotations[file] = ([], [])
                    continue

        return self

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
            # Load image using OpenCV
            img = cv2.imread(os.path.join(self.obj_dir, file), cv2.IMREAD_COLOR)

            # Convert BGR to RGB
            obj = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Increase size of the objects by 1.5
            resized = cv2.resize(obj, (int(obj.shape[0] * 1.5), int(obj.shape[1] * 1.5)) , interpolation=cv2.INTER_LANCZOS4)
            self.objects[file] = resized

        return self

    def locate_anchor(self, dst_bbox):
        """
        Generates a new bounding box that is close to but does not overlap with the given bounding box.
        The new bounding box may have a different size.

        Args:
            dst_bbox (list): [x_min, y_min, x_max, y_max] representing the original bounding box.

        Returns:
            list: [x_min, y_min, x_max, y_max] for the new non-overlapping bounding box.
        """

        # Initialize minimum and maximum distance to move the new bounding box
        min_distance = 0.2 * self.obj_size[0]
        max_distance = 0.5 * self.obj_size[0]

        # Extract destination bounding box information
        x_min, y_min, x_max, y_max = dst_bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        for i in range(self.max_iter):
            # Compute diagonal distances (to ensure proper separation)
            orig_diag = np.sqrt(width ** 2 + height ** 2) / 2

            # Choose a random angle (0 to 360 degrees)
            angle = random.uniform(0, 2 * np.pi)

            # Ensure the distance is at least greater than the diagonal
            min_safe_distance = orig_diag + min_distance
            max_safe_distance = orig_diag + max_distance
            distance = random.uniform(min_safe_distance, max_safe_distance)

            # Then apply distance shift in target space
            new_center_x = center_x + distance * np.cos(angle)
            new_center_y = center_y + distance * np.sin(angle)

            # Compute new bounding box coordinates
            new_x_min = int(new_center_x - self.obj_size[0] / 2)
            new_y_min = int(new_center_y - self.obj_size[1] / 2)
            new_x_max = int(new_center_x + self.obj_size[0] / 2)
            new_y_max = int(new_center_y + self.obj_size[1] / 2)

            new_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]

            if min(new_bbox) > self.margin and max(new_bbox) < (self.target_size[0] - self.margin):
                final_bbox = new_bbox
                return final_bbox

        return None

    def find_bbox(self, anchor_bbox, existing_bboxes):
        """
        Tries to find a non-overlapping, valid anchor bounding box near `anchor_bbox`.
        Returns the new bbox if found, else None.
        """
        for _ in range(self.max_iter):
            new_bbox = self.locate_anchor(anchor_bbox)

            if new_bbox is None:
                print("Failed to find valid anchor — skipping to next bounding box.")
                return None

            if not self.check_overlap(existing_bboxes, new_bbox):
                return new_bbox

            print("Overlap detected — trying again...")

        # All attempts failed
        return None

    def select_candidate_object(self, target_bbox, exclude_key=None):
        """
        Selects an object from the pool that has a similar max dimension (width/height)
        to the target bounding box. Excludes a given object key (optional) and allows random sampling.

        Args:
            target_bbox (list): [x_min, y_min, x_max, y_max]
            exclude_key (str, optional): Key of the object to avoid selecting.

        Returns:
            (str, np.array): Tuple of selected key and object image array.
        """
        target_width = target_bbox[2] - target_bbox[0]
        target_height = target_bbox[3] - target_bbox[1]
        target_max_dim = max(target_width, target_height)

        # Filter objects that have a similar max dimension
        matching_objects = []
        for key, obj in self.objects.items():
            if key == exclude_key:
                continue  # Skip the object to exclude

            obj_h, obj_w = obj.shape[:2]
            obj_max_dim = max(obj_w, obj_h)

            # If the object's max dimension matches the target max dimension
            if obj_max_dim == target_max_dim:
                matching_objects.append(key)

        # If no matching object is found
        if not matching_objects:
            random_obj = np.random.choice(list(self.objects.keys()), replace=False)
            return self.objects[random_obj], self.masks[random_obj]

        # Randomly select an object from the matching set
        if self.sample_method == "without":
            # Without replacement: Randomly sample one object from the matching set without repeating
            selected_obj = matching_objects[np.random.choice(len(matching_objects), replace=False)]
        else:
            # With replacement: You can pick the same object multiple times
            selected_obj = matching_objects[np.random.choice(len(matching_objects))]

        return self.objects[selected_obj], self.masks[selected_obj]

    def create_masks(self):
        """
        Quickly create a binary mask from an image using thresholding and morphology.

        Args:
            self: Use the self.objects to access the objects and store them in self.masks.

        Returns:
            mask (np.ndarray): Binary mask (uint8, values 0 or 255).
        """
        for file, obj in self.objects.items():
            gray_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)

            # Auto threshold using Otsu's method
            _, binary_mask = cv2.threshold(gray_obj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations to clean the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            final_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Apply Gaussian blur to reduce the noise
            blur_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
            self.masks[file] = blur_mask

        return self

    @staticmethod
    def plot_elements(obj, mask, old_img, clone, center = None):
        """
        Visualizes the object, background, mask, and clone side by side using subplots.

        Args:
            obj (np.array): The object image to insert.
            old_img (np.array): The background image.
            clone (np.array): The result of the seamlessClone operation.
            center (tuple, optional): Center of the inserted object, printed for reference.
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Object Image
        axes[0].imshow(obj)
        axes[0].set_title("Object Image")
        axes[0].axis('off')

        # Mask
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask Result")
        axes[1].axis('off')

        # Background Image
        axes[2].imshow(old_img)
        axes[2].set_title("Background Image")
        axes[2].axis('off')

        # Clone
        axes[3].imshow(clone)
        axes[3].set_title("Clone Result")
        axes[3].axis('off')
        axes[3].scatter(center[0], center[1], color='red', s=10, marker='x')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def check_overlap(tabu_lst, candidate):
        """
        Checks if the candidate bounding box overlaps with any bounding box in the tabu list.

        Args:
            tabu_lst (list): List of bounding boxes, each in the format [x_min, y_min, x_max, y_max].
            candidate (list): The candidate bounding box in the format [x_min, y_min, x_max, y_max].

        Returns:
            bool: True if the candidate overlaps with any box in the tabu list, False otherwise.
        """

        # Skip the check when the tabu_list is empty
        if not tabu_lst:
            return False

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

    @staticmethod
    def check_insertion(image1, image2):
        """
        Compare two images using Structural Similarity Index (SSIM) to assess
        whether an object has been successfully inserted.

        Args:
            image1 (numpy.ndarray): The original image before insertion.
            image2 (numpy.ndarray): The modified image after insertion.

        Returns:
            float: The SSIM score (ranges from -1 to 1, where 1 means identical images).
        """
        # Convert images to grayscale
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_score = metrics.structural_similarity(image1_gray, image2_gray)
        print(f"SSIM Score: ", round(ssim_score, 3))
        return ssim_score

    def object_insertion(self):
        """
        The main function in the object insertion class. It uses the other functions to perform guided object insertion.
        It automatically saves the images to the provided output directory.
        """
        # Load the background images and annotations from the directories
        self.load_images()

        # Load the objects
        self.load_objects()

        # Create the object masks
        self.create_masks()

        # Start performing the object insertion
        print(f"Start insertion objects for {len(self.images)} images...")
        score_dct = dict()
        saves = 0
        fails = 0

        # Iterate of the background images
        for img in tqdm(self.annotations.keys()):
            dst_bbox = self.annotations[img]
            print(img, dst_bbox)

            # Create a list to store the new bounding boxes
            new_bbox_lst = list()

            # Store the old image for comparison
            old_img = self.images[img]

            # Some images have multiple bounding boxes
            print(f"Calculating new bounding box for {len(dst_bbox)} images...")
            for bbox in dst_bbox:

                # Sample an object from the set
                if self.sample_method == "random":
                    # Without replacement
                    obj = np.random.choice(list(self.objects.keys()), replace=False)
                    mask = self.masks[obj]
                    obj = self.objects[obj]
                else:
                    obj, mask = self.select_candidate_object(bbox)

                # Store the shape of the object image
                self.obj_size = obj.shape

                print(f"Finding new bounding box...")
                # Calculate a new bbox for insertion
                new_bbox = self.find_bbox(bbox, new_bbox_lst)
                print(bbox, new_bbox)

                # Skip to next bbox if we find a no valid one
                if new_bbox is None:
                    continue

                # Add the bounding box to the list
                new_bbox_lst.append(new_bbox)

                # Calculate the center of the candidate bounding box
                center = int((new_bbox[0] + new_bbox[2]) / 2), int((new_bbox[1] + new_bbox[3]) / 2)

                # Iteratively create a normal clone of the image with an inserted object
                clone = cv2.seamlessClone(
                    obj, old_img, mask, center,
                    flags=cv2.NORMAL_CLONE,
                )

                # Check if the new image has an additional object
                ssim_score = self.check_insertion(old_img, clone)
                score_dct[img] = ssim_score
                if ssim_score == 1.0:
                    print(f"Object likely did not insert.")
                    fails += 1

                # Display the images and mask
                print("Displaying the object, mask, image, and clone...")
                self.plot_elements(obj, mask, old_img, clone, center)

                # Overwrite the current image with the clone
                old_img = clone

                # Save object inserted images to directory
                #self.save_data(img, f"{img}_{count}")
                saves += 1


# Define the directories here:
root_dir = "E:/Datasets/"
img_dir = root_dir + "masati-thesis/images"
obj_dir = root_dir + "MasatiV2/MasatiV2Boats"
xml_dir = root_dir + "masati-thesis/annotations"
output_dir = root_dir + "masati-thesis/"

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
    sample_method="random"
)
augmenter.object_insertion()














