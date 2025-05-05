import os
import json
import albumentations as A
import cv2
from tqdm import tqdm


class BasicAugmentation:
    def __init__(self, img_dir, json_dir, output_dir, input_size, target_size, p=0.5, n_augmentations=1):
        """
        Initialize the basic augmentation pipeline.
        This pipeline utilizes albumentations to perform geometric and
        color space transformations.

        Args:
            p (float): Probability of applying each augmentation
        """

        # Define the input variable to use
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.output_dir = output_dir
        self.input_size = input_size
        self.target_size = target_size
        self.p = p
        self.n_augmentations = n_augmentations

        # Define information to store
        self.images = dict()
        self.annotations = dict()
        self.dataset = {
            "images": {},
            "annotations": {},
        }

        # Use the albumentations library for basic transformations
        self.transform = A.Compose(transforms=[
            A.HorizontalFlip(p=p),
            A.RandomRotate90(p=p),
            A.VerticalFlip(p=p),
            A.RandomCrop(height=self.target_size[0], width=self.target_size[1], p=p),
            A.RandomBrightnessContrast(p=p),
            A.HueSaturationValue(hue_shift_limit=10,       # default is 20
                                sat_shift_limit=15,       # default is 30
                                val_shift_limit=10,       # default is 20
                                p=p),
            A.GaussNoise(std_range=(0.1, 0.25), mean_range=(0,0), p=p)
        ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=["class_labels"]))

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
            # Assume sample_image is a NumPy array (cv2 format, BGR)
            success, encoded_image = cv2.imencode('.png', sample_image)
            if success:
                original_image_size_mb = len(encoded_image) / (1024 * 1024)
            else:
                raise ValueError("Failed to encode sample image for size estimation.")

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
        # Load image using OpenCV
        img = cv2.imread(img_path)

        # Resize image
        resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)

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
                img_path = os.path.join(self.img_dir, file)
                img = self.preprocess(img_path)
                self.images[file] = img

                json_file = os.path.splitext(file)[0] + ".json"
                json_path = os.path.join(self.json_dir, json_file)

                # Save the annotations for images with objects
                if os.path.exists(json_path):
                    boxes = self.parse_annotation_json(json_path)
                    self.annotations[file] = boxes
                else:
                    self.annotations[file] = []

        return self

    @staticmethod
    def parse_annotation_json(json_path):
        """
        Parse JSON annotation file to extract bounding boxes and class labels.

        Args:
            json_path (str): Path to JSON annotation file

        Returns:
            tuple: (list of bounding boxes, list of class labels)
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        boxes = data.get("boxes", [])
        #classes = data.get("classes", [])

        # Optional: Check that boxes and classes match up
        # if len(classes) != 0 and len(boxes) != len(classes):
        #     raise ValueError(f"Mismatch: {len(boxes)} boxes vs {len(classes)} classes in {json_path}")

        return boxes

    def save_data(self, img, augmented_img, new_boxes, img_id):
        """
        Save the images to a new directory.

        Args:
            self: class information
            img: Image file name
            augmented_img (np.array): The augmented image as a numpy array
            classes: List of class labels
            image_id: Unique image ID representing the index

        Returns:
            None: directory
        """

        # Define subdirectories for images and annotations
        images_dir = os.path.join(self.output_dir, "2_augmentation")
        annotations_dir = os.path.join(self.output_dir, "2_augmentation_annotation")

        # Create directories if they do not exist
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        # Define file name
        base, _ = os.path.splitext(img)
        image_filename = base + "_aug" + f"{img_id}.png"
        annotation_filename = base + "_aug" + f"{img_id}.json"

        # Save image
        image_path = os.path.join(images_dir, image_filename)
        augmented_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
        self.dataset["images"][image_filename] = augmented_rgb
        cv2.imwrite(image_path, augmented_rgb)

        # Save annotation as JSON
        annotation_data = {"boxes": new_boxes}
        annotation_path = os.path.join(annotations_dir, annotation_filename)
        self.dataset["annotations"][image_filename] = annotation_data
        with open(annotation_path, "w") as f:
            json.dump(annotation_data, f)

        return self

    def augment(self, load = True):
        """
        Apply augmentations to an image.

        Args:
            load (bool): Flag to load the data from the directories if not already loaded.

        Returns:
            PIL.Image: Augmented image
        """

        # Load the images and annotations from the directories
        if load:
            self.load_data()

        # Calculate final dataset size
        print(f"Calculating augmented dataset size...")
        print(f"{self.calculate_size(augmentations_per_image=7, sample_image=list(self.images.values())[0])}")

        # Start performing the basic augmentation
        print(f"Start augmentation for {len(self.images)} images...")

        # Convert PIL images to numpy arrays
        for img in tqdm(self.images.keys()):

            # Retrieve bounding boxes and class labels using the filename
            boxes = self.annotations[img][0]
            class_labels = ["ship" for _ in boxes]

            for i in range(self.n_augmentations):
                # Apply augmentations
                augmented = self.transform(
                    image=self.images[img],
                    bboxes=boxes,
                    class_labels=class_labels)

                # Convert back to PIL image
                augmented_image = augmented['image']
                augmented_box = augmented['bboxes']

                # Save augmented items to directory
                self.save_data(img, augmented_image, augmented_box, i)

        return self

if __name__ == "__main__":

    # Define the directories here:
    root_dir = "E:/Datasets/"
    img_dir = root_dir + "masati-thesis/clones"
    json_dir = root_dir + "masati-thesis/clone_annotations"
    output_dir = root_dir + "masati-thesis/"

    # Create a BasicAugmentation instance
    augmenter = BasicAugmentation(
        img_dir=img_dir,
        json_dir=json_dir,
        output_dir=output_dir,
        input_size=(512, 512),
        target_size=(512, 512),
        p=0.5,
        n_augmentations=7,
    )
    augmenter.augment()
