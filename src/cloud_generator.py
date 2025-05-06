import cv2
import satellite_cloud_generator as scg
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from tqdm import tqdm


class CloudGenerator:
    """
    CloudGenerator simplifies the process of generating clouds over satellite images
    and provides visualization utilities.
    """

    def __init__(self, min_lvl=0, max_lvl=0.8, locality_degree=2, blur_scaling=0, channel_offset=0,
                 shadow_max_lvl=0.25, cloud_probability=0.5):
        """
        Initialize the cloud generator with default cloud generation parameters.

        Args:
            min_lvl (float): Minimum cloud intensity level.
            max_lvl (float): Maximum cloud intensity level.
            locality_degree (int): Degree of locality for clouds.
            blur_scaling (float): Gaussian blur scaling for cloud mask.
            channel_offset (float): Channel offset for cloud overlay.
            shadow_max_lvl (float): Maximum intensity level for shadow mask.
            cloud_probability (float): Probability of generating clouds for an image (0.0-1.0).

        """
        self.min_lvl = min_lvl
        self.max_lvl = max_lvl
        self.locality_degree = locality_degree
        self.blur_scaling = blur_scaling
        self.channel_offset = channel_offset
        self.shadow_max_lvl = shadow_max_lvl
        self.cloud_probability = cloud_probability

    def generate_clouds(self, img):
        """
        Generate clouds over a single RGB satellite image.

        Args:
            img: A single image (e.g., a NumPy array or PIL image).

        Returns:
            tuple: (clouded_img, cloud_mask, shadow_mask) or None if no cloud was added.
        """
        # Convert image to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Convert to PyTorch tensor and rearrange dimensions (H, W, C) → (C, H, W)
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)

        # Determine if clouds should be added based on probability
        if np.random.random() < self.cloud_probability:
            # Add clouds and shadows using the satellite_cloud_generator library
            cl, cmask, smask = scg.add_cloud_and_shadow(
                img_tensor,
                min_lvl=self.min_lvl,
                max_lvl=self.max_lvl,
                locality_degree=self.locality_degree,
                cloud_color=False,
                channel_offset=self.channel_offset,
                blur_scaling=self.blur_scaling,
                return_cloud=True,
                const_scale=True,
                noise_type='perlin',
                shadow_max_lvl=self.shadow_max_lvl
            )
            return cl.squeeze(0).permute(1,2,0).detach().cpu().numpy(), cmask, smask
        else:
            # No clouds - return None or original with empty masks, based on your design preference
            return None

    def plot_results(self, dataset, results):
        """
        Plot satellite images with generated clouds, cloud masks, and shadow masks.

        Args:
            dataset (list): Original dataset of satellite images.
            results (list): Results from `generate_clouds()` containing
                            tuples of (clouded_img, cloud_mask, shadow_mask).
        """
        for i, (original_img, (clouded_img, cloud_mask, shadow_mask)) in enumerate(zip(dataset, results)):
            n_plots = 4 if shadow_mask is not None else 3
            plt.figure(figsize=(16, 6))

            # Plot the original image
            plt.subplot(1, n_plots, 1)
            self._imshow(torch.FloatTensor(original_img).permute(2, 0, 1) / 255.0)
            plt.title('Input')

            # Plot clouded image
            plt.subplot(1, n_plots, 2)
            self._imshow(clouded_img)
            plt.title('Clouded Image')

            # Plot cloud mask
            plt.subplot(1, n_plots, 3)
            self._imshow(cloud_mask)
            plt.title('Cloud Mask')

            # Plot shadow mask (if applicable)
            if n_plots == 4:
                plt.subplot(1, n_plots, n_plots)
                self._imshow(shadow_mask)
                plt.title('Shadow Mask')

            plt.show()

    @staticmethod
    def save_data(img, original_filename, annotations, output_dir):
        """
        Save a single clouded image and its corresponding annotation.

        Args:
            img (np.array): The clouded image.
            original_filename (str): The original image filename (e.g., "image1.png").
            output_dir (str): Path where the clouded image and annotation will be saved.
            copy_dir (str): Path to the original annotation files.

        Returns:
            self
        """

        # Define output subdirectories
        images_dir = os.path.join(output_dir, "3_cloud_generation")
        annotations_dir = os.path.join(output_dir, "3_cloud_annotation")

        # Ensure directories exist
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        # Strip extension and build new filenames
        base, _ = os.path.splitext(original_filename)

        # Save clouded image
        image_filename = base + "_cloud.png"
        image_path = os.path.join(images_dir, image_filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imsave(image_path, img_rgb)

        # Save annotation as JSON
        annotation_filename = base + "_cloud.json"
        annotation_path = os.path.join(annotations_dir, annotation_filename)
        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

    @staticmethod
    def _imshow(tensor, *args, **kwargs):
        """
        Simplified reusable function for plotting images from tensors.
        """
        plt.rcParams["figure.figsize"] = (12, 8)
        while len(tensor.shape) > 3:
            tensor = tensor[0]  # Remove extra batch dimensions
        plt.imshow(tensor.permute(1, 2, 0).detach().cpu(), *args, **kwargs)
        plt.axis('off')

if __name__ == "__main__":
    import os
    from PIL import Image
    import numpy as np


    def load_images_from_folder(folder_path):
        images = []
        for file_name in tqdm(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            try:
                img = Image.open(file_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
        return images


    cloud_generator = CloudGenerator()

    # Folder path with images
    folder_path = "E:\Datasets\masati-thesis\synthetic_images/2_augmentation/c0001_aug0.png"
    dataset = cv2.imread(folder_path)

    # Generate clouds
    results = cloud_generator.generate_clouds(dataset)
    print(results)
    plt.imshow(results)
    print("Cloud generation completed.")
