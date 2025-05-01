import satellite_cloud_generator as scg
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class CloudGenerator:
    """
    CloudGenerator simplifies the process of generating clouds over satellite images
    and provides visualization utilities.
    """

    def __init__(self, min_lvl=0, max_lvl=0.8, locality_degree=2, blur_scaling=0, channel_offset=0,
                 shadow_max_lvl=0.25):
        """
        Initialize the cloud generator with default cloud generation parameters.

        Args:
            min_lvl (float): Minimum cloud intensity level.
            max_lvl (float): Maximum cloud intensity level.
            locality_degree (int): Degree of locality for clouds.
            blur_scaling (float): Gaussian blur scaling for cloud mask.
            channel_offset (float): Channel offset for cloud overlay.
            shadow_max_lvl (float): Maximum intensity level for shadow mask.
        """
        self.min_lvl = min_lvl
        self.max_lvl = max_lvl
        self.locality_degree = locality_degree
        self.blur_scaling = blur_scaling
        self.channel_offset = channel_offset
        self.shadow_max_lvl = shadow_max_lvl

    def generate_clouds(self, dataset):
        """
        Generate clouds over a list of RGB satellite images.

        Args:
            dataset (list): List or iterable of images (e.g., numpy arrays or PIL images).

        Returns:
            list: A list of tuples [(clouded_img, cloud_mask, shadow_mask), ...].
        """
        cloud_dataset = []

        # Iterate over the dataset and generate clouds per image
        for img in tqdm(dataset, desc="Generating clouds"):
            # Convert image to numpy array and normalize
            img_array = np.array(img) / 255.0

            # Convert to PyTorch tensor and rearrange dimensions (H, W, C) → (C, H, W)
            img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)

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
            cloud_dataset.append((cl, cmask, smask))

        return cloud_dataset

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
    cloud_generator = CloudGenerator()

    # Generate clouds for a sample dataset
    dataset = [...]  # Replace with your dataset (list of numpy arrays or PIL images)
    results = cloud_generator.generate_clouds(dataset)

    # Plot the results
    cloud_generator.plot_results(dataset, results)

