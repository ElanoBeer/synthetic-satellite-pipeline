import os
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCreator:
    """
    A class to create different dataset configurations for experimentation.

    Creates four types of datasets:
    1. Original images only
    2. Non-original images only
    3. Random mix of original and non-original images
    4. Mix with specific percentage of VAE-generated images
    """

    def __init__(self, original_dir, secondary_dir, vae_dir, output_base_dir):
        """
        Initialize the DatasetCreator with directory paths.

        Args:
            original_dir (str): Path to directory containing original images
            secondary_dir (str): Path to directory containing secondary/alternative images
            vae_dir (str): Path to directory containing VAE-generated images
            output_base_dir (str): Base directory for output datasets
        """
        self.original_dir = Path(original_dir)
        self.secondary_dir = Path(secondary_dir)
        self.vae_dir = Path(vae_dir)
        self.output_base_dir = Path(output_base_dir)

        # Validate directories
        self._validate_directories()

        # Cache for filenames
        self._original_files = None
        self._non_original_files = None
        self._vae_files = None

    def _validate_directories(self):
        """Validate that input directories exist."""
        for dir_path, dir_name in [
            (self.original_dir, "Original"),
            (self.secondary_dir, "Secondary"),
            (self.vae_dir, "VAE")
        ]:
            if not dir_path.exists():
                raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")

        # Create output base directory if it doesn't exist
        if not self.output_base_dir.exists():
            logger.info(f"Creating output directory: {self.output_base_dir}")
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def get_original_files(self):
        """
        Get list of image filenames from the original directory.

        Returns:
            list: List of image filenames
        """
        if self._original_files is None:
            self._original_files = [
                f.name for f in self.original_dir.glob("*")
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            ]
            logger.info(f"Found {len(self._original_files)} original images")
        return self._original_files

    def get_non_original_files(self):
        """
        Get list of image filenames from secondary directory that are not in original directory.

        Returns:
            list: List of image filenames
        """
        if self._non_original_files is None:
            orig_files = set(self.get_original_files())
            self._non_original_files = [
                f.name for f in self.secondary_dir.glob("*")
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
                   and f.name not in orig_files
            ]
            logger.info(f"Found {len(self._non_original_files)} non-original images")
        return self._non_original_files

    def get_vae_files(self):
        """
        Get list of VAE-generated image filenames.

        Returns:
            list: List of image filenames
        """
        if self._vae_files is None:
            self._vae_files = [
                f.name for f in self.vae_dir.glob("*")
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            ]
            logger.info(f"Found {len(self._vae_files)} VAE-generated images")
        return self._vae_files

    def create_dataset_1(self, output_dir=None):
        """
        Create dataset 1: Images from original directory only.

        Args:
            output_dir (str, optional): Output directory. If None, uses 'dataset1' in base dir.

        Returns:
            Path: Path to created dataset directory
        """
        if output_dir is None:
            output_dir = self.output_base_dir / "dataset1"
        else:
            output_dir = Path(output_dir)

        orig_files = self.get_original_files()

        logger.info(f"Creating Dataset 1 with {len(orig_files)} original files")
        source_dir = self.original_dir
        self._copy_files_to_dataset(orig_files, source_dir, output_dir)

        return output_dir

    def create_dataset_2(self, output_dir=None):
        """
        Create dataset 2: Images not in original directory.

        Args:
            output_dir (str, optional): Output directory. If None, uses 'dataset2' in base dir.

        Returns:
            Path: Path to created dataset directory
        """
        if output_dir is None:
            output_dir = self.output_base_dir / "dataset2"
        else:
            output_dir = Path(output_dir)

        non_orig_files = self.get_non_original_files()

        logger.info(f"Creating Dataset 2 with {len(non_orig_files)} non-original files")
        source_dir = self.secondary_dir
        self._copy_files_to_dataset(non_orig_files, source_dir, output_dir)

        return output_dir

    def create_dataset_3(self, output_dir=None, ratio=0.5, total_images=None):
        """
        Create dataset 3: Random mix of original and non-original images.

        Args:
            output_dir (str, optional): Output directory. If None, uses 'dataset3' in base dir.
            ratio (float): Ratio of original to non-original images (0.5 means 50% original)
            total_images (int, optional): Total number of images in the dataset.
                                         If None, uses all available images.

        Returns:
            Path: Path to created dataset directory
        """
        if output_dir is None:
            output_dir = self.output_base_dir / "dataset3"
        else:
            output_dir = Path(output_dir)

        orig_files = self.get_original_files()
        non_orig_files = self.get_non_original_files()

        # Calculate numbers
        if total_images is None:
            num_orig = int(len(orig_files) * ratio)
            num_non_orig = len(non_orig_files)
        else:
            num_orig = int(total_images * ratio)
            num_non_orig = total_images - num_orig

            # Cap at available images
            num_orig = min(num_orig, len(orig_files))
            num_non_orig = min(num_non_orig, len(non_orig_files))

        # Sample images
        sampled_orig = random.sample(orig_files, num_orig)
        sampled_non_orig = random.sample(non_orig_files, num_non_orig)

        logger.info(f"Creating Dataset 3 with {num_orig} original images and {num_non_orig} non-original images")

        # Copy files
        self._copy_files_to_dataset(sampled_orig, self.original_dir, output_dir)
        self._copy_files_to_dataset(sampled_non_orig, self.secondary_dir, output_dir)

        return output_dir

    def create_dataset_4(self, output_dir=None, vae_percentage=0.25, orig_ratio=0.5, total_images=None):
        """
        Create dataset 4: Mix with specific percentage of VAE-generated images.

        Args:
            output_dir (str, optional): Output directory. If None, uses 'dataset4' in base dir.
            vae_percentage (float): Percentage of images that should be VAE-generated (0.25 = 25%)
            orig_ratio (float): Ratio of original to non-original images in the remaining
                               non-VAE portion (0.5 means 50% original, 50% non-original)
            total_images (int, optional): Total number of images in the dataset.
                                         If None, uses all available images.

        Returns:
            Path: Path to created dataset directory
        """
        if output_dir is None:
            output_dir = self.output_base_dir / "dataset4"
        else:
            output_dir = Path(output_dir)

        orig_files = self.get_original_files()
        non_orig_files = self.get_non_original_files()
        vae_files = self.get_vae_files()

        # Calculate numbers
        if total_images is None:
            num_vae = min(int(len(vae_files)),
                          int((len(orig_files) + len(non_orig_files)) * vae_percentage / (1 - vae_percentage)))
            num_non_vae = len(orig_files) + len(non_orig_files)
        else:
            num_vae = int(total_images * vae_percentage)
            num_non_vae = total_images - num_vae

            # Cap at available images
            num_vae = min(num_vae, len(vae_files))

        # For non-VAE images, split between original and non-original
        num_orig = int(num_non_vae * orig_ratio)
        num_non_orig = num_non_vae - num_orig

        # Cap at available images
        num_orig = min(num_orig, len(orig_files))
        num_non_orig = min(num_non_orig, len(non_orig_files))

        # Sample images
        sampled_vae = random.sample(vae_files, num_vae)
        sampled_orig = random.sample(orig_files, num_orig)
        sampled_non_orig = random.sample(non_orig_files, num_non_orig)

        logger.info(
            f"Creating Dataset 4 with {num_vae} VAE images, {num_orig} original images, and {num_non_orig} non-original images")

        # Copy files
        self._copy_files_to_dataset(sampled_vae, self.vae_dir, output_dir)
        self._copy_files_to_dataset(sampled_orig, self.original_dir, output_dir)
        self._copy_files_to_dataset(sampled_non_orig, self.secondary_dir, output_dir)

        return output_dir

    def create_all_datasets(self, total_images=None):
        """
        Create all four datasets.

        Args:
            total_images (int, optional): Total number of images per dataset.
                                         If None, uses all available images.

        Returns:
            dict: Dictionary of dataset paths
        """
        datasets = {
            "dataset1": self.create_dataset_1(),
            "dataset2": self.create_dataset_2(),
            "dataset3": self.create_dataset_3(total_images=total_images),
            "dataset4": self.create_dataset_4(total_images=total_images)
        }

        logger.info("All datasets created successfully")
        return datasets

    def _copy_files_to_dataset(self, filenames, source_dir, target_dir):
        """
        Copy files from source directory to target directory.

        Args:
            filenames (list): List of filenames to copy
            source_dir (Path): Source directory
            target_dir (Path): Target directory
        """
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        for filename in filenames:
            source_file = source_dir / filename
            target_file = target_dir / filename

            if source_file.exists():
                shutil.copy2(source_file, target_file)
            else:
                logger.warning(f"Source file not found: {source_file}")

        logger.info(f"Copied {len(filenames)} files to {target_dir}")


def main():
    """Example usage of the DatasetCreator class."""

    # Example paths - replace with your actual paths
    original_dir = "path/to/original/images"
    secondary_dir = "path/to/secondary/images"
    vae_dir = "path/to/vae/generated/images"
    output_dir = "path/to/output/datasets"

    # Create DatasetCreator instance
    creator = DatasetCreator(
        original_dir=original_dir,
        secondary_dir=secondary_dir,
        vae_dir=vae_dir,
        output_base_dir=output_dir
    )

    # Create all datasets with 1000 images each
    datasets = creator.create_all_datasets(total_images=1000)

    # Or create individual datasets with custom parameters
    # dataset1 = creator.create_dataset_1()
    # dataset2 = creator.create_dataset_2()
    # dataset3 = creator.create_dataset_3(ratio=0.7, total_images=800)
    # dataset4 = creator.create_dataset_4(vae_percentage=0.3, orig_ratio=0.6, total_images=1200)

    print("Datasets created at:")
    for name, path in datasets.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()  # %%
