import os
import random
import shutil
from pathlib import Path
import numpy as np
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

    For each image, corresponding JSON annotation files will also be copied if available.
    """

    def __init__(self, original_dir, insertion_dir, augmentation_dir, cloud_dir, vae_dir,
                 output_base_dir, sample_weights=None, seed=None, copy_annotations=True):
        """
        Initialize the DatasetCreator with directory paths.

        Args:
            original_dir (str): Path to directory containing original images
            insertion_dir (str): Path to directory containing insertion images
            augmentation_dir (str): Path to directory containing augmentation images
            cloud_dir (str): Path to directory containing cloud-generated images
            vae_dir (str): Path to directory containing VAE-generated images
            output_base_dir (str): Base directory for output datasets
            sample_weights (list): Weights for sampling from non-original dirs [insertion, augmentation, cloud]
            seed (int): Random seed for reproducibility
            copy_annotations (bool): Whether to copy annotation files along with images
        """
        self.original_dir = Path(original_dir)
        self.insertion_dir = Path(insertion_dir)
        self.augmentation_dir = Path(augmentation_dir)
        self.cloud_dir = Path(cloud_dir)
        self.vae_dir = Path(vae_dir)
        self.output_base_dir = Path(output_base_dir)
        self.non_original_dirs = [self.insertion_dir, self.augmentation_dir, self.cloud_dir]
        self.copy_annotations = copy_annotations

        # Map image directories to their corresponding annotation directories
        root_dir = self.original_dir.parent
        self.annotation_dirs = {
            str(self.original_dir): root_dir / f"0_original_annotation",
            str(self.insertion_dir): root_dir / f"1_object_annotation",
            str(self.augmentation_dir): root_dir / f"2_augmentation_annotation",
            str(self.cloud_dir): root_dir / f"3_cloud_annotation_copy",
            str(self.vae_dir): root_dir / f"4_vae_annotation"
        }

        # Default weights if none provided
        if sample_weights is None:
            self.sample_weights = {
                str(self.insertion_dir): 0.6,
                str(self.augmentation_dir): 0.25,
                str(self.cloud_dir): 0.15
            }
        else:
            # Convert list to dictionary
            if isinstance(sample_weights, list) and len(sample_weights) == 3:
                self.sample_weights = {
                    str(self.insertion_dir): sample_weights[0],
                    str(self.augmentation_dir): sample_weights[1],
                    str(self.cloud_dir): sample_weights[2]
                }
            else:
                self.sample_weights = sample_weights

        # Set random seed
        self.set_seed(seed)

        # Validate directories
        self._validate_directories()

        # Cache for filenames
        self._original_files = None
        self._non_original_files = None
        self._vae_files = None

    def set_seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            logger.info(f"Random seed set to {seed}")
        return self

    def _validate_directories(self):
        """Validate that input directories exist."""
        for dir_path, dir_name in [
            (self.original_dir, "Original"),
            (self.insertion_dir, "Insertion"),
            (self.augmentation_dir, "Augmentation"),
            (self.cloud_dir, "Cloud"),
            (self.vae_dir, "VAE")
        ]:
            if not dir_path.exists():
                raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")

        # Check annotation directories if enabled
        if self.copy_annotations:
            for img_dir, anno_dir in self.annotation_dirs.items():
                if not anno_dir.exists():
                    logger.warning(
                        f"Annotation directory not found: {anno_dir}. Will skip copying annotations for {Path(img_dir).name}.")

        # Create output base directory if it doesn't exist
        if not self.output_base_dir.exists():
            logger.info(f"Creating output directory: {self.output_base_dir}")
            self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def get_image_files(self, directory):
        """Get list of image files from a directory."""
        return [
            f.name for f in directory.glob("*")
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        ]

    def get_original_files(self):
        """Get list of image filenames from the original directory."""
        if self._original_files is None:
            self._original_files = self.get_image_files(self.original_dir)
            logger.info(f"Found {len(self._original_files)} original images")
        return self._original_files

    def get_non_original_files(self):
        """Get list of image filenames from all non-original directories."""
        if self._non_original_files is None:
            orig_files = set(self.get_original_files())
            self._non_original_files = []

            for directory in self.non_original_dirs:
                files_in_dir = [
                    f for f in self.get_image_files(directory)
                    if f not in orig_files
                ]
                self._non_original_files.extend(files_in_dir)
                logger.info(f"Found {len(files_in_dir)} non-original images in {directory}")

            logger.info(f"Found total of {len(self._non_original_files)} non-original images")
        return self._non_original_files

    def get_vae_files(self):
        """Get list of VAE-generated image filenames."""
        if self._vae_files is None:
            self._vae_files = self.get_image_files(self.vae_dir)
            logger.info(f"Found {len(self._vae_files)} VAE-generated images")
        return self._vae_files

    def create_dataset_1(self, total_images=None):
        """
        Create dataset 1: Images from original directory only.

        Args:
            total_images (int, optional): Total number of images to include

        Returns:
            Path: Path to created dataset directory
        """
        # Set seed for reproducibility within this method
        if hasattr(self, 'seed'):
            self.set_seed(self.seed)

        # Create dataset directory
        dataset_dir = self.output_base_dir / "dataset1"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Get and sample original files
        orig_files = self.get_original_files()
        sampled_files = random.sample(orig_files, min(total_images, len(orig_files))) if total_images else orig_files

        logger.info(f"Creating Dataset 1 with {len(sampled_files)} original files")

        # Copy files to dataset directory
        self._copy_files_to_dataset(sampled_files, self.original_dir, dataset_dir)

        return dataset_dir

    def create_dataset_2(self, total_images=100):
        """
        Create dataset 2: Non-original images only.

        Args:
            total_images (int): Number of images to include

        Returns:
            Path: Path to created dataset directory
        """
        # Set seed for reproducibility within this method
        if hasattr(self, 'seed'):
            self.set_seed(self.seed)

        # Create dataset directory
        dataset_dir = self.output_base_dir / "dataset2"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Sample non-original files
        sampled_files = self._get_files_from_non_original_dirs(total_images)
        logger.info(f"Creating Dataset 2 with {len(sampled_files)} non-original files")

        # Copy files from each source directory
        for dir_path in self.non_original_dirs:
            # Get files from this directory that are in our sampled set
            dir_files = [f for f in sampled_files if (dir_path / f).exists()]
            if dir_files:
                self._copy_files_to_dataset(dir_files, dir_path, dataset_dir)

        return dataset_dir

    def create_dataset_3(self, ratio=0.5, total_images=None):
        """
        Create dataset 3: Random mix of original and non-original images.

        Args:
            ratio (float): Ratio of original to non-original images (0.5 means 50% original)
            total_images (int, optional): Total number of images in the dataset

        Returns:
            Path: Path to created dataset directory
        """
        # Set seed for reproducibility within this method
        if hasattr(self, 'seed'):
            self.set_seed(self.seed)

        # Create dataset directory
        dataset_dir = self.output_base_dir / "dataset3"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        orig_files = self.get_original_files()
        non_orig_files = self.get_non_original_files()

        # Calculate numbers of images to sample
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
        sampled_non_orig = self._get_files_from_non_original_dirs(num_non_orig)

        logger.info(f"Creating Dataset 3 with {num_orig} original images and {num_non_orig} non-original images")

        # Copy original files
        self._copy_files_to_dataset(sampled_orig, self.original_dir, dataset_dir)

        # Copy non-original files from each source directory
        for dir_path in self.non_original_dirs:
            dir_files = [f for f in sampled_non_orig if (dir_path / f).exists()]
            if dir_files:
                self._copy_files_to_dataset(dir_files, dir_path, dataset_dir)

        return dataset_dir

    def create_dataset_4(self, vae_percentage=0.25, orig_ratio=0.5, total_images=None):
        """
        Create dataset 4: Mix with specific percentage of VAE-generated images.

        Args:
            vae_percentage (float): Percentage of images that should be VAE-generated
            orig_ratio (float): Ratio of original to non-original images in the non-VAE portion
            total_images (int, optional): Total number of images in the dataset

        Returns:
            Path: Path to created dataset directory
        """
        # Set seed for reproducibility within this method
        if hasattr(self, 'seed'):
            self.set_seed(self.seed)

        # Create dataset directory
        dataset_dir = self.output_base_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        orig_files = self.get_original_files()
        non_orig_files = self.get_non_original_files()
        vae_files = self.get_vae_files()

        # Calculate number of images from each source
        if total_images is None:
            num_vae = min(len(vae_files),
                          int((len(orig_files) + len(non_orig_files)) * vae_percentage / (1 - vae_percentage)))
            num_non_vae = len(orig_files) + len(non_orig_files)
        else:
            num_vae = int(total_images * vae_percentage)
            num_non_vae = total_images - num_vae
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
        sampled_non_orig = self._get_files_from_non_original_dirs(num_non_orig)

        logger.info(
            f"Creating Dataset 4 with {num_vae} VAE images, {num_orig} original images, and {num_non_orig} non-original images"
        )

        # Copy files
        self._copy_files_to_dataset(sampled_vae, self.vae_dir, dataset_dir)
        self._copy_files_to_dataset(sampled_orig, self.original_dir, dataset_dir)

        # Copy non-original files from each source directory
        for dir_path in self.non_original_dirs:
            dir_files = [f for f in sampled_non_orig if (dir_path / f).exists()]
            if dir_files:
                self._copy_files_to_dataset(dir_files, dir_path, dataset_dir)

        return dataset_dir

    def _get_files_from_non_original_dirs(self, count):
        """
        Get a weighted random sample of files from non-original directories with no duplicates.

        Args:
            count (int): Number of files to sample

        Returns:
            list: List of sampled filenames with no duplicates
        """
        # Get all files from each directory
        files_by_dir = {
            str(self.insertion_dir): [],
            str(self.augmentation_dir): [],
            str(self.cloud_dir): []
        }

        # Collect all available files and map them to their source directories
        for file in self.get_non_original_files():
            for dir_path in files_by_dir.keys():
                if os.path.exists(os.path.join(dir_path, file)):
                    files_by_dir[dir_path].append(file)
                    break

        # Calculate target counts based on weights and total requested count
        target_counts = {}
        total_weight = sum(self.sample_weights.values())

        for dir_path, weight in self.sample_weights.items():
            # Calculate proportional count for this directory
            dir_count = int(count * (weight / total_weight))
            # Ensure we don't try to sample more files than available
            available_count = len(files_by_dir[dir_path])
            target_counts[dir_path] = min(dir_count, available_count)

        # Sample files from each directory according to weights
        sampled_files = []
        remaining_count = count

        # First pass: sample according to weights
        for dir_path, target_count in target_counts.items():
            if target_count > 0 and files_by_dir[dir_path]:
                samples = np.random.choice(
                    files_by_dir[dir_path],
                    size=min(target_count, remaining_count),
                    replace=False  # Ensure no duplicates within directory
                ).tolist()
                sampled_files.extend(samples)
                remaining_count -= len(samples)

                # Remove sampled files from other directories to prevent duplicates
                for other_dir in files_by_dir.values():
                    other_dir[:] = [f for f in other_dir if f not in samples]

        # Second pass: fill remaining count if needed
        if remaining_count > 0:
            # Collect all remaining files that haven't been sampled
            remaining_files = []
            for dir_files in files_by_dir.values():
                remaining_files.extend(dir_files)

            remaining_files = list(set(remaining_files) - set(sampled_files))

            if remaining_files:
                additional_samples = np.random.choice(
                    remaining_files,
                    size=min(remaining_count, len(remaining_files)),
                    replace=False
                ).tolist()
                sampled_files.extend(additional_samples)

        # Shuffle the final list to avoid any ordering bias
        np.random.shuffle(sampled_files)
        return sampled_files

    def create_all_datasets(self, total_images=None):
        """
        Create all four datasets with consistent seed for reproducibility.

        Args:
            total_images (int, optional): Total number of images per dataset

        Returns:
            dict: Dictionary of dataset paths
        """
        datasets = {
            "dataset1": self.create_dataset_1(total_images),
            "dataset2": self.create_dataset_2(total_images),
            "dataset3": self.create_dataset_3(ratio=0.5, total_images=total_images),
            "dataset4": self.create_dataset_4(vae_percentage=0.25, orig_ratio=0.5, total_images=total_images)
        }

        logger.info("All datasets created successfully")
        return datasets

    def _copy_files_to_dataset(self, filenames, source_dir, target_dir):
        """
        Copy files from source directory to target directory.
        Also copy corresponding annotation files if available.

        Args:
            filenames (list): List of filenames to copy
            source_dir (Path): Source directory
            target_dir (Path): Target directory
        """
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create annotation directory if needed
        if self.copy_annotations:
            target_anno_dir = target_dir / "annotations"
            target_anno_dir.mkdir(parents=True, exist_ok=True)

        # Copy files
        successful_copies = 0
        successful_anno_copies = 0

        for filename in filenames:
            # Copy image file
            source_file = source_dir / filename
            target_file = target_dir / filename

            if source_file.exists():
                shutil.copy2(source_file, target_file)
                successful_copies += 1
            else:
                logger.warning(f"Source file not found: {source_file}")
                continue

            # Copy annotation file if enabled
            if self.copy_annotations:
                # Get annotation directory for this source directory
                anno_dir = self.annotation_dirs.get(str(source_dir))
                if anno_dir and anno_dir.exists():
                    # Get base filename without extension and look for json file
                    base_name = Path(filename).stem
                    anno_file = anno_dir / f"{base_name}.json"

                    if anno_file.exists():
                        target_anno_file = target_anno_dir / f"{base_name}.json"
                        shutil.copy2(anno_file, target_anno_file)
                        successful_anno_copies += 1
                    else:
                        logger.debug(f"Annotation file not found: {anno_file}")

        logger.info(f"Copied {successful_copies} files from {source_dir.name} to {target_dir}")
        if self.copy_annotations:
            logger.info(f"Copied {successful_anno_copies} annotation files for {source_dir.name}")


def main():
    """Example usage of the DatasetCreator class."""

    # Example paths - replace with your actual paths
    root_dir = "E:/Datasets/masati-thesis/synthetic_images/"
    original_dir = root_dir + "0_original_images"
    insertion_dir = root_dir + "1_object_insertion"
    augmentation_dir = root_dir + "2_augmentation"
    cloud_dir = root_dir + "3_cloud_generation"
    vae_dir = root_dir + "4_vae_generation"
    output_dir = root_dir + "5_output_datasets"

    # Create DatasetCreator instance with seed for reproducibility
    creator = DatasetCreator(
        original_dir=original_dir,
        insertion_dir=insertion_dir,
        augmentation_dir=augmentation_dir,
        cloud_dir=cloud_dir,
        vae_dir=vae_dir,
        output_base_dir=output_dir,
        sample_weights=[0.6, 0.25, 0.15],
        seed=1583891,
        copy_annotations=True  # Enable copying of annotation files
    )

    # Create all datasets with 1000 images each
    # datasets = creator.create_all_datasets(total_images=1000)

    # Or create individual datasets with custom parameters
    # creator.create_dataset_1(total_images=4000)
    # creator.create_dataset_2(total_images=6000)
    # creator.create_dataset_3(ratio=0.25, total_images=8000)
    creator.create_dataset_4(vae_percentage=0.4, orig_ratio=0.2, total_images=10000)


if __name__ == "__main__":
    main()