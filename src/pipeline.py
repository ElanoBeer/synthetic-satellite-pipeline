import os
import shutil
from object_insertion import ObjectInsertion
from basic_augmentation import BasicAugmentation
from cloud_generator import CloudGenerator


class ImageGenerationPipeline:
    """
    A pipeline that generates synthetic satellite images by sequentially applying:
    1. Object insertion
    2. Basic augmentation
    3. Cloud generation
    """

    def __init__(self,
                 # Base paths
                 root_dir,
                 output_dir,
                 seed,

                 # Object Insertion parameters
                 img_dir=None,
                 obj_dir=None,
                 xml_dir=None,
                 input_size=(512, 512),
                 target_size=(512, 512),
                 max_iter=1000,
                 margin=10,
                 max_insert=3,
                 sample_method='selective',
                 replacement=True,

                 # Basic Augmentation parameters
                 aug_probability=0.5,
                 n_augmentations=7,

                 # Cloud Generator parameters
                 min_cloud_intensity=0,
                 max_cloud_intensity=0.8,
                 locality_degree=2,
                 blur_scaling=0,
                 channel_offset=0,
                 max_shadow_intensity=0.25):
        """
        Initialize the pipeline with parameters for all three components.

        Args:
            root_dir: Root directory for input data
            output_dir: Directory to save final outputs
            seed: Random seed for reproducibility

            # Object Insertion parameters
            img_dir: Directory containing background images
            obj_dir: Directory containing objects to insert
            xml_dir: Directory containing XML annotations
            input_size: Size of input images
            target_size: Target size for output images
            max_iter: Maximum iterations for insertion attempts
            margin: Margin to maintain from image edges
            max_insert: Maximum number of objects to insert
            sample_method: Method to sample locations ('random', 'grid', etc.)
            replacement: Whether to allow object reuse

            # Basic Augmentation parameters
            aug_probability: Probability of applying each augmentation
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            noise_range: Range for noise addition
            blur_range: Range for blur effect

            # Cloud Generator parameters
            cloud_density: Density of cloud cover
            cloud_opacity: Range of cloud opacity
            shadow_intensity: Intensity of cloud shadows
        """
        # Store all parameters
        self.params = {
            "root_dir": root_dir,
            "output_dir": output_dir,
            "seed": seed,

            # Object Insertion
            "img_dir": img_dir or os.path.join(root_dir, "images"),
            "obj_dir": obj_dir or os.path.join(root_dir, "objects"),
            "xml_dir": xml_dir or os.path.join(root_dir, "annotations"),
            "input_size": input_size,
            "target_size": target_size,
            "max_iter": max_iter,
            "margin": margin,
            "max_insert": max_insert,
            "sample_method": sample_method,
            "replacement": replacement,

            # Basic Augmentation
            "aug_probability": aug_probability,
            "n_augmentations": n_augmentations,

            # Cloud Generator
            "min_cloud_intensity": min_cloud_intensity,
            "max_cloud_intensity": max_cloud_intensity,
            "locality_degree": locality_degree,
            "blur_scaling": blur_scaling,
            "channel_offset": channel_offset,
            "max_shadow_intensity": max_shadow_intensity,
        }

        # Initialize component directories
        self.obj_insertion_output = os.path.join(output_dir, "1_object_insertion")
        self.obj_annotation_output = os.path.join(output_dir, "1_object_annotation")
        self.augmentation_output = os.path.join(output_dir, "2_augmentation")
        self.final_output = os.path.join(output_dir, "3_final_images")

        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.obj_insertion_output, exist_ok=True)
        os.makedirs(self.augmentation_output, exist_ok=True)
        os.makedirs(self.final_output, exist_ok=True)

        # Initialize components (will be created during run)
        self.inserter = None
        self.augmenter = None
        self.cloud_gen = None

    def run(self):
        """
        Run the complete pipeline:
        1. Object Insertion
        2. Basic Augmentation
        3. Cloud Generation
        """
        print("Starting synthetic image generation pipeline...")

        # Step 1: Object Insertion
        print("\n--- Step 1: Object Insertion ---")
        self.inserter = ObjectInsertion(
            img_dir=self.params["img_dir"],
            obj_dir=self.params["obj_dir"],
            xml_dir=self.params["xml_dir"],
            out_dir=self.obj_insertion_output,
            input_size=self.params["input_size"],
            target_size=self.params["target_size"],
            max_iter=self.params["max_iter"],
            margin=self.params["margin"],
            max_insert=self.params["max_insert"],
            sample_method=self.params["sample_method"],
            replacement=self.params["replacement"],
            seed=self.params["seed"]
        )
        self.inserter.object_insertion()
        print(f"Object insertion complete. Images saved to {self.obj_insertion_output}")

        # Step 2: Basic Augmentation
        print("\n--- Step 2: Basic Augmentation ---")
        self.augmenter = BasicAugmentation(
            img_dir=self.obj_insertion_output,
            json_dir=self.obj_annotation_output,
            output_dir=self.augmentation_output,
            input_size=self.params["input_size"],
            target_size=self.params["target_size"],
            p=self.params["aug_probability"],
            n_augmentations=self.params["n_augmentations"]
        )
        self.augmenter.augment()
        print(f"Augmentation complete. Images saved to {self.augmentation_output}")

        # Step 3: Cloud Generation
        print("\n--- Step 3: Cloud Generation ---")
        self.cloud_gen = CloudGenerator(
            min_lvl=self.params["min_cloud_intensity"],
            max_lvl=self.params["max_cloud_intensity"],
            locality_degree=self.params["locality_degree"],
            blur_scaling=self.params["blur_scaling"],
            channel_offset=self.params["channel_offset"],
            shadow_max_lvl=self.params["max_shadow_intensity"],
        )

        # Apply cloud generation to each image
        image_files = [f for f in os.listdir(self.augmentation_output)
                       if f.endswith(('.jpg', '.png', '.jpeg'))]

        # add preprocess here

        self.cloud_gen.generate_clouds()

        print(f"Cloud generation complete. Final images saved to {self.final_output}")

        # Print parameters summary for reference
        self.print_parameters()

        return self.final_output

    def print_parameters(self):
        """Print all parameters used in the pipeline for reference"""
        print("\n=== Pipeline Parameters ===")

        print("\nObject Insertion Parameters:")
        for key in ["img_dir", "obj_dir", "xml_dir", "input_size", "target_size",
                    "max_iter", "margin", "max_insert", "sample_method", "replacement"]:
            print(f"  - {key}: {self.params[key]}")

        print("\nAugmentation Parameters:")
        for key in ["aug_probability", "brightness_range", "contrast_range",
                    "noise_range", "blur_range"]:
            print(f"  - {key}: {self.params[key]}")

        print("\nCloud Generation Parameters:")
        for key in ["cloud_density", "cloud_opacity", "shadow_intensity", "seed"]:
            print(f"  - {key}: {self.params[key]}")


# Example usage:
if __name__ == "__main__":
    pipeline = ImageGenerationPipeline(
        root_dir="data",
        output_dir="output/synthetic_images",
        # You can override any default parameters here
        max_insert=3,
        cloud_density=0.4,
        seed=42
    )

    output_path = pipeline.run()
    print(f"Pipeline complete! Generated images are in: {output_path}")