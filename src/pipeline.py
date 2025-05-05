import os
from tqdm import tqdm
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
                 max_shadow_intensity=0.25,
                 cloud_probability=0.5):
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
            n_augmentations: Number of augmentations to apply to each image

            # Cloud Generator parameters
            min_cloud_intensity: Minimum cloud intensity
            max_cloud_intensity: Maximum cloud intensity
            locality_degree: Degree of locality for cloud generation
            blur_scaling: Scaling factor for blurring
            channel_offset: Offset for cloud generation
            max_shadow_intensity: Maximum shadow intensity for cloud generation
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
            "huesat_values":{"hue_shift_limit": 10, "sat_shift_limit": 15, "val_shift_limit": 10},
            "gauss_noise_std": (0.1, 0.25),
            "gauss_noise_mean": (0,0),

            # Cloud Generator
            "min_cloud_intensity": min_cloud_intensity,
            "max_cloud_intensity": max_cloud_intensity,
            "locality_degree": locality_degree,
            "blur_scaling": blur_scaling,
            "channel_offset": channel_offset,
            "max_shadow_intensity": max_shadow_intensity,
            "cloud_probability": cloud_probability
        }

        # Initialize component directories

        self.original_images_output = os.path.join(output_dir, "0_original_images")
        self.original_annotation_output = os.path.join(output_dir, "0_original_annotation")
        self.obj_insertion_output = os.path.join(output_dir, "1_object_insertion")
        self.obj_annotation_output = os.path.join(output_dir, "1_object_annotation")
        self.augmentation_output = os.path.join(output_dir, "2_augmentation")
        self.augmentation_annotation_output = os.path.join(output_dir, "2_augmentation_annotation")
        self.cloud_generation_output = os.path.join(output_dir, "3_cloud_generation")
        self.cloud_annotation_output = os.path.join(output_dir, "3_cloud_annotation")

        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.original_images_output, exist_ok=True)
        os.makedirs(self.original_annotation_output, exist_ok=True)
        os.makedirs(self.obj_insertion_output, exist_ok=True)
        os.makedirs(self.obj_annotation_output, exist_ok=True)
        os.makedirs(self.augmentation_output, exist_ok=True)
        os.makedirs(self.augmentation_annotation_output, exist_ok=True)
        os.makedirs(self.cloud_generation_output, exist_ok=True)
        os.makedirs(self.cloud_annotation_output, exist_ok=True)

        # Initialize components (will be created during run)
        self.inserter = None
        self.original_augmenter = None
        self.insertion_augmenter = None
        self.augmenter = None
        self.cloud_gen = None
        self.tracker = None

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
            out_dir=self.params["output_dir"],
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
        self.tracker = self.inserter.dataset

        # Step 2: Basic Augmentation
        print("\n--- Step 2: Basic Augmentation ---")
        self.augmenter = BasicAugmentation(
            img_dir=self.original_images_output,
            json_dir=self.original_annotation_output,
            output_dir=self.params["output_dir"],
            input_size=self.params["input_size"],
            target_size=self.params["target_size"],
            p=self.params["aug_probability"],
            n_augmentations=self.params["n_augmentations"]
        )

        self.augmenter.images = self.tracker["images"]
        self.augmenter.annotations = self.tracker["annotations"]
        print(f"Length of annotations: {len(self.augmenter.annotations)}")
        self.augmenter.augment(load=False)

        self.tracker["images"].update(self.augmenter.dataset["images"])
        self.tracker["annotations"].update(self.augmenter.dataset["annotations"])

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
            cloud_probability=self.params["cloud_probability"],
        )

        # Process and save images one by one with their filenames
        for filename, image in tqdm(self.tracker["images"].items(), desc="Processing images"):
            result = self.cloud_gen.generate_clouds(image)
            annotations = self.tracker["annotations"][filename]

            if result is not None:
                cl, _, _ = result  # Clouded image, cloud mask, shadow mask
                self.cloud_gen.save_data(cl, filename, annotations, self.params["output_dir"])

        print(f"Cloud generation complete. Final images saved to {self.cloud_generation_output}")

        # Print parameters summary for reference
        self.print_parameters(save_path=os.path.join(self.params["output_dir"], "parameters.txt"))

        return self.cloud_generation_output


    def print_parameters(self, save_path=None):
        """Print all parameters used in the pipeline and optionally save to a text file"""

        lines = list()
        lines.append("\n=== Pipeline Parameters ===\n")
        lines.append(f"Random Seed: {self.params['seed']}\n")

        lines.append("\nObject Insertion Parameters:")
        for key in ["img_dir", "obj_dir", "xml_dir", "input_size", "target_size",
                    "max_iter", "margin", "max_insert", "sample_method", "replacement"]:
            lines.append(f"  - {key}: {self.params[key]}")

        lines.append("\nAugmentation Parameters:")
        for key in ["aug_probability", "n_augmentations", "huesat_values", "gauss_noise_std", "gauss_noise_mean"]:
            lines.append(f"  - {key}: {self.params[key]}")

        lines.append("\nCloud Generation Parameters:")
        for key in ["min_cloud_intensity", "max_cloud_intensity", "locality_degree", "blur_scaling", "channel_offset",
                    "max_shadow_intensity", "cloud_probability"]:
            lines.append(f"  - {key}: {self.params[key]}")

        # Print to console
        print("\n".join(lines))

        # Optionally write to a file
        if save_path:
            with open(save_path, "w") as f:
                f.write("\n".join(lines))
            print(f"\nParameter log saved to: {save_path}")


# Example usage:
if __name__ == "__main__":
    pipeline = ImageGenerationPipeline(
        root_dir="E:/Datasets/masati-thesis",
        img_dir="E:/Datasets/masati-thesis/images",
        obj_dir="E:/Datasets/MasatiV2/MasatiV2Boats",
        xml_dir="E:/Datasets/masati-thesis/annotations",
        output_dir="E:/Datasets/masati-thesis/synthetic_images",
        seed=89,
        input_size=(512, 512),
        target_size=(256, 256),
        max_iter=100,
        margin=10,
        max_insert=1,
        sample_method='selective',
        replacement=True,

        # Basic Augmentation parameters
        aug_probability=0.5,
        n_augmentations=1,

        # Cloud Generator parameters
        min_cloud_intensity=0,
        max_cloud_intensity=0.8,
        locality_degree=2,
        blur_scaling=0,
        channel_offset=0,
        max_shadow_intensity=0.25,
        cloud_probability=0.2,
    )

    output_path = pipeline.run()
    print(f"Pipeline complete! Generated images are in: {output_path}")