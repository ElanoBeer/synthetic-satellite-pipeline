import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from cloud_generator import CloudGenerator

# Ensure reproducibility
np.random.seed(1583891)
torch.manual_seed(1583891)

def process_image_with_cloud_generator(input_folder, mask_output_folder, cloud_generator):
    """
    Process images one by one from the input folder and generate cloud masks
    
    Args:
        input_folder (str): Path to the folder containing input images
        mask_output_folder (str): Path to the folder where masks will be saved
        cloud_generator (CloudGenerator): Initialized cloud generator object
    """
    # Ensure output directory exists
    os.makedirs(mask_output_folder, exist_ok=True)
    
    def tensor_to_image(tensor):
        """Convert tensor to PIL Image"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        if isinstance(tensor, np.ndarray):
            # Remove batch dimension if present
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]  # (1, C, H, W) -> (C, H, W)

            # Handle channel-first shapes
            if tensor.ndim == 3:
                if tensor.shape[0] == 1:  # (1, H, W)
                    tensor = tensor[0]
                elif tensor.shape[0] == 3:  # (3, H, W) → choose first channel
                    tensor = tensor[0]
                else:
                    raise ValueError(f"Unexpected channel size in shape {tensor.shape}")

            if tensor.ndim != 2:
                raise ValueError(f"Expected 2D array for grayscale image, got shape {tensor.shape}")

            # Normalize to uint8 if necessary
            if tensor.dtype != np.uint8:
                tensor = (tensor * 255).astype(np.uint8)

            return Image.fromarray(tensor)

        raise TypeError(f"Unsupported input type: {type(tensor)}")
    
    # Iterate through images in the input folder
    for file_name in tqdm(os.listdir(input_folder)):
        # Construct full file path
        file_path = os.path.join(input_folder, file_name)
        
        try:
            # Open image
            img = Image.open(file_path).convert("RGB")
            img.filename = file_name  # preserve filename manually
            
            # Generate cloud masks
            results, cmask, smask = cloud_generator.generate_clouds(img)
            
            # If cloud generation was successful
            if results is not None:
                # Convert masks to images
                cmask_img = tensor_to_image(cmask)
                #smask_img = tensor_to_image(smask)
                
                # Generate base filename (without extension)
                base_name = os.path.splitext(file_name)[0]
                
                # Save masks
                cmask_img.save(os.path.join(mask_output_folder, f"{base_name}_cloud_mask.png"))
                #smask_img.save(os.path.join(mask_output_folder, f"{base_name}_shadow_mask.png"))
            
        except Exception as e:
            print(f"Error processing image {file_name}: {e}")

def main():
    # Initialize the cloud generator
    cloud_generator = CloudGenerator(
        min_lvl=0,
        max_lvl=0.8,
        locality_degree=2,
        blur_scaling=0,
        channel_offset=0,
        shadow_max_lvl=0.25,
        cloud_probability=0.25
    )
    
    # Folder paths
    input_folder = r"E:\Datasets\masati-thesis\synthetic_images\2_augmentation"
    mask_output_folder = r"E:\Datasets\masati-thesis\synthetic_images\5_cloud_masks"
    
    # Process images
    process_image_with_cloud_generator(input_folder, mask_output_folder, cloud_generator)
    
    print("Cloud masks saved.")

if __name__ == "__main__":
    main()
