import cv2
import matplotlib.pyplot as plt
from empatches import EMPatches

def visualize_patches_grid(patches, title=None):
    """
    Visualize image patches in a grid layout.

    Args:
        patches (list): List of image patches (numpy arrays)
        title (str, optional): Title for the plot. If None, will show patch size.
    """
    num_patches = len(patches)

    # Calculate grid dimensions using round-up division
    grid_size = int(-(-num_patches**0.5 // 1))  # Equivalent to math.ceil(num_patches ** 0.5)
    fig_size = (grid_size * 2, grid_size * 2)  # Adjust figure size dynamically based on grid size

    # Create the subplot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=fig_size)
    axes = axes.flatten()

    for i in range(num_patches):
        patch = patches[i]
        if len(patch.shape) == 2:  # Grayscale
            axes[i].imshow(patch, cmap="gray")
        else:  # RGB or multi-bands
            axes[i].imshow(patch[:, :, :3])  # Show the first 3 bands as RGB approximation
        axes[i].axis("off")

    # Hide unused axes
    for j in range(num_patches, len(axes)):
        axes[j].axis("off")

    # Add title
    if title is None:
        title = f"Original Patches with Size {patches[0].shape[:2]}"
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

# Specify and pre-process image
image_path = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/S2SHIPS/dataset_tif/toulon/2021-04-24-00_00_2021-04-24-23_59_Sentinel-2_L2A_True_color.jpg"
image = cv2.imread(image_path)
converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the patch and step size
patch_size = 224
overlap = 0.2

# Split the image into patches
emp = EMPatches()
img_patches, indices = emp.extract_patches(converted_image, patchsize=patch_size, overlap=overlap)

# Visualize the patches
visualize_patches_grid(img_patches, title="Visualized Patches Grid")

