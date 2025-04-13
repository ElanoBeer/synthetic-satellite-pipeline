import satellite_cloud_generator as scg
from tqdm import tqdm

def cloud_generation(dataset, min_lvl=0.2, max_lvl=0.8, locality_degree=2, blur_scaling=0, channel_offset=0):
    """
    Generate clouds over a list of RGB satellite images using satellite_cloud_generator.

    Args:
        dataset: List or iterable of images (e.g., numpy arrays or PIL images)
        min_lvl, max_lvl: Cloud intensity levels
        locality_degree: Controls how localized the clouds are
        blur_scaling: Gaussian blur applied to cloud mask
        channel_offset: Channel color shift in cloud overlay

    Returns:
        List of tuples: [(clouded_img, cloud_mask, shadow_mask), ...]
    """

    cloud_dataset = [] # list

    # Iterate of the image dataset to generate clouds.
    for img in tqdm(dataset, desc="Generating clouds"):
        cl, cmask, smask = scg.add_cloud_and_shadow(
            img,
            min_lvl=min_lvl,
            max_lvl=max_lvl,
            locality_degree=locality_degree,
            cloud_color=False,
            channel_offset=channel_offset,
            blur_scaling=blur_scaling,
            return_cloud=True,
            const_scale=True,
            noise_type='perlin'
        )
        cloud_dataset.append((cl, cmask, smask))

    return cloud_dataset
