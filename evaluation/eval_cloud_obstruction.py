import os
import numpy as np
from PIL import Image
import json
import pandas as pd
from tqdm import tqdm


def load_annotations(annotations_folder):
    """
    Load annotations from JSON files with 'boxes' key in Pascal VOC format.

    Args:
        annotations_folder (str): Path to the folder with annotation files

    Returns:
        dict: A dictionary with base filename as key and list of bounding boxes as value
    """
    annotations = {}

    for filename in os.listdir(annotations_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(annotations_folder, filename)

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract base filename (remove .json extension)
                base_filename = os.path.splitext(filename)[0]

                # Extract boxes in the format [x_min, y_min, x_max, y_max]
                raw_boxes = data.get('boxes', [])

                # Convert to dictionary format
                boxes = [
                    {
                        'xmin': int(box[0]),
                        'ymin': int(box[1]),
                        'xmax': int(box[2]),
                        'ymax': int(box[3])
                    } for box in raw_boxes
                ]

                annotations[base_filename] = boxes

            except Exception as e:
                print(f"Error processing annotation file {filename}: {e}")

    return annotations


def calculate_image_obstruction(cloud_mask, bboxes, obstruction_thresholds=[0.25, 0.5, 0.75, 1.0]):
    """
    Calculate binary obstruction for an image at multiple thresholds.

    Args:
        cloud_mask (numpy.ndarray): Cloud mask image as a numpy array
        bboxes (list): List of bounding boxes for the image
        obstruction_thresholds (list): List of obstruction levels to calculate

    Returns:
        dict: Binary obstruction status for each threshold
    """
    # Initialize obstruction results
    obstruction_results = {threshold: False for threshold in obstruction_thresholds}

    # Check each bounding box
    for bbox in bboxes:
        # Extract bounding box coordinates
        xmin, ymin = bbox['xmin'], bbox['ymin']
        xmax, ymax = bbox['xmax'], bbox['ymax']

        # Ensure coordinates are within image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(cloud_mask.shape[1], xmax)
        ymax = min(cloud_mask.shape[0], ymax)

        # Crop cloud mask to bounding box
        bbox_mask = cloud_mask[ymin:ymax, xmin:xmax]

        # Total pixels in the bounding box
        total_pixels = (ymax - ymin) * (xmax - xmin)

        # Skip if no pixels
        if total_pixels == 0:
            continue

        # Check obstruction for each threshold
        for threshold in obstruction_thresholds:
            # Count pixels above the threshold
            obstructed_pixels = np.sum(bbox_mask >= (threshold * 255))

            # Calculate obstruction percentage
            obstruction_score = obstructed_pixels / total_pixels

            # Update obstruction status
            if obstruction_score > 0:
                obstruction_results[threshold] = True

    return obstruction_results


def process_cloud_obstruction(annotations_folder, cloud_masks_folder,
                              output_csv,
                              obstruction_thresholds=[0.25, 0.5, 0.75, 1.0],
                              image_extensions=['.jpg', '.png']):
    """
    Process all images to calculate dataset-level cloud obstruction scores.

    Args:
        annotations_folder (str): Path to annotations folder
        cloud_masks_folder (str): Path to cloud masks folder
        output_csv (str): Path to save output CSV
        obstruction_thresholds (list): Cloud mask thresholds to calculate
        image_extensions (list): Possible image file extensions to check
    """
    # Load annotations
    annotations = load_annotations(annotations_folder)

    # Prepare results storage
    dataset_obstruction = {threshold: [] for threshold in obstruction_thresholds}
    example_filenames = {threshold: None for threshold in obstruction_thresholds}

    # Process each annotated image
    for base_filename, bboxes in tqdm(annotations.items(), desc="Processing images"):
        # Find corresponding cloud mask
        cloud_mask_filename = f"{base_filename}_cloud_mask.png"
        cloud_mask_path = os.path.join(cloud_masks_folder, cloud_mask_filename)

        # If cloud mask doesn't exist, try finding the original image
        original_image_filename = None
        if not os.path.exists(cloud_mask_path):
            for ext in image_extensions:
                potential_image_path = os.path.join(cloud_masks_folder, base_filename + ext)
                if os.path.exists(potential_image_path):
                    original_image_filename = potential_image_path
                    break

            if original_image_filename is None:
                continue

        # Load cloud mask
        if cloud_mask_path and os.path.exists(cloud_mask_path):
            cloud_mask = np.array(Image.open(cloud_mask_path).convert('L'))
        elif original_image_filename:
            cloud_mask = np.array(Image.open(original_image_filename).convert('L'))
        else:
            print(f"Could not load cloud mask for {base_filename}")
            continue

        # Calculate binary obstruction for the image
        image_obstruction = calculate_image_obstruction(
            cloud_mask,
            bboxes,
            obstruction_thresholds
        )

        used_examples = set()
        ...
        for threshold, is_obstructed in image_obstruction.items():
            dataset_obstruction[threshold].append(is_obstructed)
            if (
                    is_obstructed
                    and example_filenames[threshold] is None
                    and base_filename not in used_examples
            ):
                example_filenames[threshold] = base_filename
                used_examples.add(base_filename)

    # Calculate dataset-level obstruction scores
    obstruction_scores = {}
    for threshold, obstruction_list in dataset_obstruction.items():
        obstruction_scores[threshold] = np.mean(obstruction_list)

    # Create DataFrame with results
    results_df = pd.DataFrame.from_dict(obstruction_scores, orient='index', columns=['obstruction_score'])
    results_df.index.name = 'threshold'
    results_df.reset_index(inplace=True)

    # Save to CSV
    results_df.to_csv(output_csv, index=False)

    # Print results
    print("Dataset Cloud Obstruction Scores:")
    for threshold, score in obstruction_scores.items():
        print(f"Threshold {threshold}: {score:.2%} of images obstructed")
        if example_filenames[threshold]:
            print(f"Example filename at threshold {threshold}: {example_filenames[threshold]}")

    print(f"\nDetailed results saved to {output_csv}")


def main():

    annotations_folder = r"E:\Datasets\masati-thesis\synthetic_images\2_augmentation_annotation"
    cloud_masks_folder = r"E:\Datasets\masati-thesis\synthetic_images\5_cloud_masks"
    output_csv = r"dataset_cloud_obstruction_scores.csv"

    process_cloud_obstruction(
        annotations_folder,
        cloud_masks_folder,
        output_csv,
        obstruction_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        image_extensions=['.jpg', '.png']  # Add other extensions if needed
    )


if __name__ == "__main__":
    main()