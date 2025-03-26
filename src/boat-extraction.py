import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

# Define base dataset path
dataset_path = "C:/Users/20202016/Documents/Master/Master Thesis/Datasets/MasatiV2/"
output_path = os.path.join(dataset_path, "MasatiV2Boats2/")

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)


def find_label_folders(base_path):
    """
    Find all label folders in the dataset.
    """
    return [f for f in os.listdir(base_path) if "labels" in f.lower()]


def extract_boats(image_path, xml_path, image_filename, xml_filename, target_size):
    """
    Extract boats from an image using bounding box annotations, resize them to target_size, and save them.

    Args:
        image_path (str): Path to the image file
        xml_path (str): Path to the XML annotation file
        image_filename (str): Filename of the image
        xml_filename (str): Filename of the corresponding XML annotation file
    """
    # Load image
    img = cv2.imread(os.path.join(image_path, image_filename))
    if img is None:
        print(f"Error loading image: {image_filename}")
        return

    # Parse XML
    tree = ET.parse(os.path.join(xml_path, xml_filename))
    root = tree.getroot()

    boat_count = 0
    for obj in root.findall('object'):

        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        # Extract boat
        boat_crop = img[ymin:ymax, xmin:xmax]
        if boat_crop.size == 0:
            continue

        if target_size is not None:
            # Resize to target_size
            boat_crop = cv2.resize(boat_crop, target_size, interpolation=cv2.INTER_LANCZOS4)

        # Save cropped and resized boat image
        boat_filename = f"{os.path.splitext(image_filename)[0]}_boat{boat_count}.png"
        cv2.imwrite(os.path.join(output_path, boat_filename), boat_crop)
        boat_count += 1


# Process dataset
label_folders = find_label_folders(dataset_path)
for label_folder in label_folders:
    image_folder = label_folder.replace("_labels", "")
    image_folder_path = os.path.join(dataset_path, image_folder)
    xml_folder_path = os.path.join(dataset_path, label_folder)

    print(f"Extracting {image_folder}")

    if not os.path.exists(image_folder_path):
        print(f"Skipping {label_folder} as corresponding image folder not found.")
        continue

    for filename in tqdm(os.listdir(xml_folder_path)):
        if filename.endswith(".xml"):
            image_file = os.path.splitext(filename)[0] + ".png"
            if os.path.exists(os.path.join(image_folder_path, image_file)):
                extract_boats(image_folder_path, xml_folder_path, image_file, filename, target_size=None)

print("Boat extraction complete!")


