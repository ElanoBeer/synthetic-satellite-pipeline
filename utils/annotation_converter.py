import os
import json


def aggregate_annotations(input_dirs, output_file):
    """
    Convert multiple directories of JSON files with PASCAL VOC format to a single JSON file
    with simplified format.

    Args:
        input_dirs (list): List of directories containing JSON files with PASCAL VOC annotations
        output_file (str): Path to output file
    """
    # Dictionary to store all annotations
    consolidated_annotations = {}

    total_files_processed = 0

    # Process each input directory
    for input_dir in input_dirs:
        # Get all JSON files in the current input directory
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        print(f"Found {len(json_files)} JSON files in {input_dir}")

        # Process each JSON file
        for json_file in json_files:
            json_path = os.path.join(input_dir, json_file)

            with open(json_path, 'r') as f:
                annotation_data = json.load(f)

            # Extract the bboxes and filename
            image_filename = os.path.splitext(json_file)[0] + ".png"
            bboxes = annotation_data["boxes"]

            # Add to consolidated annotations
            if bboxes:
                consolidated_annotations[image_filename] = {"bboxes": bboxes}
                print(f"Processed {json_file} - Found {len(bboxes)} bboxes for {image_filename}")
            else:
                print(f"Warning: No bounding boxes found in {json_file}")

            total_files_processed += 1

    # Write the consolidated annotations to output file
    with open(output_file, 'w') as f:
        json.dump(consolidated_annotations, f, indent=2)

    print(
        f"Successfully aggregated {len(consolidated_annotations)} annotations from {total_files_processed} files to {output_file}")
    return consolidated_annotations


class AnnotationConverter:
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    # ===== Format Conversions =====
    @staticmethod
    def coco_to_voc(self, coco_bbox):
        x, y, w, h = coco_bbox
        return [x, y, x + w, y + h]

    @staticmethod
    def voc_to_coco(self, voc_bbox):
        x_min, y_min, x_max, y_max = voc_bbox
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def voc_to_yolo(self, voc_bbox):
        x_min, y_min, x_max, y_max = voc_bbox
        x_center = ((x_min + x_max) / 2) / self.img_width
        y_center = ((y_min + y_max) / 2) / self.img_height
        width = (x_max - x_min) / self.img_width
        height = (y_max - y_min) / self.img_height
        return [x_center, y_center, width, height]

    def yolo_to_voc(self, yolo_bbox):
        x_center, y_center, width, height = yolo_bbox
        x_center *= self.img_width
        y_center *= self.img_height
        width *= self.img_width
        height *= self.img_height
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        return [x_min, y_min, x_max, y_max]

    def coco_to_yolo(self, coco_bbox):
        voc_bbox = self.coco_to_voc(coco_bbox)
        return self.voc_to_yolo(voc_bbox)

    def yolo_to_coco(self, yolo_bbox):
        voc_bbox = self.yolo_to_voc(yolo_bbox)
        return self.voc_to_coco(voc_bbox)

    # ===== Batch Conversion =====

    def convert(self, bbox_list, from_format, to_format):
        converted = []
        for bbox in bbox_list:
            if from_format == 'coco' and to_format == 'voc':
                converted.append(self.coco_to_voc(bbox))
            elif from_format == 'voc' and to_format == 'coco':
                converted.append(self.voc_to_coco(bbox))
            elif from_format == 'voc' and to_format == 'yolo':
                converted.append(self.voc_to_yolo(bbox))
            elif from_format == 'yolo' and to_format == 'voc':
                converted.append(self.yolo_to_voc(bbox))
            elif from_format == 'coco' and to_format == 'yolo':
                converted.append(self.coco_to_yolo(bbox))
            elif from_format == 'yolo' and to_format == 'coco':
                converted.append(self.yolo_to_coco(bbox))
            else:
                raise ValueError(f"Conversion from {from_format} to {to_format} not supported.")
        return converted


def main():
    # Define your input and output paths directly in the code
    input_dir = "E:/Datasets/masati-thesis/clone_annotations"  # Replace with your actual input directory path
    output_file = "E:/Datasets/masati-thesis/agg_annotations.json"  # Replace with your desired output file path

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert annotations
    aggregate_annotations(input_dir, output_file)


if __name__ == "__main__":
    main()