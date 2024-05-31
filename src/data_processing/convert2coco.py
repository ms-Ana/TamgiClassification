import json
import os

import cv2
import marking
from tqdm import tqdm


def convert_to_coco_format(base_dir: str, coco_file: str):
    categories = []
    images = []
    annotations = []

    category_id = 1
    image_id = 1
    annotation_id = 1

    for category_name in tqdm(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category_name)
        if os.path.isdir(category_path):
            categories.append({"id": category_id, "name": category_name})

            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                images.append(
                    {
                        "id": image_id,
                        "file_name": os.path.join(category_path, image_name),
                        "category_id": category_id,
                    }
                )
                img_array = cv2.imread(image_path)
                bbox = marking.get_detection(img_array)
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, width, height],
                        "area": area,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

                image_id += 1
            category_id += 1

    with open(coco_file, "w") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": categories},
            f,
            indent=2,
        )


def add_image_dimensions(coco_file: str, image_dir: str):
    with open(coco_file, "r") as file:
        coco_data = json.load(file)

    for image in coco_data["images"]:
        image_path = os.path.join(image_dir, image["file_name"])  # Update this path
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        image["width"] = width
        image["height"] = height

    with open(coco_file, "w") as f:
        json.dump(coco_data, f, indent=4)


def remove_nonexistent_images_from_coco(
    coco_json_path: str, images_dir: str, output_json_path: str
):
    """
    Remove images and their associated annotations from a COCO dataset if the images do not exist.

    Args:
        coco_json_path (str): Path to the COCO dataset JSON file.
        images_dir (str): Directory where the images are stored.
        output_json_path (str): Path to save the modified COCO dataset JSON file.
    """
    # Load the COCO dataset JSON file
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Initialize lists to hold existing images and annotations
    existing_images = []
    existing_annotations = []

    # Check each image to see if it exists in the specified directory
    for image in coco_data["images"]:
        image_path = os.path.join(images_dir, image["file_name"])
        if os.path.exists(image_path):
            existing_images.append(image)
        else:
            print(f"Image not found and will be removed: {image['file_name']}")

    # Filter annotations to keep only those associated with existing images
    existing_image_ids = {image["id"] for image in existing_images}
    for annotation in coco_data["annotations"]:
        if annotation["image_id"] in existing_image_ids:
            existing_annotations.append(annotation)

    # Update the COCO data with existing images and annotations
    coco_data["images"] = existing_images
    coco_data["annotations"] = existing_annotations

    # Save the modified dataset back to a JSON file
    with open(output_json_path, "w") as f:
        json.dump(coco_data, f, indent=4)


if __name__ == __name__:
    remove_nonexistent_images_from_coco(
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_test.json",
        "/home/ana/University/Tamgi/data/dataset/render",
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_test_cleaned.json",
    )
    remove_nonexistent_images_from_coco(
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_train.json",
        "/home/ana/University/Tamgi/data/dataset/render",
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_train_cleaned.json",
    )
    remove_nonexistent_images_from_coco(
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_val.json",
        "/home/ana/University/Tamgi/data/dataset/render",
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_val_cleaned.json",
    )
