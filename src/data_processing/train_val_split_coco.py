import json
import os
import random


def merge_dataset(
    coco_file1: str,
    coco_file2: str,
    save_file: str,
    num_samples1: int = None,
    num_samples2: int = None,
):
    with open(coco_file1, "r") as file:
        coco1 = json.load(file)

    with open(coco_file2, "r") as file:
        coco2 = json.load(file)

    # Combine images
    combined_images = coco1["images"][:num_samples1] + coco2["images"][:num_samples2]

    # Combine annotations
    combined_annotations = (
        coco1["annotations"][:num_samples1] + coco2["annotations"][:num_samples2]
    )

    # Combine categories (optional, if categories are unique)
    combined_categories = coco1.get("categories", []) + coco2.get("categories", [])

    # Prepare new COCO dataset
    new_coco = {
        "images": combined_images,
        "annotations": combined_annotations,
        "categories": combined_categories,
    }

    # Save new COCO file
    with open(save_file, "w") as file:
        json.dump(new_coco, file, indent=4)


def split_coco_dataset(coco_file: str, train_ratio: float = 0.8):
    with open(coco_file, "r") as file:
        coco_data = json.load(file)

    # Shuffle the images to ensure random distribution
    random.shuffle(coco_data["images"])

    # Split images into train and validation sets
    num_train = int(len(coco_data["images"]) * train_ratio)
    train_images = coco_data["images"][:num_train]
    val_images = coco_data["images"][num_train:]

    # Function to filter annotations for a given set of images
    def filter_annotations(images, annotations):
        image_ids = {image["id"] for image in images}
        return [
            annotation
            for annotation in annotations
            if annotation["image_id"] in image_ids
        ]

    # Split annotations
    train_annotations = filter_annotations(train_images, coco_data["annotations"])
    val_annotations = filter_annotations(val_images, coco_data["annotations"])
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"],
    }
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data["categories"],
    }
    with open(f"{os.path.splitext(coco_file)[0]}_train.json", "w") as f:
        json.dump(train_coco, f, indent=2)

    with open(f"{os.path.splitext(coco_file)[0]}_test.json", "w") as f:
        json.dump(val_coco, f, indent=2)


def split_coco_by_label(coco_file: str, validation_category_id: int):
    with open(coco_file, "r") as file:
        coco_data = json.load(file)

    # Separate images based on whether they contain the validation category
    train_images, val_images = [], []
    for image in coco_data["images"]:
        class_name = os.path.dirname(image["file_name"])[-1]
        if class_name == str(validation_category_id):
            val_images.append(image)
        else:
            train_images.append(image)

    # Filter annotations based on the segregated images
    def filter_annotations(images, annotations):
        image_ids = {image["id"] for image in images}
        return [
            annotation
            for annotation in annotations
            if annotation["image_id"] in image_ids
        ]

    train_annotations = filter_annotations(train_images, coco_data["annotations"])
    val_annotations = filter_annotations(val_images, coco_data["annotations"])
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"],
    }
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data["categories"],
    }

    with open(f"{os.path.splitext(coco_file)[0]}_train.json", "w") as f:
        json.dump(train_coco, f, indent=2)
    with open(f"{os.path.splitext(coco_file)[0]}_test_cat.json", "w") as f:
        json.dump(val_coco, f, indent=2)


if __name__ == "__main__":
    merge_dataset(
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_test_cat.json",
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_test.json",
        "/home/ana/University/Tamgi/data/dataset/detection_labels/coco_annotations_test.json",
        70,
    )
