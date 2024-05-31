import os

import pandas as pd
from sklearn.model_selection import train_test_split


def create_image_markup_csv_file(image_dir: str, output_csv_file: str):
    image_files = [
        os.path.join(ddir, img)
        for ddir in os.listdir(image_dir)
        for img in os.listdir(os.path.join(image_dir, ddir))
    ]
    result = pd.DataFrame({"image": image_files})
    result["label"] = result["image"].apply(lambda x: x.split("/")[-2])
    result.to_csv(output_csv_file, index=False)


def train_val_test_split(
    initial_dataset: str, val_part: str, test_part: float, testing_category: int
):
    initial_data = pd.read_csv(initial_dataset)
    test_category_data = initial_data[initial_data["label"] == testing_category]
    initial_data = initial_data[initial_data["label"] != testing_category]

    train, val = train_test_split(initial_data, test_size=val_part)
    train, test = train_test_split(train, test_size=test_part)
    test_category_part = test.shape[0] // test["label"].nunique()

    test = pd.concat([test_category_data[:test_category_part], test], ignore_index=True)
    train.to_csv(f"{os.path.dirname(initial_dataset)}/train.csv", index=False)
    val.to_csv(f"{os.path.dirname(initial_dataset)}/val.csv", index=False)
    test.to_csv(f"{os.path.dirname(initial_dataset)}/test.csv", index=False)

    print(
        "Amount of images in train: ",
        train.shape[0],
        "\n",
        "Amount of images in val: ",
        val.shape[0],
        "\n",
        "Amount of images in test: ",
        test.shape[0],
        "\n",
    )


if __name__ == "__main__":
    create_image_markup_csv_file(
        "/home/ana/University/Tamgi/data/dataset/crop",
        "/home/ana/University/Tamgi/data/dataset/crop_labels/data.csv",
    )
    train_val_test_split(
        "/home/ana/University/Tamgi/data/dataset/crop_labels/data.csv", 0.15, 0.15, 3
    )
