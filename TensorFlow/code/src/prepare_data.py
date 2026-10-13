from pathlib import Path
from urllib.parse import urlparse
import tarfile
import requests
import random
import shutil


# ============================================================
# Configuration
# ============================================================

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_SEED = 42

DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar"
)

data_dir = Path("./data")

train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"


# ============================================================
# Helper function to display dataset sizes
# ============================================================

def print_dataset_counts():
    for split_dir in [train_dir, val_dir, test_dir]:

        print(f"\n{split_dir.name}:")

        if not split_dir.exists():
            print("  Directory does not exist.")
            continue

        total = 0

        for class_dir in sorted(split_dir.iterdir()):

            if class_dir.is_dir():
                count = sum(
                    1 for p in class_dir.iterdir() if p.is_file()
                )

                print(f"  {class_dir.name}: {count}")
                total += count

        print(f"  Total: {total}")


# ============================================================
# Main data preparation function
# ============================================================

def download_and_prepare_data(
    train_size=TRAIN_SIZE,
    val_size=VAL_SIZE,
    test_size=TEST_SIZE,
    random_seed=RANDOM_SEED,
):
    # ============================================================
    # Sanity check
    # ============================================================

    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_SIZE + VAL_SIZE + TEST_SIZE must equal 1.0"
        )

    # --------------------------------------------------------
    # Check whether the dataset has already been prepared
    # --------------------------------------------------------

    if train_dir.is_dir() and val_dir.is_dir() and test_dir.is_dir():

        print(
            "Train, validation, and test directories already exist."
        )
        print("The geospatial data is already set up.")

        print_dataset_counts()

        return train_dir, val_dir, test_dir


    # --------------------------------------------------------
    # Create data directory
    # --------------------------------------------------------

    data_dir.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------------
    # Download original dataset
    # --------------------------------------------------------

    filename = Path(urlparse(DATA_URL).path).name
    tar_path = data_dir / filename

    if tar_path.exists():

        print(f" Tar file already exists: {tar_path}")

    else:

        print(f"Downloading dataset from:\n{DATA_URL}")

        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()

        with open(tar_path, "wb") as f:

            for chunk in response.iter_content(chunk_size=8192):

                if chunk:
                    f.write(chunk)

        print(f"Downloaded to: {tar_path}")


    # --------------------------------------------------------
    # Extract dataset
    # --------------------------------------------------------

    extracted_dir = data_dir / "images_dataSAT"

    if extracted_dir.exists():

        print(f"Data already extracted: {extracted_dir}")

    else:

        print("Extracting dataset...")

        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=data_dir)

        print(f"Extracted data to: {extracted_dir}")


    # --------------------------------------------------------
    # Locate class directories
    # --------------------------------------------------------

    dir_non_agri = extracted_dir / "class_0_non_agri"
    dir_agri = extracted_dir / "class_1_agri"


    # --------------------------------------------------------
    # Get image paths
    # --------------------------------------------------------

    non_agri_images = [
        p for p in dir_non_agri.iterdir()
        if p.is_file()
    ]

    agri_images = [
        p for p in dir_agri.iterdir()
        if p.is_file()
    ]


    print("\nOriginal dataset:")

    print(f"  class_0_non_agri: {len(non_agri_images)}")
    print(f"  class_1_agri:     {len(agri_images)}")


    # --------------------------------------------------------
    # Shuffle each class independently
    # --------------------------------------------------------

    rng = random.Random(random_seed)

    rng.shuffle(non_agri_images)
    rng.shuffle(agri_images)


    # --------------------------------------------------------
    # Calculate split sizes
    # --------------------------------------------------------

    def calculate_split_sizes(images):

        total = len(images)

        train_count = int(total * train_size)
        val_count = int(total * val_size)

        # Whatever remains goes into test.
        # This guarantees every image is assigned exactly once.
        test_count = total - train_count - val_count

        return train_count, val_count, test_count


    (
        non_agri_train_count,
        non_agri_val_count,
        non_agri_test_count,
    ) = calculate_split_sizes(non_agri_images)

    (
        agri_train_count,
        agri_val_count,
        agri_test_count,
    ) = calculate_split_sizes(agri_images)


    print("\nPlanned split:")

    print(
        f"  class_0_non_agri: "
        f"{non_agri_train_count} train / "
        f"{non_agri_val_count} val / "
        f"{non_agri_test_count} test"
    )

    print(
        f"  class_1_agri:     "
        f"{agri_train_count} train / "
        f"{agri_val_count} val / "
        f"{agri_test_count} test"
    )


    # --------------------------------------------------------
    # Split image lists
    # --------------------------------------------------------

    non_agri_train = non_agri_images[:non_agri_train_count]

    non_agri_val = non_agri_images[
        non_agri_train_count:
        non_agri_train_count + non_agri_val_count
    ]

    non_agri_test = non_agri_images[
        non_agri_train_count + non_agri_val_count:
    ]


    agri_train = agri_images[:agri_train_count]

    agri_val = agri_images[
        agri_train_count:
        agri_train_count + agri_val_count
    ]

    agri_test = agri_images[
        agri_train_count + agri_val_count:
    ]


    # --------------------------------------------------------
    # Create destination directories
    # --------------------------------------------------------

    splits = {
        "train": {
            "class_0_non_agri": non_agri_train,
            "class_1_agri": agri_train,
        },
        "val": {
            "class_0_non_agri": non_agri_val,
            "class_1_agri": agri_val,
        },
        "test": {
            "class_0_non_agri": non_agri_test,
            "class_1_agri": agri_test,
        },
    }


    # --------------------------------------------------------
    # Move images into their respective splits
    # --------------------------------------------------------

    for split_name, classes in splits.items():

        split_dir = data_dir / split_name

        for class_name, image_paths in classes.items():

            destination_dir = split_dir / class_name

            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for image_path in image_paths:

                shutil.move(
                    str(image_path),
                    str(destination_dir / image_path.name)
                )


    # --------------------------------------------------------
    # Remove the now-empty extracted dataset
    # --------------------------------------------------------

    shutil.rmtree(extracted_dir)

    print("\nOriginal extracted dataset removed.")


    # --------------------------------------------------------
    # Display final dataset counts
    # --------------------------------------------------------

    print("\nDataset created successfully.")

    print_dataset_counts()

    return train_dir, val_dir, test_dir


# ============================================================
# Execute only when this file is run directly
# ============================================================

if __name__ == "__main__":
    download_and_prepare_data()

