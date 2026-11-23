
import os
import shutil
import pandas as pd

# Dataset paths
DATASET_DIR = "dataset"

CSV_PATH = os.path.join(
    DATASET_DIR,
    "HAM10000_metadata.csv"
)

SOURCE_FOLDERS = [
    os.path.join(DATASET_DIR, "HAM10000_images_part_1"),
    os.path.join(DATASET_DIR, "HAM10000_images_part_2")
]

OUTPUT_DIR = os.path.join(DATASET_DIR, "organized")

# Read metadata
df = pd.read_csv(CSV_PATH)

# Seven skin lesion categories
classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# Create class folders
for category in classes:
    os.makedirs(
        os.path.join(OUTPUT_DIR, category),
        exist_ok=True
    )

# Build image lookup
image_paths = {}

for folder in SOURCE_FOLDERS:
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            image_id = os.path.splitext(filename)[0]

            image_paths[image_id] = os.path.join(
                folder,
                filename
            )

# Organize images
copied = 0
missing = 0

for _, row in df.iterrows():

    image_id = row["image_id"]
    category = row["dx"]

    source = image_paths.get(image_id)

    if source is None:
        missing += 1
        continue

    destination = os.path.join(
        OUTPUT_DIR,
        category,
        image_id + ".jpg"
    )

    shutil.copy2(source, destination)
    copied += 1

print("Dataset preparation completed!")
print("Images copied:", copied)
print("Images missing:", missing)
print("Organized dataset:", OUTPUT_DIR)
