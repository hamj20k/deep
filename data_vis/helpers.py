# Import necessary libraries
import matplotlib.pyplot as plt
import requests
import zipfile
from pathlib import Path
import torch
import torchvision
import random
import numpy as np
from PIL import Image
import matplotlib.patches as patches

def download_caltech_data():
    """
    Downloads the caltech101_5_classes data from the GitHub repository and extracts it to the data folder.

    Args:
        None
    Returns:
        image_path (Path): The path to the image folder.
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "caltech101_5_classes"

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download caltech101_5_classes data
        url = "https://github.com/muqeemmm/Deep-Learning-Assignments/raw/main/caltect_101_sub.zip"
        zip_path = data_path / "caltech101_5_classes.zip"
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            print("Downloading caltech101_5_classes data...")
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)
        else:
            print(f"Failed to download file, status code: {response.status_code}")
            response.raise_for_status()

        # Unzip caltech101_5_classes data
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                print("Unzipping caltech101_5_classes.zip...") 
                zip_ref.extractall(image_path)
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file.")
    
    return image_path

def compare_images(dataset: torch.utils.data.Subset, transform: torchvision.transforms, class_names: list, n: int=4):
    """
    Applies the given transformation to a random image from the dataset and compares the original and transformed images.

    Args:
        dataset (torch.utils.data.dataset.Subset): the original dataset containing the images.
        transform (torchvision.transforms): the transformation to apply to the images.
        class_names (list): The list of class names corresponding to the labels
        n (int): The number of pairs of images to compare. Default is 4.
    Returns:
        None
    """
    torch.manual_seed(0)
    for i in range(n):
        random_idx = torch.randint(0, len(dataset), (1,)).item()
        img, label = dataset[random_idx]
        img_transformed = transform(img)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img.squeeze(), cmap='gray') if img.shape[0] == 1 else ax[0].imshow(img.permute(1, 2, 0))
        ax[0].set_title(f'Original \n Size: {img.shape}')
        ax[0].axis('off')
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax[0].transAxes, linewidth=2, edgecolor='k', facecolor='none')
        ax[0].add_patch(rect)

        ax[1].imshow(img_transformed.squeeze(), cmap='gray') if img_transformed.shape[0] == 1 else ax[1].imshow(img_transformed.permute(1, 2, 0))
        ax[1].set_title(f'Transformed \n Size: {img_transformed.shape}')
        ax[1].axis('off')
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax[1].transAxes, linewidth=2, edgecolor='k', facecolor='none')
        ax[1].add_patch(rect)
        fig.suptitle(f"Class: {class_names[label]}", fontsize=14, fontweight='bold')

    plt.show()

def visualize_caltech_dataset(image_path: Path, transform: torchvision.transforms=None):
    """
    Visualizes a random sample of images from the CALTECH-101 dataset.

    Args:
        image_path (Path): The path to the image folder.
        transforms (torchvision.transforms): The transformation to apply to the images. Default is None.
    Returns:
        None
    """
    fig = plt.figure(figsize=(15, 4))
    rows, columns = 2, 8
    random.seed(42)
    for i in range(1, rows * columns + 1):
        fig.add_subplot(rows, columns, i)

        # 1. Get all image paths (* means "any combination")
        image_path_list = list(image_path.glob("*/*.jpg"))

        # 2. Get random image path
        random_image_path = random.choice(image_path_list)

        # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
        image_class = random_image_path.parent.stem

        # 4. Open image
        img = Image.open(random_image_path)
        if transform:
            img = transform(img)
            if img.shape[0] == 1:
                img = img.squeeze()
            else:
                img = img.permute(1, 2, 0)

        if np.array(img).ndim > 2:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap="gray")
        plt.title(f"Class:{image_class}\nShape:{list(np.array(img).shape)}", fontsize=8)
        plt.axis("off")

    if transform:
        plt.suptitle("CALTECH-101 (5 classes) - Transformed", fontweight="bold")
    else:
        plt.suptitle("CALTECH-101 (5 classes) - Original", fontweight="bold")
    plt.tight_layout()
    plt.savefig('caltech_101.png')
    plt.show()
