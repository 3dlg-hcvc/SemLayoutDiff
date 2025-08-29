import os
import json
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torchvision import transforms


class FurnitureDataset(Dataset):
    def __init__(self, root_dir, num_categories, num_orientation_class, floor_id, transform=None):
        self.num_categories = num_categories
        self.obj_infos = []
        self.num_orientation_class = num_orientation_class

        self.transform = transform

        image_paths = sorted(glob.glob(os.path.join(root_dir, '**', 'Updated_Bottom_label_map.png'), recursive=True))
        json_paths = sorted(glob.glob(os.path.join(root_dir, '**', '*_anno.json'), recursive=True))

        # Filter out invalid image paths
        valid_indices = [idx for idx, img_path in enumerate(image_paths) if Image.open(img_path).mode == "L"]
        image_paths = [image_paths[i] for i in valid_indices]
        json_paths = [json_paths[i] for i in valid_indices]

        assert len(image_paths) == len(json_paths), "Number of images and JSON files should be the same"

        for idx in tqdm(range(len(image_paths)), desc="Loading data:"):
            semantic_map_img = Image.open(image_paths[idx]).convert("L")
            semantic_map = torch.tensor(np.array(semantic_map_img), dtype=torch.uint8).unsqueeze(0)

            with open(json_paths[idx], 'r') as f:
                annotations = json.load(f)

            for anno in annotations:
                if int(anno["category"]) == floor_id or int(anno["category"]) == 37 or int(anno["category"]) == 38:
                    continue
                tmp_obj_info = {"semantic_map": semantic_map}
                instance_mask = self.decode_coco_mask(anno['mask'], semantic_map_img.size).unsqueeze(0)

                # Convert category ID to one-hot encoding
                category = torch.zeros(self.num_categories, dtype=torch.bool)
                category[int(anno['category'])] = 1

                # Extract other labels
                size = torch.tensor(anno['size'], dtype=torch.float32)
                offset = torch.tensor(anno['offset'], dtype=torch.float32).unsqueeze(0)
                orientation = torch.tensor(anno['orientation'], dtype=torch.float32)
                y_orientation = radians_to_degrees_normalized(orientation[2])
                orient = torch.tensor(self.classify_angle(y_orientation, self.num_orientation_class), dtype=torch.long)

                tmp_obj_info["attributes"] = (size, offset, orient)
                tmp_obj_info["category"] = category
                tmp_obj_info["instance_map"] = instance_mask

                self.obj_infos.append(tmp_obj_info)

    def __len__(self):
        return len(self.obj_infos)

    def __getitem__(self, idx):
        semantic_map = self.obj_infos[idx]["semantic_map"].to(torch.float32)
        instance_mask = self.obj_infos[idx]["instance_map"].to(torch.float32)
        category = self.obj_infos[idx]["category"].to(torch.float32)
        attribute = self.obj_infos[idx]["attributes"]

        if self.transform:
            # Apply the same transform to both semantic_map and instance_mask
            transformed = self.transform(semantic_map, instance_mask)
            semantic_map, instance_mask = transformed['image'], transformed['mask']

        return semantic_map, instance_mask, category, attribute

    @staticmethod
    def decode_coco_mask(polygon, image_shape):
        """
        Decode mask from COCO polygon format to binary mask.

        Args:
        - polygon (list): List of lists containing polygons.
        - image_shape (tuple): Shape of the original image (height, width).

        Returns:
        - Tensor: Binary mask of shape (1, height, width).
        """
        # Create a blank (black) image
        mask_img = Image.new('L', image_shape, 0)

        # Draw each polygon onto the image
        for poly in polygon:
            # Convert flat list into sequence of (x,y) pairs
            xy_pairs = list(zip(poly[::2], poly[1::2]))
            ImageDraw.Draw(mask_img).polygon(xy_pairs, outline=255, fill=255)

        # Convert to NumPy array and then to tensor
        mask_array = np.array(mask_img) / 255.0
        mask = torch.tensor(mask_array, dtype=torch.bool)
        return mask

    @staticmethod
    def classify_angle(angle, num_classes):
        # Calculate the class by dividing the angle by 45 or 90
        if num_classes == 4:
            class_label = int((angle + 45) // 90) % 4
        else:
            class_label = int((angle + 22.5) // 45) % 8
        return class_label

    @staticmethod
    def transform_pair(img, mask):
        """
        Apply the same transformation to both the image and the mask.
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(600),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
        ])

        return {
            'image': transform(img),
            'mask': transform(mask)
        }


def radians_to_degrees_normalized(radians):
    """Convert radians to degrees and normalize the result to the range [0, 360)."""
    degrees = radians * (180.0 / np.pi)
    # Normalize to the range [0, 360)
    degrees_normalized = degrees % 360
    return degrees_normalized

