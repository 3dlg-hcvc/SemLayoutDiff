"""
FRONT3D Dataset loader and utilities.

This module provides PyTorch Dataset classes and utilities for loading and processing
FRONT3D room layout data for machine learning tasks.
"""

from PIL import Image
import os
import torch
import numpy as np
from typing import Optional, Tuple, Union, Callable

class ToTensorNoNorm:
    """
    Convert PIL Image to PyTorch tensor without normalization.

    This transform converts a PIL Image to a PyTorch tensor while preserving
    the original pixel values (no normalization applied).
    """

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Convert image to tensor.

        Args:
            image: PIL Image or numpy array to convert

        Returns:
            PyTorch tensor with dimensions (C, H, W)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        else:
            image = np.array(image)

        if len(image.shape) == 2:
            # Add channel dimension for grayscale images
            image = image[:, :, None]

        return torch.from_numpy(image.copy()).permute(2, 0, 1)


class FRONT3D(torch.utils.data.Dataset):
    """
    FRONT3D Dataset for room layout processing.

    This dataset loads FRONT3D room layout images and optionally provides
    room type labels for conditional generation tasks.
    """

    # Room type mappings
    ROOM_TYPES = {
        'bedroom': 0,
        'dining': 1, 
        'living': 2
    }

    def __init__(self, root_dir: str, transform: Optional[Callable] = None, room_condition: bool = False):
        """
        Initialize FRONT3D dataset.

        Args:
            root_dir: Directory containing the FRONT3D room data
            transform: Optional transform to be applied to the images
            room_condition: Whether to include room type labels in the output
        """
        self.root_dir = root_dir
        self.transform = transform
        self.room_condition = room_condition

        # Find all valid room directories
        self.room_list = self._build_room_list()

        # Count rooms that need duplication (living-dining combined rooms)
        self.duplicate = len([room for room in self.room_list 
                            if self._is_living_dining_room(room)])

    def _build_room_list(self) -> list:
        """Build list of valid room directories."""
        room_list = []

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        for room_name in os.listdir(self.root_dir):
            room_path = os.path.join(self.root_dir, room_name)
            label_map_path = os.path.join(room_path, 'Updated_Bottom_label_map.png')

            if os.path.isdir(room_path) and os.path.exists(label_map_path):
                room_list.append(room_path)

        if not room_list:
            raise ValueError(f"No valid room directories found in {self.root_dir}")

        return room_list

    def _is_living_dining_room(self, room_path: str) -> bool:
        """Check if room is a combined living-dining room."""
        room_name = room_path.lower()
        return 'living' in room_name and 'dining' in room_name
            
    def __len__(self) -> int:
        """
        Get total number of samples including duplicates.

        Living-dining rooms are counted twice (once as living, once as dining).
        """
        return len(self.room_list) + self.duplicate

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[int, torch.Tensor], list]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            - If room_condition is False: image tensor
            - If room_condition is True: (room_type, image_tensor) or list of tuples for living-dining rooms
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Handle index bounds for room list
        room_idx = min(idx, len(self.room_list) - 1)
        room_path = self.room_list[room_idx]

        # Load the label map image
        instance_map = self._load_image(room_path)

        # Apply transforms if specified
        if self.transform:
            instance_map = self.transform(instance_map)

        # Return based on room conditioning requirements
        if not self.room_condition:
            return instance_map

        # Handle living-dining rooms first (they need special treatment)
        if self._is_living_dining_room(room_path):
            # Return both living and dining variants
            return [
                (self.ROOM_TYPES['living'], instance_map),
                (self.ROOM_TYPES['dining'], instance_map)
            ]

        # Determine room type for single room types
        room_type = self._get_room_type(room_path)
        return room_type, instance_map

    def _load_image(self, room_path: str) -> Image.Image:
        """Load the label map image for a room."""
        image_path = os.path.join(room_path, 'Updated_Bottom_label_map.png')

        try:
            return Image.open(image_path)
        except Exception as e:
            raise IOError(f"Failed to load image from {image_path}: {str(e)}")

    def _get_room_type(self, room_path: str) -> int:
        """
        Determine room type from path for single room types only.

        Note: This method should not be called for living-dining rooms,
        as those are handled specially in __getitem__.

        Args:
            room_path: Path to the room directory

        Returns:
            Room type ID (0=bedroom, 1=dining, 2=living)
        """
        room_name = room_path.lower()

        if 'bedroom' in room_name:
            return self.ROOM_TYPES['bedroom']
        elif 'living' in room_name and 'dining' not in room_name:
            return self.ROOM_TYPES['living']
        elif 'dining' in room_name and 'living' not in room_name:
            return self.ROOM_TYPES['dining']
        else:
            raise ValueError(f"Unknown or unsupported room type for path: {room_path}")
