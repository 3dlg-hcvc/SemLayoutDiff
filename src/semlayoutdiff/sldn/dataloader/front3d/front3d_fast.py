import os
import json

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class Front3DFast(data.Dataset):
    """
    Fast dataloader for Front3D dataset supporting multiple conditioning modes.
    
    This dataset class provides efficient loading of 3D indoor layout data with support for:
    - Floor plan conditioning
    - Architecture conditioning (floor, doors, windows)
    - Room type conditioning
    - Mixed condition training
    - Text conditioning
    
    Args:
        root (str): Root directory of the dataset
        split (str): Dataset split ('train', 'val', 'test')
        resolution (tuple): Image resolution as (height, width)
        transform: Optional transform to apply to images
        floor_plan (bool): Whether to generate floor plan conditioning
        wo_floor (bool): Whether to remove floor from semantic maps
        room_type_condition (bool): Whether to use room type conditioning
        w_arch (bool): Whether to include architecture elements
        wo_arch (bool): Whether to exclude architecture elements
        specific_room_type (str): Filter for specific room type
        text_condition (bool): Whether to use text conditioning
        mixed_condition (bool): Whether to use mixed conditioning
        condition_types (list): Types of conditions for mixed training
        condition_prob (list): Probabilities for each condition type
    """
    
    def __init__(self, root, split='unified_w_arch', resolution=(32, 64), transform=None, floor_plan=False, wo_floor=False,
                 room_type_condition=False, w_arch=False, wo_arch=False, specific_room_type=None, text_condition=False,
                 mixed_condition=False, condition_types=['none', 'floor', 'arch'], condition_prob=[0.33, 0.33, 0.34]):
        assert resolution in [(32, 64), (128, 256), (240, 320), (48, 64), (64, 64), (120, 120)]

        H, W = resolution

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.floor_plan = floor_plan
        self.wo_floor = wo_floor
        self.room_type_condition = room_type_condition
        self.w_arch = w_arch
        self.wo_arch = wo_arch
        self.specific_room_type = specific_room_type
        self.text_condition = text_condition
        
        # Mixed condition training parameters
        self.mixed_condition = mixed_condition
        self.condition_types = condition_types
        if condition_prob is None:
            # Equal probability for all condition types
            self.condition_prob = [1.0 / len(condition_types)] * len(condition_types)
        else:
            assert len(condition_prob) == len(condition_types), "condition_prob length must match condition_types length"
            assert abs(sum(condition_prob) - 1.0) < 1e-6, "condition_prob must sum to 1.0"
            self.condition_prob = condition_prob
        if not self._check_exists(H, W, split):
            raise RuntimeError('Dataset not found (or incomplete) at {}'.format(self.root))
        if text_condition:
            self.text_data = np.load(os.path.join(self.root, split + f'_{H}x{W}.npy'), allow_pickle=True).item()
            self.data =  torch.from_numpy(self.text_data["layout"])
            self.text_data = torch.from_numpy(self.text_data["text_embeddings"])
        else:
            self.data = torch.from_numpy(
                np.load(os.path.join(self.root, split + f'_{H}x{W}.npy'), allow_pickle=True))
        
        # Filter and map data for specific room type if specified
        if specific_room_type is not None:
            self.data = self._filter_and_map_room_type(self.data, specific_room_type)

        if room_type_condition:
            # Assign weights for each data item based on room type
            sample_weights = []
            for item in self.data:
                room_type_id = torch.unique(item[0])
                if room_type_id == 0:
                    sample_weights.append(0.07)
                elif room_type_id == 1:
                    sample_weights.append(0.9)
                elif room_type_id == 2:
                    sample_weights.append(0.35)
                
            self.weights = sample_weights

        # Get class IDs for floor, door, window from JSON mapping (based on room type)
        room_types = ['bedroom', 'diningroom', 'livingroom', 'unified']
        room_type = next((rt for rt in room_types if rt in self.split), None)
        if self.specific_room_type is not None:
            room_type = self.specific_room_type
        if room_type:
            config_file = os.path.join("preprocess/metadata", f"{room_type}_idx_to_generic_label.json")
            with open(config_file, 'r') as f:
                self.cls_label_map = json.load(f)
                for item in ["floor", "door", "window"]:
                    setattr(self, f"{item}_id", int(next(key for key, value in self.cls_label_map.items() if value == item)))

    def _check_exists(self, H, W, split):
        """
        Check if the dataset files exist for the specified resolution and split.
        
        Args:
            H (int): Image height
            W (int): Image width  
            split (str): Dataset split name
            
        Returns:
            bool: True if dataset files exist, False otherwise
        """
        return os.path.exists(os.path.join(self.root, split + f'_{H}x{W}.npy'))

    def shift_ids_without_arch(self, img):
        """
        Shift object IDs when door and window are removed (set to 0).
        
        This method removes architectural elements (doors and windows) by setting them to 0
        and shifts all higher-valued object IDs down to maintain sequential numbering.
        
        Args:
            img (torch.Tensor): Input semantic map tensor
            
        Returns:
            torch.Tensor: Semantic map with shifted IDs
        """
        # Create a copy to avoid modifying the original
        shifted_img = img.clone()
        window_id = self.window_id
        door_id = self.door_id

        # Remove doors and shift higher IDs down
        shifted_img[img == door_id] = 0
        mask = shifted_img > door_id
        shifted_img[mask] -= 1
        
        # Update window_id after door removal and repeat process
        window_id -= 1
        shifted_img[shifted_img == window_id] = 0
        mask = shifted_img > window_id
        shifted_img[mask] -= 1

        return shifted_img

    def _create_floor_plan_condition(self, img, condition_type):
        """Create floor plan conditioning based on condition type."""
        floor_plan = torch.zeros_like(img)
        
        if condition_type == 'none':
            # No conditioning - floor_plan is all zeros
            pass
        elif condition_type == 'floor':
            # Floor conditioning - binary floor plan (1 for floor, 0 for background)
            floor_plan[img != 0] = 1
        elif condition_type == 'arch':
            # Architecture conditioning - floor=1, door=2, window=3
            floor_plan[img != 0] = 1  # floor is 1
            if hasattr(self, 'door_id'):
                floor_plan[img == self.door_id] = 2
            if hasattr(self, 'window_id'):
                floor_plan[img == self.window_id] = 3
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")
            
        return floor_plan

    def _create_arch_map(self, img):
        """Create architecture map including floor, door, window."""
        arch_map = torch.zeros_like(img)
        arch_map[img != 0] = 1
        
        if hasattr(self, 'door_id'):
            arch_map[img == self.door_id] = 2
        if hasattr(self, 'window_id'):
            arch_map[img == self.window_id] = 3
            
        return arch_map

    def _process_floor_plan_removal(self, img):
        """Remove floor from semantic map and update category IDs."""
        if self.room_type_condition:
            img[img > 1] -= 1
            img[img == 1] = 0
        else:
            if "living" in self.split or "dining" in self.split:
                img[img > 13] -= 1
                img[img == 13] = 0
            elif "bed" in self.split:
                img[img > 12] -= 1
                img[img == 12] = 0
        return img

    def __getitem__(self, index):
        room_type = -1
        if self.room_type_condition:
            room_type = torch.tensor(np.unique(self.data[index][0])).long()
            assert len(np.unique(self.data[index][0])) == 1
            img = self.data[index][1]
        else:
            img = self.data[index]

        img = img.long()
        
        # Initialize condition_id for mixed condition training
        condition_id = -1
        
        # Handle mixed condition training
        if self.mixed_condition:
            # Randomly sample condition type
            condition_id = np.random.choice(len(self.condition_types), p=self.condition_prob)
            selected_condition = self.condition_types[condition_id]
            
            floor_plan = self._create_floor_plan_condition(img, selected_condition)
            
            # Ensure proper dimensions
            if floor_plan.dim() == 2:  # If it's [H, W]
                floor_plan = floor_plan.unsqueeze(0)
        else:
            # Original logic for non-mixed condition training
            if self.w_arch:
                arch_map = self._create_arch_map(img)
            elif self.wo_arch:
                img = self.shift_ids_without_arch(img)

            if self.floor_plan:
                # Create binary floor plan (1 for floor, 0 for background)
                floor_plan = torch.zeros_like(img)
                floor_plan[img != 0] = 1
                floor_plan = floor_plan.clone().detach().long().unsqueeze(0)

                # Remove floor from semantic map if requested
                if self.wo_floor:
                    img = self._process_floor_plan_removal(img)
            else:
                floor_plan = torch.tensor(0)
            
            if self.w_arch:
                floor_plan = arch_map.clone().detach().long()
                if floor_plan.dim() == 2:  # If it's [H, W]
                    floor_plan = floor_plan.unsqueeze(0)

        if self.transform:
            if img.size(0) == 1:
                img = img[0]

            img = Image.fromarray(img.numpy().astype('uint8'))
            img = self.transform(img)

            img = np.array(img)

            img = torch.tensor(img).long()

            img = img.unsqueeze(0)
        
        # Single return: always return a fixed 5-tuple; fill missing parts with None
        room_type = room_type.squeeze(0) if isinstance(room_type, torch.Tensor) and room_type.dim() > 0 else room_type
        text_embedding = self.text_data[index].unsqueeze(0) if self.text_condition else []
        mixed_condition_id = torch.tensor(condition_id) if self.mixed_condition else []
        return (img, floor_plan, room_type, text_embedding, mixed_condition_id)

    def __len__(self):
        return len(self.data)
    
    def get_arch_floor_plan(self, room_type):
        """Get all architecture maps for a specific room type or all room types."""
        arch_maps = []
        
        for i in range(len(self.data)):
            data_item = self.__getitem__(i)
            
            # Filter by room type if specified
            if room_type != -1 and data_item[2] != room_type:
                continue
                
            # Extract the floor plan/architecture map
            floor_plan_item = data_item[1]
            
            # Handle different return formats
            if len(data_item) in [3, 4, 5]:
                arch_maps.append(floor_plan_item)
            else:
                arch_maps.append(floor_plan_item[2].unsqueeze(0))
                
        return arch_maps
    
    def _filter_and_map_room_type(self, data, specific_room_type):
        """
        Filter data for specific room type and map unified IDs to room-specific IDs
        Args:
            data: tensor data to filter
            specific_room_type: target room type ('bedroom', 'livingroom', or 'diningroom')
        Returns:
            filtered and mapped tensor data
        """
        room_type_map = {'bedroom': 0, 'livingroom': 1, 'diningroom': 2}
        if specific_room_type not in room_type_map:
            raise ValueError(f'specific_room_type should be one of {list(room_type_map.keys())}')
        
        # Filter data by room type
        target_type = room_type_map[specific_room_type]
        filtered_data = []
        for item in data:
            if np.unique(item[0])[0] == target_type:
                filtered_data.append(item)
        filtered_data = torch.stack(filtered_data)
        
        # Load label mappings
        with open("preprocess/scripts/config/unified_idx_to_generic_label_w_arch.json", 'r') as f:
            unified_labels = json.load(f)
        with open(f"preprocess/scripts/config/{specific_room_type}_idx_to_generic_label.json", 'r') as f:
            room_specific_labels = json.load(f)
        
        # Create mapping from unified IDs to room-specific IDs
        unified_to_specific = {}
        for unified_id, unified_label in unified_labels.items():
            for specific_id, specific_label in room_specific_labels.items():
                if unified_label == specific_label:
                    unified_to_specific[int(unified_id)] = int(specific_id)
                    break
        
        # Transform the data labels
        transformed_data = []
        for item in filtered_data:
            if self.room_type_condition:
                room_type, sem_map = item
            else:
                sem_map = item[1]
            
            # Create new semantic map with transformed IDs
            new_sem_map = torch.zeros_like(sem_map)
            for unified_id, specific_id in unified_to_specific.items():
                new_sem_map[sem_map == unified_id] = specific_id
            
            if self.room_type_condition:
                transformed_data.append(torch.stack([room_type, new_sem_map]))
            else:
                transformed_data.append(new_sem_map)
        
        return torch.stack(transformed_data)
