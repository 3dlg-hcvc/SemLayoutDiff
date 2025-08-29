"""
Sampling Utilities Module

This module contains utility classes for layout sampling operations including:
- Image processing and color mapping
- Floor plan loading from various sources  
- Text embedding processing
- Main layout sampling orchestration

These classes are designed to be reusable across different sampling scripts.
"""

import os
import json
import time
import pickle
from glob import glob
from typing import Optional, List, Tuple, Dict, Any

import cv2
import torch
import numpy as np
import imageio
import torchvision
from omegaconf import DictConfig

from .dataloader.front3d.front3d_fast import Front3DFast
from .dataloader.dataset_front3d import get_data
from .model import get_model
from diffusion_utils.base import DataParallelDistribution


class ImageProcessor:
    """Handles image processing and color mapping operations."""

    def __init__(self, colormap: Dict[int, Tuple[int, int, int]]):
        self.colormap = colormap
        # Floor plan colors (BGR format for OpenCV)
        self.floor_plan_colors = {
            0: [255, 255, 255],  # Background -> White
            1: [211, 211, 211],  # Floor -> Gray
            2: [0, 0, 153],      # Door -> Dark Blue
            3: [153, 153, 255]   # Window -> Light Blue
        }

    def instance_map_to_color(self, batch_instance_maps: torch.Tensor) -> np.ndarray:
        """Convert instance maps to colored images."""
        batch_size, _, height, width = batch_instance_maps.shape
        batch_instance_maps = np.squeeze(batch_instance_maps, axis=1)
        batch_colored_maps = np.zeros((batch_size, height, width, 3), dtype=np.uint8)

        for i in range(batch_size):
            instance_map = batch_instance_maps[i]
            for instance_id, color in self.colormap.items():
                mask = (instance_map == instance_id)
                batch_colored_maps[i, mask] = color

        return batch_colored_maps

    def create_composite_semantic_map(self, semantic_map: np.ndarray, floor_plan: np.ndarray, 
                                    with_arch: bool = False, floor_id: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a composite semantic map by replacing generated floor with custom floor plan
        and preserving furniture objects.

        Args:
            semantic_map: Generated semantic map with furniture objects
            floor_plan: Original floor plan (architectural elements)  
            with_arch: Whether the floor plan contains architectural elements
            floor_id: The semantic map ID that represents floor areas to be replaced

        Returns:
            Tuple of (composite_semantic_map, colored_composite_map)
        """
        # Start with the generated semantic map
        composite_map = semantic_map.copy()

        # Find furniture objects (non-floor, non-background)
        furniture_mask = (semantic_map > 0) & (semantic_map != floor_id)

        # Replace the entire map with custom floor plan first
        composite_map = floor_plan.copy()

        # Then overlay furniture objects back onto the custom floor plan
        composite_map[furniture_mask] = semantic_map[furniture_mask]

        # Create colored visualization
        colored_composite = np.zeros((*composite_map.shape, 3), dtype=np.uint8)

        # Color the custom floor plan elements first
        if with_arch:
            # Color architectural elements from custom floor plan
            for floor_plan_id, color in self.floor_plan_colors.items():
                if floor_plan_id == 0:  # Background will be handled separately
                    continue
                # Only color floor plan elements where there's no furniture
                floor_element_mask = (floor_plan == floor_plan_id) & (~furniture_mask)
                colored_composite[floor_element_mask] = color
        else:
            # Simple binary floor plan - color floor areas
            custom_floor_mask = (floor_plan > 0) & (~furniture_mask)
            colored_composite[custom_floor_mask] = self.floor_plan_colors[1]  # Gray for floor

        # Color furniture objects using original colormap (preserve original colors)
        for instance_id, color in self.colormap.items():
            if instance_id == 0 or instance_id == floor_id:  # Skip background and floor
                continue
            furniture_obj_mask = (semantic_map == instance_id)
            if np.any(furniture_obj_mask):
                # Convert RGB to BGR for OpenCV
                bgr_color = [color[2], color[1], color[0]] if len(color) == 3 else color
                colored_composite[furniture_obj_mask] = bgr_color

        # Set background color for areas with no floor plan or furniture
        background_mask = (composite_map == 0)
        colored_composite[background_mask] = [255, 255, 255]  # White background

        return composite_map, colored_composite

    def process_samples(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of samples into colored images and raw layout."""
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(1)

        raw_layout = batch[0]
        colored_batch = self.instance_map_to_color(batch)
        batch_tensor = torch.tensor(colored_batch.transpose(0, 3, 1, 2)).to(torch.uint8)

        return batch_tensor.permute(0, 2, 3, 1), raw_layout

    def create_sample_grid(self, batch: torch.Tensor) -> torch.Tensor:
        """Create a grid of samples for visualization."""
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(1)

        colored_batch = self.instance_map_to_color(batch)
        batch_tensor = torch.tensor(colored_batch.transpose(0, 3, 1, 2)).to(torch.uint8)
        grid = torchvision.utils.make_grid(batch_tensor, nrow=5, padding=2, normalize=False)

        return grid.permute(1, 2, 0)

    @staticmethod
    def create_animation_sequence(images: List, num_steps: int = 150, repeat_last: int = 10) -> List:
        """Create a smooth animation sequence from a chain of images."""
        out = []
        for i in np.linspace(0, len(images) - 1, num_steps):
            idx = int(i) if int(i) < len(images) else len(images) - 1
            out.append(images[idx])
        out.extend([images[-1]] * repeat_last)
        return out


class FloorPlanLoader:
    """Handles loading of floor plans from various sources."""

    def __init__(self, cfg: DictConfig, args):
        self.cfg = cfg
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_custom_floor_plan(self, floor_plan_path: str, target_size: Tuple[int, int] = (120, 120)) -> torch.Tensor:
        """Load and preprocess a custom floor plan image."""
        floor_plan = cv2.imread(floor_plan_path, cv2.IMREAD_GRAYSCALE)
        if floor_plan is None:
            raise ValueError(f"Could not load floor plan from {floor_plan_path}")

        if floor_plan.shape != target_size:
            floor_plan = cv2.resize(floor_plan, target_size, interpolation=cv2.INTER_NEAREST)

        if self.args.w_arch:
            floor_plan_classes = self._process_architectural_floor_plan(floor_plan)
        else:
            floor_plan_classes = (floor_plan > 127).astype(np.int64)

        return torch.tensor(floor_plan_classes).unsqueeze(0).unsqueeze(0).to(self.device)

    def _process_architectural_floor_plan(self, floor_plan: np.ndarray) -> np.ndarray:
        """Process floor plan with architectural elements."""
        floor_plan_classes = np.zeros_like(floor_plan, dtype=np.int64)
        floor_plan_classes[floor_plan == 0] = 0      # Background
        floor_plan_classes[floor_plan == 85] = 1     # Floor
        floor_plan_classes[floor_plan == 170] = 2    # Door
        floor_plan_classes[floor_plan == 255] = 3    # Window

        # Handle intermediate values
        floor_plan_classes[(floor_plan > 0) & (floor_plan < 85)] = 1
        floor_plan_classes[(floor_plan >= 85) & (floor_plan < 170)] = 1
        floor_plan_classes[(floor_plan >= 170) & (floor_plan < 255)] = 2

        return floor_plan_classes

    def load_original_floor_plans(self) -> List[torch.Tensor]:
        """Load floor plans from the original Front3D dataset."""
        floor_plans = []

        if self.cfg.w_arch:
            floor_plans = self._load_arch_floor_plans()
        elif self.args.floor_plan:
            floor_plans = self._load_regular_floor_plans()

        return floor_plans

    def _load_arch_floor_plans(self) -> List[torch.Tensor]:
        """Load architectural floor plans from Front3D."""
        front3d_fast = Front3DFast(
            root="datasets", 
            split="unified_w_arch", 
            floor_plan=self.args.floor_plan, 
            resolution=(self.args.data_size, self.args.data_size),  
            w_arch=self.cfg.w_arch, 
            room_type_condition=self.cfg.room_type_condition, 
            text_condition=self.cfg.text_condition,
        )

        floor_plans = front3d_fast.get_arch_floor_plan(self.cfg.sample_room_type)

        # Augment data if needed
        if len(floor_plans) < 1000:
            num_needed = 1000 - len(floor_plans)
            random_indices = np.random.choice(len(floor_plans), size=num_needed, replace=True)
            floor_plans.extend([floor_plans[idx] for idx in random_indices])

        return floor_plans

    def _load_regular_floor_plans(self) -> List[torch.Tensor]:
        """Load regular floor plans from dataset."""
        room_type_map = {0: "bed", 1: "living", 2: "dining"}

        if self.args.room_type_condition:
            room_type = room_type_map[self.cfg.sample_room_type]
        else:
            room_type = self.args.specific_room_type.replace("room", "")

        floor_plan_dir = os.path.join(self.cfg.floor_plan_dir, room_type)
        floor_plan_paths = sorted(os.listdir(floor_plan_dir))

        floor_plans = []
        for i in range(1000):
            floor_plan_path = os.path.join(floor_plan_dir, floor_plan_paths[i])
            floor_plan = cv2.imread(floor_plan_path, cv2.IMREAD_GRAYSCALE)
            floor_plan_tensor = torch.tensor(floor_plan / 255).unsqueeze(0).unsqueeze(0).to(self.device).to(torch.int64)
            floor_plans.append(floor_plan_tensor)

        return floor_plans


class TextEmbeddingProcessor:
    """Handles text embedding processing for layout generation."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_text_embeddings(self) -> Tuple[List[str], List[torch.Tensor]]:
        """Load text descriptions and embeddings."""
        text_data_dir = self.cfg.get('text_data_dir', "../datasets/output/unified_w_arch_3dfront_bbox_V1/train")
        text_descriptions = []
        text_embeddings = []

        room_dirs = [d for d in os.listdir(text_data_dir) if not d.endswith('.json')]

        for room_dir in room_dirs:
            text_file_path = os.path.join(text_data_dir, room_dir, "Updated_Bottom_inst_anno_description.json")

            with open(text_file_path, "r") as f:
                text_data = json.load(f)

            text_descriptions.append(text_data["description"])
            text_embeddings.append(torch.tensor(text_data["desc_emb"]).unsqueeze(0))

        return text_descriptions, text_embeddings

    def prepare_embeddings_for_sampling(self, text_embeddings: List[torch.Tensor], 
                                      start_idx: int, end_idx: int) -> torch.Tensor:
        """Prepare text embeddings for sampling."""
        selected_embeddings = text_embeddings[start_idx:end_idx]
        return torch.cat(selected_embeddings, dim=0).unsqueeze(1).to(self.device)


class LayoutSampler:
    """Main class for handling layout sampling operations."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize components
        self.model = None
        self.args = None
        self.colormap = None
        self.image_processor = None
        self.floor_plan_loader = None
        self.text_processor = None

        self._setup_paths()
        self._setup_model()
        self._setup_colormap()
        self._setup_processors()

    def _setup_paths(self):
        """Setup required paths."""
        self.model_args_path = os.path.join(self.cfg.model, 'args.pickle')
        self.checkpoint_path = os.path.join(self.cfg.model, 'check', 'checkpoint.pt')
        self.idx_to_label_path = os.path.join(
            self.cfg.idx_to_label_dir, 
            f"{self.cfg.room_type}_idx_to_generic_label.json"
        )

        # Ensure output directory exists
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    def _setup_model(self):
        """Load and setup the model."""
        # Load arguments
        with open(self.model_args_path, 'rb') as f:
            self.args = pickle.load(f)

        # Setup data loader to get data shape
        train_loader, _, data_shape, num_classes = get_data(self.args)

        # Create model
        self.model = get_model(self.args, data_shape=data_shape)

        if self.args.parallel == 'dp':
            self.model = DataParallelDistribution(self.model)

        # Load checkpoint
        if torch.cuda.is_available():
            checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        else:
            checkpoint = torch.load(self.checkpoint_path, weights_only=True, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.checkpoint_info = checkpoint
        print(f'Loaded model checkpoint from epoch {checkpoint["current_epoch"]}/{self.args.epochs}')

    def _setup_colormap(self):
        """Setup color mapping for visualization."""
        with open(self.cfg.color_palette_path, 'r') as f:
            color_palette = json.load(f)

        with open(self.idx_to_label_path, 'r') as f:
            idx_to_label = json.load(f)

        self.colormap = self._create_colormap(color_palette, idx_to_label)

    def _create_colormap(self, color_palette: Dict, idx_to_label: Dict) -> Dict[int, Tuple[int, int, int]]:
        """Create colormap from palette and labels."""
        colors = {}

        if self.cfg.wo_room:
            colors = self._create_colormap_without_room(color_palette, idx_to_label)
        else:
            for idx, label in idx_to_label.items():
                if label in color_palette:
                    colors[int(idx)] = tuple(color_palette[label])

        return colors

    def _create_colormap_without_room(self, color_palette: Dict, idx_to_label: Dict) -> Dict[int, Tuple[int, int, int]]:
        """Create colormap excluding room elements."""
        door_id = window_id = None

        # Find door and window IDs
        for idx, label in idx_to_label.items():
            if label.lower() == 'door':
                door_id = int(idx)
            elif label.lower() == 'window':
                window_id = int(idx)

        colors = {}
        for idx, label in idx_to_label.items():
            idx = int(idx)
            if label.lower() in ['door', 'window'] or label not in color_palette:
                continue

            adjusted_idx = idx
            if door_id and idx > door_id:
                adjusted_idx -= 1
            if window_id and idx > window_id:
                adjusted_idx -= 1

            colors[adjusted_idx] = tuple(color_palette[label])

        return colors

    def _setup_processors(self):
        """Setup helper processors."""
        self.image_processor = ImageProcessor(self.colormap)
        self.floor_plan_loader = FloorPlanLoader(self.cfg, self.args)
        self.text_processor = TextEmbeddingProcessor(self.cfg)

    def _find_floor_id_in_semantic_map(self, semantic_map: np.ndarray) -> int:
        """
        Find the ID that represents floor in the generated semantic map.

        Args:
            semantic_map: Generated semantic map

        Returns:
            int: The ID representing floor areas (usually 1, but can vary)
        """
        # Load the idx_to_label mapping to find floor ID
        try:
            with open(self.idx_to_label_path, 'r') as f:
                idx_to_label = json.load(f)

            # Look for floor-related labels
            floor_keywords = ['floor', 'ground', 'base']
            for idx, label in idx_to_label.items():
                if any(keyword in label.lower() for keyword in floor_keywords):
                    return int(idx)

            # Fallback: assume ID 1 is floor (common convention)
            return 1

        except Exception as e:
            print(f"Warning: Could not determine floor ID from labels, using default (1): {e}")
            return 1

    def sample_custom_floor_plans(self):
        """Sample layouts using custom floor plans."""
        print("Using custom floor plan data...")

        # Get custom floor plan files
        floor_plan_files = sorted(glob(os.path.join(self.cfg.custom_floor_dir, "*.png")))

        if not floor_plan_files:
            raise ValueError(f"No PNG files found in {self.cfg.custom_floor_dir}")

        print(f"Found {len(floor_plan_files)} custom floor plans")

        samples_per_floor = self.cfg.get('samples_per_floor', self.cfg.samples)

        # Load text embeddings if needed
        text_embedding = None
        if self.args.text_condition:
            _, text_embeddings = self.text_processor.load_text_embeddings()
            text_embedding = self.text_processor.prepare_embeddings_for_sampling(
                text_embeddings, 0, samples_per_floor
            )

        # Process each floor plan
        for floor_idx, floor_plan_path in enumerate(floor_plan_files):
            self._process_single_custom_floor_plan(
                floor_plan_path, floor_idx, len(floor_plan_files), 
                samples_per_floor, text_embedding
            )

        print(f"Completed sampling for all {len(floor_plan_files)} floor plans!")

    def _process_single_custom_floor_plan(self, floor_plan_path: str, floor_idx: int, 
                                        total_floors: int, samples_per_floor: int, 
                                        text_embedding: Optional[torch.Tensor]):
        """Process a single custom floor plan."""
        floor_name = os.path.splitext(os.path.basename(floor_plan_path))[0]
        print(f"Processing floor plan {floor_idx + 1}/{total_floors}: {floor_name}")

        floor_output_dir = os.path.join(self.cfg.out_dir, floor_name)
        os.makedirs(floor_output_dir, exist_ok=True)

        try:
            # Load floor plan
            floor_plan = self.floor_plan_loader.load_custom_floor_plan(floor_plan_path)
            print(f"  Loaded floor plan with shape: {floor_plan.shape}")

            # Generate samples
            sampling_time = self._generate_and_save_samples(
                floor_plan, samples_per_floor, floor_output_dir, text_embedding
            )

            # Save metadata
            self._save_metadata(floor_output_dir, {
                "floor_plan_source": floor_plan_path,
                "floor_plan_name": floor_name,
                "num_samples": samples_per_floor,
                "sampling_time": sampling_time,
                "model_checkpoint_epoch": self.checkpoint_info['current_epoch'],
                "saves_composite_maps": self.cfg.get('save_composite_maps', True),
                "with_architectural_elements": self.cfg.get('w_arch', False),
                "output_files_per_sample": {
                    "furniture_only_colored": "sample_XXX.png",
                    "furniture_only_labels": "sample_XXX_label.png", 
                    "floor_plan_visualization": "sample_XXX_floor_plan.png",
                    "composite_labels": "sample_XXX_composite_label.png" if self.cfg.get('save_composite_maps', True) else None,
                    "composite_colored": "sample_XXX_composite_colored.png" if self.cfg.get('save_composite_maps', True) else None
                }
            })

            print(f"  Completed processing {floor_name}")

        except Exception as e:
            print(f"  Error processing floor plan {floor_plan_path}: {e}")

    def _generate_and_save_samples(self, floor_plan: torch.Tensor, num_samples: int, 
                                 output_dir: str, text_embedding: Optional[torch.Tensor]) -> float:
        """Generate samples and save results."""
        print(f"  Generating {num_samples} samples...")

        # Prepare batch
        floor_plans_batch = floor_plan.repeat(num_samples, 1, 1, 1)

        # Apply condition modifications
        if self.cfg.mixed_condition:
            if self.cfg.condition_type == "uncon":
                floor_plans_batch[floor_plans_batch != 0] = 0
            elif self.cfg.condition_type == "floor":
                floor_plans_batch[floor_plans_batch != 0] = 1

        # Prepare conditions
        room_type_batch = self._prepare_room_type_condition(num_samples)
        mixed_condition_batch = self._prepare_mixed_condition(num_samples)

        # Sample layouts
        start_time = time.time()
        with torch.no_grad():
            samples_chain = self.model.sample_chain(
                num_samples, 
                floor_plan=floor_plans_batch,
                room_type=room_type_batch, 
                text_condition=text_embedding, 
                mixed_condition_id=mixed_condition_batch
            )

        sampling_time = time.time() - start_time
        print(f"  Sampling completed in {sampling_time:.2f} seconds")

        # Save results
        self._save_sample_results(samples_chain, output_dir, floor_plans_batch, num_samples, 
                                is_custom_floor_plan=True)

        return sampling_time

    def _prepare_room_type_condition(self, batch_size: int) -> Optional[torch.Tensor]:
        """Prepare room type condition tensor."""
        if not self.cfg.room_type_condition:
            return None

        room_type_tensor = torch.tensor(self.cfg.sample_room_type).to(self.device)
        return room_type_tensor.unsqueeze(0).repeat(batch_size)

    def _prepare_mixed_condition(self, batch_size: int) -> Optional[torch.Tensor]:
        """Prepare mixed condition tensor."""
        if not self.cfg.mixed_condition:
            return None

        condition_map = {"uncon": 0, "floor": 1, "arch": 2}
        condition_id = torch.tensor(condition_map[self.cfg.condition_type]).to(self.device)
        return condition_id.unsqueeze(0).repeat(batch_size)

    def _save_sample_results(self, samples_chain: torch.Tensor, output_dir: str, 
                           floor_plans_batch: torch.Tensor, num_samples: int, 
                           is_custom_floor_plan: bool = False):
        """Save the generated sample results."""
        samples_chain = samples_chain.permute(1, 0, 2, 3, 4)

        for sample_idx, samples_i in enumerate(samples_chain):
            file_idx = sample_idx + self.cfg.seed * num_samples

            # Process sample
            images, raw_layout = self.image_processor.process_samples(samples_i)
            images = list(reversed(images))
            images = self.image_processor.create_animation_sequence(images)

            # Save colored layout (furniture only)
            output_path = os.path.join(output_dir, f"sample_{file_idx:03d}.png")
            imageio.imsave(output_path, images[-1])

            # Save raw layout
            raw_layout = raw_layout.squeeze().numpy()
            raw_output_path = os.path.join(output_dir, f"sample_{file_idx:03d}_label.png")
            cv2.imwrite(raw_output_path, raw_layout)

            # Save floor plan
            self._save_floor_plan(floor_plans_batch[sample_idx], output_dir, file_idx)

            # For custom floor plans, create and save composite semantic map
            if is_custom_floor_plan and self.cfg.get('save_composite_maps', True):
                floor_plan_array = floor_plans_batch[sample_idx].squeeze().cpu().numpy()

                # Find the floor ID from the generated semantic map
                floor_id = self._find_floor_id_in_semantic_map(raw_layout)

                composite_map, colored_composite = self.image_processor.create_composite_semantic_map(
                    raw_layout, floor_plan_array, 
                    with_arch=self.cfg.get('w_arch', False),
                    floor_id=floor_id
                )

                # Save composite semantic map (raw)
                composite_raw_path = os.path.join(output_dir, f"sample_{file_idx:03d}_composite_label.png")
                cv2.imwrite(composite_raw_path, composite_map)

                # Save colored composite map
                composite_colored_path = os.path.join(output_dir, f"sample_{file_idx:03d}_composite_colored.png")
                cv2.imwrite(composite_colored_path, colored_composite)

        print(f"  Saved {num_samples} samples to {output_dir}")
        if is_custom_floor_plan and self.cfg.get('save_composite_maps', True):
            print(f"  Generated composite maps combining floor plans with furniture layouts")

    def _save_floor_plan(self, floor_plan: torch.Tensor, output_dir: str, file_idx: int):
        """Save floor plan visualization."""
        floor_plan_array = floor_plan.squeeze().cpu().numpy()

        if not self.cfg.w_arch:
            # Simple binary floor plan
            floor_plan_path = os.path.join(output_dir, f"sample_{file_idx:03d}_floor_plan.png")
            cv2.imwrite(floor_plan_path, floor_plan_array * 255)
        else:
            # Architectural floor plan with colors
            colored_floor_plan = self._create_colored_floor_plan(floor_plan_array)
            floor_plan_path = os.path.join(output_dir, f"sample_{file_idx:03d}_floor_plan.png")
            cv2.imwrite(floor_plan_path, colored_floor_plan)

    def _create_colored_floor_plan(self, floor_plan: np.ndarray) -> np.ndarray:
        """Create colored visualization of architectural floor plan."""
        colored_floor_plan = np.zeros((120, 120, 3), dtype=np.uint8)

        # Color mapping (BGR format for OpenCV)
        colored_floor_plan[floor_plan == 0] = [255, 255, 255]  # Background -> White
        colored_floor_plan[floor_plan == 1] = [211, 211, 211]  # Floor -> Gray
        colored_floor_plan[floor_plan == 2] = [0, 0, 153]      # Door -> Dark Red
        colored_floor_plan[floor_plan == 3] = [153, 153, 255]  # Window -> Light Red

        return colored_floor_plan

    def sample_original_data(self):
        """Sample layouts using original Front3D data."""
        print("Using original Front3D data...")

        # Load floor plans
        floor_plans = self.floor_plan_loader.load_original_floor_plans()

        # Apply condition modifications
        self._apply_condition_modifications(floor_plans)

        # Prepare text embeddings
        text_embedding = None
        if self.args.text_condition:
            text_descriptions, text_embeddings = self.text_processor.load_text_embeddings()
            start_idx = self.cfg.seed * self.cfg.samples
            end_idx = (self.cfg.seed + 1) * self.cfg.samples
            text_embedding = self.text_processor.prepare_embeddings_for_sampling(
                text_embeddings, start_idx, end_idx
            )

            # Save text descriptions
            self._save_text_descriptions(text_descriptions[start_idx:end_idx])

        # Generate samples
        self._generate_original_data_samples(floor_plans, text_embedding)

    def _apply_condition_modifications(self, floor_plans: List[torch.Tensor]):
        """Apply condition modifications to floor plans."""
        if not self.cfg.mixed_condition:
            return

        if self.cfg.condition_type == "uncon":
            for floor_plan in floor_plans:
                floor_plan[floor_plan != 0] = 0
        elif self.cfg.condition_type == "floor":
            for floor_plan in floor_plans:
                floor_plan[floor_plan != 0] = 1

    def _save_text_descriptions(self, text_descriptions: List[str]):
        """Save text descriptions to JSON file."""
        output_path = os.path.join(self.cfg.out_dir, "text_description.json")
        with open(output_path, "w") as f:
            json.dump(text_descriptions, f, indent=2)

    def _generate_original_data_samples(self, floor_plans: List[torch.Tensor], 
                                      text_embedding: Optional[torch.Tensor]):
        """Generate samples for original data mode."""
        print("Starting sample generation...")
        start_time = time.time()

        with torch.no_grad():
            if self.args.floor_plan:
                # Prepare floor plans
                start_idx = self.cfg.seed * self.cfg.samples
                end_idx = (self.cfg.seed + 1) * self.cfg.samples
                selected_floor_plans = floor_plans[start_idx:end_idx]

                if self.cfg.w_arch:
                    concatenated_floor_plans = torch.cat(selected_floor_plans, dim=0).unsqueeze(1).to(self.device)
                else:
                    concatenated_floor_plans = torch.cat(selected_floor_plans, dim=0).to(self.device)

                # Generate samples
                samples_chain = self.model.sample_chain(
                    self.cfg.samples, 
                    floor_plan=concatenated_floor_plans,
                    room_type=self._prepare_room_type_condition(1), 
                    text_condition=text_embedding, 
                    mixed_condition_id=self._prepare_mixed_condition(1)
                )
            else:
                # Generate without floor plans
                samples_chain = self.model.sample_chain(
                    self.cfg.samples, 
                    floor_plan=None, 
                    room_type=self._prepare_room_type_condition(1)
                )

        sampling_time = time.time() - start_time
        print(f"Sampling completed in {sampling_time:.2f} seconds")

        # Save results
        if self.cfg.save_grid:
            self._save_grid_results(samples_chain)
        else:
            floor_plans_for_saving = concatenated_floor_plans if self.args.floor_plan else None
            self._save_individual_results(samples_chain, floor_plans_for_saving, is_custom_floor_plan=False)

    def _save_grid_results(self, samples_chain: torch.Tensor):
        """Save results as a grid."""
        samples_list = [samples_chain[:, i] for i in range(samples_chain.shape[1])]
        images = []

        for samples_i in samples_list:
            grid = self.image_processor.create_sample_grid(samples_i)
            images.append(grid)

        images = list(reversed(images))
        images = self.image_processor.create_animation_sequence(images)

        # Save files
        output_path = os.path.join(self.cfg.out_dir, f"sample_{self.cfg.room_type}.png")
        imageio.mimsave(output_path[:-4] + '_chain.gif', images)
        imageio.imsave(output_path, images[-1])

    def _save_individual_results(self, samples_chain: torch.Tensor, 
                               floor_plans: Optional[torch.Tensor],
                               is_custom_floor_plan: bool = False):
        """Save individual sample results."""
        samples_chain = samples_chain.permute(1, 0, 2, 3, 4)

        for idx, samples_i in enumerate(samples_chain):
            file_idx = idx + self.cfg.seed * self.cfg.samples

            # Process samples
            images, raw_layout = self.image_processor.process_samples(samples_i)
            images = list(reversed(images))
            images = self.image_processor.create_animation_sequence(images)

            # Save files
            base_path = os.path.join(self.cfg.out_dir, f"sample_{self.cfg.room_type}")

            # Save colored layout
            imageio.imsave(f"{base_path}-{file_idx}.png", images[-1])

            # Save raw layout
            raw_layout = raw_layout.squeeze().numpy()
            cv2.imwrite(f"{base_path}_label-{file_idx}.png", raw_layout)

            # Save floor plan if available
            if floor_plans is not None:
                floor_plan = floor_plans[idx].squeeze().cpu().numpy()

                if not self.cfg.w_arch:
                    cv2.imwrite(f"{base_path}_floor_plan-{file_idx}.png", floor_plan * 255)
                else:
                    colored_floor_plan = self._create_colored_floor_plan(floor_plan)
                    cv2.imwrite(f"{base_path}_floor_plan-{file_idx}.png", colored_floor_plan)

                # For original data with custom composite mapping option
                if is_custom_floor_plan and self.cfg.get('save_composite_maps', True):
                    # Find the floor ID from the generated semantic map
                    floor_id = self._find_floor_id_in_semantic_map(raw_layout)

                    composite_map, colored_composite = self.image_processor.create_composite_semantic_map(
                        raw_layout, floor_plan, 
                        with_arch=self.cfg.get('w_arch', False),
                        floor_id=floor_id
                    )

                    # Save composite maps
                    cv2.imwrite(f"{base_path}_composite_label-{file_idx}.png", composite_map)
                    cv2.imwrite(f"{base_path}_composite_colored-{file_idx}.png", colored_composite)

    def _save_metadata(self, output_dir: str, metadata: Dict[str, Any]):
        """Save metadata to JSON file."""
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def run(self):
        """Main execution method."""
        torch.manual_seed(self.cfg.seed)

        # Print configuration
        self._print_configuration()

        # Run appropriate sampling mode
        if self.cfg.get('use_custom_data', False):
            self.sample_custom_floor_plans()
        else:
            self.sample_original_data()

        print(f"Results saved to: {self.cfg.out_dir}")

    def _print_configuration(self):
        """Print current configuration."""
        if self.cfg.get('use_custom_data', False):
            print("=== Custom Floor Plan Sampling ===")
            print(f"Floor plan directory: {self.cfg.custom_floor_dir}")
            print(f"Samples per floor plan: {self.cfg.get('samples_per_floor', self.cfg.samples)}")
            print(f"Generate composite maps: {self.cfg.get('save_composite_maps', True)}")
        else:
            print("=== Original Data Sampling ===")
            print(f"Room type: {self.cfg.room_type}")
            print(f"Number of samples: {self.cfg.samples}")

        print(f"Output directory: {self.cfg.out_dir}")
        print(f"Random seed: {self.cfg.seed}")
        print(f"Architectural elements: {self.cfg.get('w_arch', False)}")
