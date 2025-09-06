"""
Unified Data Processing Script for 3D-FRONT Dataset

This script processes 3D-FRONT dataset by:
1. First padding images to target size and converting masks to bounding boxes
2. Then unifying category IDs across different room types
3. Saving directly to unified output directory

Usage:
    python data_processor.py --room_types bedroom livingroom --input_dir ./data --output_dir ./unified_output
"""

import json
import cv2
import numpy as np
import os
import argparse
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any


class DataProcessor:
    """Main class for processing 3D-FRONT dataset images and annotations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize processor with configuration."""
        self.config = config
        self.mappings = {}
        self.color_palette = {}
        self.unified_mapping = {}
        self.room_idx_to_label = {}

        # Load configuration files
        self._load_mappings()

    def _load_mappings(self):
        """Load all mapping and configuration files."""
        try:
            # Load room-specific mappings
            if 'mapping_files' in self.config:
                for room_type, mapping_file in self.config['mapping_files'].items():
                    with open(mapping_file, 'r') as f:
                        self.mappings[room_type] = json.load(f)

            # Load unified mapping
            if 'unified_mapping_file' in self.config:
                with open(self.config['unified_mapping_file'], 'r') as f:
                    self.unified_mapping = json.load(f)

            # Load color palette
            if 'color_palette_file' in self.config:
                with open(self.config['color_palette_file'], 'r') as f:
                    self.color_palette = json.load(f)

            # Load room-specific idx to label mappings
            if 'idx_to_label_files' in self.config:
                for room_type, idx_file in self.config['idx_to_label_files'].items():
                    with open(idx_file, 'r') as f:
                        self.room_idx_to_label[room_type] = json.load(f)

        except FileNotFoundError as e:
            print(f"Warning: Could not load mapping file: {e}")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in mapping file: {e}")
            raise

    def mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Convert a mask to its bounding box."""
        coords = np.nonzero(mask)
        if len(coords[0]) == 0:  # Empty mask
            return mask

        min_x, max_x = np.min(coords[0]), np.max(coords[0])
        min_y, max_y = np.min(coords[1]), np.max(coords[1])

        bbox_mask = np.zeros_like(mask)
        bbox_mask[min_x:max_x + 1, min_y:max_y + 1] = 1

        return bbox_mask

    def _convert_to_bboxes(self, label_image: np.ndarray, room_type: str) -> np.ndarray:
        """Convert object masks to bounding boxes."""
        if room_type not in self.room_idx_to_label:
            return label_image

        idx_to_label = self.room_idx_to_label[room_type]
        unique_labels = np.unique(label_image)

        # Get lamp labels and floor label
        lamp_labels = [
            int(idx) for idx, label in idx_to_label.items() 
            if "lamp" in label.lower()
        ]
        floor_labels = [
            int(idx) for idx, label in idx_to_label.items() 
            if label.lower() == "floor"
        ]
        floor_label = floor_labels[0] if floor_labels else None

        other_labels = [
            label for label in unique_labels 
            if label not in lamp_labels and label != 0 and label != floor_label
        ]

        original_label_image = label_image.copy()

        # Process other objects
        for label in other_labels:
            object_mask = (original_label_image == label).astype(int)
            labeled_mask, num_objects = scipy.ndimage.label(object_mask)

            for i in range(1, num_objects + 1):
                single_object_mask = (labeled_mask == i).astype(int)
                bbox_mask = self.mask_to_bbox(single_object_mask)
                label_image = np.where(bbox_mask == 1, label, label_image)

        # Process lamp objects separately
        for label in lamp_labels:
            if label in unique_labels and label != 0 and label != floor_label:
                object_mask = (original_label_image == label).astype(int)
                labeled_mask, num_objects = scipy.ndimage.label(object_mask)

                for i in range(1, num_objects + 1):
                    single_object_mask = (labeled_mask == i).astype(int)
                    bbox_mask = self.mask_to_bbox(single_object_mask)
                    label_image = np.where(bbox_mask == 1, label, label_image)

        return label_image

    def pad_image_and_update_annotation(
        self, 
        label_image: np.ndarray,
        color_image: np.ndarray, 
        annotations: List[Dict],
        room_type: str,
        target_size: Tuple[int, int] = (1200, 1200)
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Pad image to target size and update annotations accordingly. Always converts masks to bounding boxes."""

        original_height, original_width = label_image.shape[:2]

        # Calculate padding
        pad_width = target_size[0] - original_width
        pad_height = target_size[1] - original_height

        pad_left = max(0, pad_width // 2)
        pad_right = max(0, pad_width - pad_left)
        pad_top = max(0, pad_height // 2)
        pad_bottom = max(0, pad_height - pad_top)

        # Always convert masks to bounding boxes
        label_image = self._convert_to_bboxes(label_image, room_type)

        # Pad images
        padded_label_image = np.pad(
            label_image, 
            ((pad_top, pad_bottom), (pad_left, pad_right)), 
            mode="constant"
        )
        padded_color_image = np.pad(
            color_image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant"
        )

        # Update annotations
        updated_annotations = []
        for annotation in annotations:
            updated_mask = []
            for polygon in annotation.get("mask", []):
                updated_polygon = []
                for i in range(0, len(polygon), 2):
                    x, y = polygon[i], polygon[i + 1]
                    updated_polygon.extend([x + pad_left, y + pad_top])
                updated_mask.append(updated_polygon)

            annotation_copy = annotation.copy()
            annotation_copy["mask"] = updated_mask
            updated_annotations.append(annotation_copy)

        return padded_label_image, padded_color_image, updated_annotations

    def update_image_ids(self, image: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Update object IDs in an image using the provided mapping."""
        return np.vectorize(mapping.get)(image)

    def update_image_colors(self, image: np.ndarray) -> np.ndarray:
        """Update colors in an image based on unified category mappings."""
        colored_image = np.zeros((*image.shape, 3), dtype=np.uint8)

        for category_id in np.unique(image):
            if category_id == 0:  # Skip background
                continue

            # Find category name from unified mapping
            category_names = [k for k, v in self.unified_mapping.items() if v == int(category_id)]
            if not category_names:
                continue

            category_name = category_names[0]
            color = self.color_palette.get(category_name, [0, 0, 0])
            mask = (image == category_id)
            colored_image[mask] = color  # Keep RGB order, don't convert to BGR

        return colored_image

    def process_and_unify_datasets(
        self, 
        input_dir: str, 
        output_dir: str, 
        room_types: List[str],
        target_size: Tuple[int, int] = (1200, 1200),
        use_bbox: bool = False,
        include_arch: bool = True
    ):
        """Process datasets with padding and unification in one step."""

        total_scenes = 0

        for room_type in room_types:
            print(f"Processing {room_type}...")

            if room_type not in self.mappings:
                print(f"Warning: No mapping found for {room_type}, skipping...")
                continue

            # Get ID mapping for this room type
            mapping = {int(k): v for k, v in self.mappings[room_type].items()}

            # Load data splits for this room type
            splits = self._load_data_splits(input_dir, room_type)

            # Find the room data directory - try multiple patterns
            room_data_dir = None
            possible_patterns = [
                f"{room_type}_w_arch_V1",
                f"filtered_3dfront_data_{room_type}_V1", 
                f"filtered_3dfront_w_arch_{room_type}"
            ]

            for pattern in possible_patterns:
                candidate_dir = os.path.join(input_dir, pattern)
                if os.path.exists(candidate_dir):
                    room_data_dir = candidate_dir
                    print(f"Found room data in: {pattern}")
                    break

            if not room_data_dir:
                print(f"Warning: No room data directory found for {room_type}")
                continue

            # Create output split directories
            for split_name in ['train', 'val', 'test']:
                split_output_dir = os.path.join(output_dir, split_name)
                os.makedirs(split_output_dir, exist_ok=True)

            # Process all rooms based on their split assignment
            all_room_ids = []
            for split_name, room_ids in splits.items():
                all_room_ids.extend(room_ids)

            for room_id in tqdm(all_room_ids, desc=f"{room_type}"):
                try:
                    # Determine which split this room belongs to
                    target_split = None
                    for split_name, room_ids in splits.items():
                        if room_id in room_ids:
                            target_split = split_name
                            break

                    if not target_split:
                        continue

                    # Find room folder in the flat directory structure
                    room_folder = os.path.join(room_data_dir, room_id)
                    if not os.path.isdir(room_folder):
                        continue

                    # Get target split output directory
                    split_output_dir = os.path.join(output_dir, target_split)

                    # Process this room (7 arguments plus self = 8 total)
                    self._process_single_room(
                        room_folder, split_output_dir, room_id, room_type,
                        mapping, target_size, include_arch
                    )
                    total_scenes += 1

                except Exception as e:
                    print(f"Error processing room {room_id}: {e}")
                    continue

        print(f"Total scenes processed: {total_scenes}")
        self._print_split_statistics(output_dir)

    def _process_single_room(
        self,
        room_folder: str,
        split_output_dir: str,
        room_id: str,
        room_type: str,
        mapping: Dict[int, int],
        target_size: Tuple[int, int],
        include_arch: bool
    ):
        """Process a single room with padding and unification. Always converts masks to bounding boxes."""

        # Define possible file names (try with and without 'Updated_' prefix)
        possible_prefixes = ["", "Updated_"]

        label_image_path = None
        color_image_path = None
        annotation_path = None

        # Find the correct files
        for prefix in possible_prefixes:
            label_path = os.path.join(room_folder, f"{prefix}Bottom_label_map.png")
            color_path = os.path.join(room_folder, f"{prefix}Bottom_color.png")
            anno_path = os.path.join(room_folder, f"{prefix}Bottom_inst_anno.json")

            if all(os.path.exists(p) for p in [label_path, color_path, anno_path]):
                label_image_path = label_path
                color_image_path = color_path
                annotation_path = anno_path
                break

        if not all([label_image_path, color_image_path, annotation_path]):
            raise FileNotFoundError(f"Required files not found in {room_folder}")

        # Load data
        label_image = np.array(Image.open(label_image_path))
        color_image = np.array(Image.open(color_image_path))

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        # Step 1: Pad images and update annotations (with bbox conversion)
        padded_label_image, padded_color_image, updated_annotations = \
            self.pad_image_and_update_annotation(
                label_image, color_image, annotations, room_type, target_size
            )

        # Step 2: Apply ID mapping to unified categories
        unified_label_image = self.update_image_ids(padded_label_image, mapping)

        # Step 3: Update annotation categories
        for annotation in updated_annotations:
            old_category = annotation.get('category')
            new_category = mapping.get(old_category, old_category)
            annotation['category'] = new_category

        # Step 4: Generate colored image
        unified_colored_image = np.zeros(
            (unified_label_image.shape[0], unified_label_image.shape[1], 3), dtype=np.uint8
        )

        # Use the unified mapping to get colors
        unique_labels = np.unique(unified_label_image)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            # Find category name from unified mapping
            category_names = [k for k, v in self.unified_mapping.items() if v == int(label)]
            if not category_names:
                continue

            category_name = category_names[0]
            color = self.color_palette.get(category_name, [0, 0, 0])
            unified_colored_image[unified_label_image == label] = color

        # Step 5: Handle architectural elements
        if not include_arch:
            # Find architectural elements by looking at category names
            for label in unique_labels:
                if label == 0:
                    continue

                category_names = [k for k, v in self.unified_mapping.items() if v == int(label)]
                if category_names:
                    category_name = category_names[0]
                    if category_name.lower() in ["door", "window"]:
                        unified_colored_image[unified_label_image == label] = [0, 0, 0]
                        unified_label_image[unified_label_image == label] = 0

            # Filter annotations
            updated_annotations = [
                ann for ann in updated_annotations
                if ann.get("basename", "").lower() not in ["window", "door"]
            ]

        # Step 6: Save results
        room_output_dir = os.path.join(split_output_dir, room_id)
        os.makedirs(room_output_dir, exist_ok=True)

        # Save files
        Image.fromarray(unified_label_image.astype(np.uint8)).save(
            os.path.join(room_output_dir, "Updated_Bottom_label_map.png")
        )
        Image.fromarray(unified_colored_image).save(
            os.path.join(room_output_dir, "Updated_Bottom_color.png")
        )

        with open(os.path.join(room_output_dir, "Updated_Bottom_inst_anno.json"), "w") as f:
            json.dump(updated_annotations, f, indent=4)

    def _load_data_splits(self, root_dir: str, room_type: str) -> Dict[str, List[str]]:
        """Load train/val/test splits from JSON files."""
        splits = {}
        split_files = {
            'train': f"train_{room_type}_ids.json",
            'val': f"val_{room_type}_ids.json", 
            'test': f"test_{room_type}_ids.json"
        }

        for split_name, filename in split_files.items():
            filepath = os.path.join(root_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    splits[split_name] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Split file not found: {filepath}")
                splits[split_name] = []

        return splits

    def _print_split_statistics(self, output_dir: str):
        """Print statistics about processed splits."""
        total_count = 0
        for split_name in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split_name)
            if os.path.exists(split_dir):
                count = len([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
                print(f"Scenes in {split_name} split: {count}")
                total_count += count
        print(f"Total scenes across all splits: {total_count}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process and unify 3D-FRONT dataset")

    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration JSON file"
    )

    # Input/Output directories
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Input directory containing the dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory for unified data"
    )

    # Room configuration
    parser.add_argument(
        "--room_types", 
        nargs='+',
        default=['bedroom', 'livingroom', 'diningroom'],
        help="Room types to process"
    )

    # Processing options
    parser.add_argument(
        "--target_size", 
        type=int, 
        nargs=2,
        default=[1200, 1200],
        help="Target size for padding (width height)"
    )
    parser.add_argument(
        "--include_arch", 
        action="store_true",
        help="Include architectural elements (doors/windows)"
    )

    # Mapping files
    parser.add_argument(
        "--unified_mapping_file", 
        type=str,
        help="Path to unified category mapping file"
    )
    parser.add_argument(
        "--color_palette_file", 
        type=str,
        help="Path to color palette file"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Build configuration from command line arguments
        config = {
            'unified_mapping_file': args.unified_mapping_file,
            'color_palette_file': args.color_palette_file,
            'mapping_files': {},
            'idx_to_label_files': {}
        }

        # Add room-specific mappings if available
        for room_type in args.room_types:
            mapping_file = f"./config/{room_type}_to_unified_mapping.json"
            idx_file = f"./config/{room_type}_idx_to_generic_label.json"

            if os.path.exists(mapping_file):
                config['mapping_files'][room_type] = mapping_file
            if os.path.exists(idx_file):
                config['idx_to_label_files'][room_type] = idx_file

    # Initialize processor
    processor = DataProcessor(config)

    # Process and unify datasets
    print("Starting data processing and unification (with bbox conversion)...")
    processor.process_and_unify_datasets(
        args.input_dir,
        args.output_dir, 
        args.room_types,
        tuple(args.target_size),
        args.include_arch
    )
    print("Processing completed!")


if __name__ == "__main__":
    main()
