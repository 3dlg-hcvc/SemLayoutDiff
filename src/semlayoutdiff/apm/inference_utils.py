"""
Inference Utilities for Furniture Attributes Prediction Model

This module contains utility functions for processing semantic maps, floor plans,
and generating visualizations during inference.

Author: AI Assistant
License: MIT
"""

import os
import json
import shutil
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image


def load_json(file_path: str) -> Dict:
    """
    Load JSON data from file.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        Dict: Loaded JSON data

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is invalid
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")


def assign_color_palette(color_palette_path: str, idx_to_label_path: str) -> Dict[int, Tuple[int, int, int]]:
    """
    Create color mapping for semantic categories based on color palette.

    Args:
        color_palette_path (str): Path to color palette JSON file
        idx_to_label_path (str): Path to index-to-label mapping JSON file

    Returns:
        Dict[int, Tuple[int, int, int]]: Mapping from category index to BGR color tuple
    """
    color_palette = load_json(color_palette_path)
    idx_to_label = load_json(idx_to_label_path)

    colors_bgr = {}
    for idx, label in idx_to_label.items():
        if label in color_palette:
            # Convert RGB to BGR for OpenCV
            rgb_color = color_palette[label]
            colors_bgr[int(idx)] = tuple(rgb_color[::-1])

    return colors_bgr


def scale_semantic_map(semantic_map_path: str, target_width: int, target_height: int) -> Optional[np.ndarray]:
    """
    Scale semantic map to target dimensions while preserving semantic labels.

    Args:
        semantic_map_path (str): Path to input semantic map
        target_width (int): Target width for scaling
        target_height (int): Target height for scaling

    Returns:
        Optional[np.ndarray]: Scaled semantic map array or None if loading fails
    """
    semantic_map = cv2.imread(semantic_map_path, cv2.IMREAD_GRAYSCALE)

    if semantic_map is None:
        return None

    # Use nearest neighbor interpolation to preserve semantic labels
    scaled_map = cv2.resize(
        semantic_map, 
        (target_width, target_height), 
        interpolation=cv2.INTER_NEAREST
    )

    return scaled_map


def generate_colored_visualization(semantic_map: np.ndarray, 
                                 color_palette_path: str, 
                                 idx_to_label_path: str) -> Optional[np.ndarray]:
    """
    Generate colored visualization of semantic map.

    Args:
        semantic_map (np.ndarray): Semantic map array
        color_palette_path (str): Path to color palette JSON file
        idx_to_label_path (str): Path to index-to-label mapping JSON file

    Returns:
        Optional[np.ndarray]: Colored map array or None if generation fails
    """
    try:
        colormap = assign_color_palette(color_palette_path, idx_to_label_path)
        height, width = semantic_map.shape
        colored_map = np.zeros((height, width, 3), dtype=np.uint8)

        for category_id, color in colormap.items():
            mask = (semantic_map == category_id)
            colored_map[mask] = color

        return colored_map
    except Exception:
        return None


def process_semantic_map(semantic_map_path: str, 
                        target_width: int, 
                        target_height: int,
                        color_palette_path: Optional[str] = None,
                        idx_to_label_path: Optional[str] = None,
                        output_dir: Optional[str] = None,
                        sample_id: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    Process semantic map with scaling and optional colored visualization.

    Args:
        semantic_map_path (str): Path to input semantic map
        target_width (int): Target width for scaling
        target_height (int): Target height for scaling
        color_palette_path (Optional[str]): Path to color palette JSON
        idx_to_label_path (Optional[str]): Path to index-to-label mapping JSON
        output_dir (Optional[str]): Directory to save outputs
        sample_id (Optional[str]): Sample ID for output filenames

    Returns:
        Tuple containing:
            - scaled_semantic_map: NumPy array of scaled semantic map
            - scaled_map_path: Path to saved scaled semantic map
            - colored_map_path: Path to saved colored visualization
    """
    # Scale semantic map
    scaled_map = scale_semantic_map(semantic_map_path, target_width, target_height)
    if scaled_map is None:
        return None, None, None

    scaled_map_path = None
    colored_map_path = None

    if output_dir and sample_id:
        os.makedirs(output_dir, exist_ok=True)

        # Save scaled semantic map
        scaled_map_path = os.path.join(output_dir, f"{sample_id}_scaled_label.png")
        cv2.imwrite(scaled_map_path, scaled_map)

        # Generate and save colored visualization if palette is provided
        if color_palette_path and idx_to_label_path:
            colored_map = generate_colored_visualization(scaled_map, color_palette_path, idx_to_label_path)
            if colored_map is not None:
                colored_map_path = os.path.join(output_dir, f"{sample_id}_colored.png")
                cv2.imwrite(colored_map_path, colored_map)

    return scaled_map, scaled_map_path, colored_map_path


def convert_floor_plan_format(floor_plan_path: str, 
                            output_path: str, 
                            target_width: int = 1200, 
                            target_height: int = 1200) -> bool:
    """
    Convert and scale floor plan to standard architectural mask format.

    Maps pixel values to architectural elements:
    - 0: void/background
    - 1: floor
    - 2: door
    - 3: window

    Args:
        floor_plan_path (str): Path to input floor plan
        output_path (str): Path to save converted floor plan
        target_width (int): Target width for scaling (default: 1200)
        target_height (int): Target height for scaling (default: 1200)

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        floor_plan = cv2.imread(floor_plan_path, cv2.IMREAD_GRAYSCALE)
        if floor_plan is None:
            return False

        # Scale to target size using nearest neighbor to preserve discrete values
        if floor_plan.shape != (target_height, target_width):
            floor_plan = cv2.resize(
                floor_plan, 
                (target_width, target_height), 
                interpolation=cv2.INTER_NEAREST
            )

        # Create architectural mask with standard values
        arch_mask = np.zeros_like(floor_plan, dtype=np.uint8)
        arch_mask[floor_plan == 255] = 0  # void/background
        arch_mask[floor_plan == 211] = 1  # floor
        arch_mask[floor_plan == 45] = 2   # door
        arch_mask[floor_plan == 183] = 3  # window

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, arch_mask)

        return True
    except Exception:
        return False


def discover_custom_samples(custom_samples_dir: str) -> List[str]:
    """
    Discover floor plan directories in custom samples directory.

    Args:
        custom_samples_dir (str): Directory containing custom sample subdirectories

    Returns:
        List[str]: List of floor plan directory names

    Raises:
        ValueError: If no valid floor plan directories are found
    """
    if not os.path.exists(custom_samples_dir):
        raise ValueError(f"Custom samples directory does not exist: {custom_samples_dir}")

    # Look for floor plan directories (old and new naming conventions)
    floor_plan_dirs = [
        d for d in os.listdir(custom_samples_dir) 
        if os.path.isdir(os.path.join(custom_samples_dir, d)) and 
        (d.startswith('floorplan_') or d.startswith('enhanced_center_'))
    ]

    if not floor_plan_dirs:
        raise ValueError(
            f"No valid floor plan directories found in {custom_samples_dir}. "
            "Expected directories starting with 'floorplan_' or 'enhanced_center_'"
        )

    return sorted(floor_plan_dirs)


def process_custom_samples(custom_samples_dir: str, 
                          output_dir: str, 
                          target_width: int = 1200, 
                          target_height: int = 1200) -> Tuple[List[str], List[Dict], str]:
    """
    Process custom floor plan samples and prepare them for inference.

    Args:
        custom_samples_dir (str): Directory containing custom sample subdirectories
        output_dir (str): Directory to save processed results
        target_width (int): Target width for scaling (default: 1200)
        target_height (int): Target height for scaling (default: 1200)

    Returns:
        Tuple containing:
            - semantic_map_paths: List of paths to semantic map files
            - floor_plan_info: List of dictionaries with floor plan information
            - scaled_semantic_dir: Path to scaled semantic maps directory
    """
    floor_plan_dirs = discover_custom_samples(custom_samples_dir)

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    converted_floor_dir = os.path.join(output_dir, "converted_floor_plans")
    scaled_semantic_dir = os.path.join(output_dir, "scaled_semantic_maps")
    os.makedirs(converted_floor_dir, exist_ok=True)
    os.makedirs(scaled_semantic_dir, exist_ok=True)

    all_semantic_maps = []
    all_floor_plan_info = []

    for floor_dir in floor_plan_dirs:
        floor_path = os.path.join(custom_samples_dir, floor_dir)

        # Find semantic maps and floor plans
        label_files = sorted([f for f in os.listdir(floor_path) if f.endswith('_label.png')])
        floor_plan_files = sorted([f for f in os.listdir(floor_path) if f.endswith('_floor_plan.png')])

        # Create subdirectory for this floor plan type
        floor_output_dir = os.path.join(converted_floor_dir, floor_dir)
        os.makedirs(floor_output_dir, exist_ok=True)

        for label_file, floor_file in zip(label_files, floor_plan_files):
            sample_id = label_file.replace('_label.png', '')

            # File paths
            semantic_map_path = os.path.join(floor_path, label_file)
            original_floor_plan_path = os.path.join(floor_path, floor_file)
            converted_floor_path = os.path.join(floor_output_dir, f"{sample_id}_floor_01.png")

            # Convert floor plan format
            if convert_floor_plan_format(original_floor_plan_path, converted_floor_path, 
                                       target_width, target_height):
                all_semantic_maps.append(semantic_map_path)

                floor_info = {
                    'original_path': original_floor_plan_path,
                    'converted_path': converted_floor_path,
                    'sample_id': sample_id,
                    'floor_type': floor_dir,
                    'scaled_semantic_dir': scaled_semantic_dir
                }
                all_floor_plan_info.append(floor_info)

    return all_semantic_maps, all_floor_plan_info, scaled_semantic_dir


def create_inference_summary(total_processed: int, 
                           output_paths: Dict[str, str], 
                           config_info: Dict[str, str], 
                           is_custom: bool = False) -> Dict:
    """
    Create a summary of the inference run.

    Args:
        total_processed (int): Number of samples processed
        output_paths (Dict[str, str]): Dictionary of output directory paths
        config_info (Dict[str, str]): Configuration information
        is_custom (bool): Whether this was custom inference

    Returns:
        Dict: Summary information
    """
    summary = {
        "inference_type": "custom" if is_custom else "original",
        "total_samples_processed": total_processed,
        "model_checkpoint": config_info.get("checkpoint_path", ""),
        "room_type": config_info.get("room_type", ""),
        "version": config_info.get("version", ""),
        "target_resolution": config_info.get("resolution", ""),
        "output_directories": output_paths,
        "timestamp": None  # Can be added if needed
    }

    if is_custom:
        summary.update({
            "scaling_enabled": True,
            "floor_plan_processing": True,
            "colored_visualizations": config_info.get("generate_colored", False)
        })

    return summary


def save_inference_summary(summary: Dict, output_path: str) -> None:
    """
    Save inference summary to JSON file.

    Args:
        summary (Dict): Summary information
        output_path (str): Path to save summary file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
