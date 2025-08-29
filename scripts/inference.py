"""
Unified Inference Script for Furniture Attributes Prediction

This script supports both standard inference on existing datasets and custom inference
on user-provided floor plan samples. It automatically detects the inference mode based
on configuration parameters and handles all necessary preprocessing including semantic
map scaling, floor plan conversion, and colored visualization generation.

Usage:
    python scripts/inference.py [--config-path CONFIG_PATH] [--config-name CONFIG_NAME]

Features:
    - Standard inference for existing semantic maps
    - Custom inference with automatic scaling and floor plan processing
    - Colored visualization generation
    - Scene state export for 3D visualization
    - Comprehensive error handling and logging

Author: AI Assistant
License: MIT
"""

import os
import torch
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig
import numpy as np
from PIL import Image
import cv2
import json
import glob
import shutil
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from semlayoutdiff.apm.attr_module import FurnitureAttributesModel
from semlayoutdiff.apm.utils import mask_to_coco_polygon, convert_2d_to_3d, get_instance_masks, export_scenestate
from semlayoutdiff.apm.inference_utils import (
    load_json, 
    process_semantic_map, 
    process_custom_samples, 
    create_inference_summary, 
    save_inference_summary
)
from preprocess.threed_front.datasets.threed_future_dataset import ThreedFutureDataset


def load_model_and_datasets(cfg: DictConfig) -> Tuple[FurnitureAttributesModel, Dict, Dict, ThreedFutureDataset]:
    """Load and prepare model, configurations, and datasets."""
    # Load configuration files
    pix_ratio_threshold = load_json(cfg.pix_ratio_threshold)
    new_label_to_generic_label = load_json(cfg.new_label_to_generic_label_path)

    # Load and prepare 3D objects dataset
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        cfg.scenestate.path_to_pickled_3d_futute_models
    )
    for obj in objects_dataset:
        obj.size = obj.size * 2

    # Load and prepare model
    model = FurnitureAttributesModel.load_from_checkpoint(cfg.checkpoint_path, config=cfg)
    model.eval()
    model.freeze()

    return model, pix_ratio_threshold, new_label_to_generic_label, objects_dataset


def predict_furniture_attributes(model: FurnitureAttributesModel, 
                                semantic_map_tensor: torch.Tensor,
                                instances_data: Dict,
                                cfg: DictConfig,
                                new_label_to_generic_label: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Run furniture attribute prediction on all instances."""
    predictions = []
    scene_state_pred = []

    for instance_id, data in instances_data.items():
        with torch.no_grad():
            size_pred, offset_pred, orientation_pred = model(
                semantic_map_tensor.unsqueeze(0).cuda(), 
                data['mask'].cuda(), 
                data['category'].unsqueeze(0).cuda()
            )

        # Process predictions
        size_pred_list = size_pred.squeeze(0).squeeze(0).tolist()
        orientation_pred_class = torch.argmax(orientation_pred, dim=-1)
        orientation_pred_radians = orientation_pred_class * 90 * (np.pi / 180)
        orientation_pred_scalar = orientation_pred_radians.item()
        offset_pred_scalar = offset_pred.item()

        # Create standard orientation vector
        orientation = [1.570796251296997, 4.371138828673793e-08, orientation_pred_scalar]

        # Extract category information
        category_id = data['category'].argmax().item()
        category = new_label_to_generic_label[str(category_id)]

        # Calculate 3D location
        location = convert_2d_to_3d(
            data['mask'].squeeze(0).squeeze(0).numpy(), 
            image_width=cfg.image_width,
            image_height=cfg.image_height
        )
        location_3d = [location[0]["x"], location[0]["y"], offset_pred_scalar]

        # Format prediction
        prediction = {
            "category": category,
            "size": size_pred_list,
            "orientation": orientation,
            "location": location_3d
        }
        predictions.append(prediction)

        # Prepare scene state data if needed
        if cfg.for_scenestate:
            coco_polygon = mask_to_coco_polygon(data['mask'].squeeze())
            pred = {
                "category": category_id,
                "mask": coco_polygon,
                "size": size_pred_list,
                "orientation": orientation_pred_scalar,
                "offset": offset_pred_scalar,
                "inst_id": instance_id
            }
            scene_state_pred.append(pred)

    return predictions, scene_state_pred


def run_standard_inference(cfg: DictConfig, 
                          model: FurnitureAttributesModel,
                          pix_ratio_threshold: Dict,
                          new_label_to_generic_label: Dict,
                          objects_dataset: ThreedFutureDataset) -> int:
    """Run inference on standard semantic maps."""
    semantic_map_paths = glob.glob(f"{cfg.semantic_map_dir}/*label*")

    # Get floor plan paths if available
    floor_plan_paths = []
    if hasattr(cfg.scenestate, 'floor_plan_dir') and cfg.scenestate.floor_plan_dir:
        floor_plan_paths = sorted(os.listdir(cfg.scenestate.floor_plan_dir))

    # Create output directory
    anno_file_name = f"generate_anno_{cfg.room_type}_{cfg.version}_w_arch"
    output_annotations_dir = os.path.join(cfg.output_dir, anno_file_name)
    os.makedirs(output_annotations_dir, exist_ok=True)

    print(f"Processing {len(semantic_map_paths)} semantic maps...")

    for path in tqdm(semantic_map_paths, desc="Standard Inference"):
        try:
            # Load and process semantic map
            image_id = os.path.basename(path).split(".")[0]
            semantic_map = Image.open(path).convert("L")
            transform = transforms.ToTensor()
            semantic_map_tensor = transform(semantic_map) * 255

            # Get instance data
            instances_data = get_instance_masks(semantic_map_tensor, pix_ratio_threshold, new_label_to_generic_label)
            if not instances_data:
                continue

            # Run predictions
            predictions, scene_state_pred = predict_furniture_attributes(
                model, semantic_map_tensor, instances_data, cfg, new_label_to_generic_label
            )

            # Save annotations
            annotation_path = os.path.join(output_annotations_dir, f"{image_id}_for_scenestate.json")
            with open(annotation_path, 'w') as f:
                json.dump(scene_state_pred, f, indent=4)

            # Export scene state if enabled
            if cfg.for_scenestate and scene_state_pred:
                semantic_map_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                export_scenestate(
                    cfg.scenestate, new_label_to_generic_label, pix_ratio_threshold,
                    objects_dataset, scene_state_pred, image_id, semantic_map_cv, floor_plan_paths
                )

        except Exception as e:
            print(f"Warning: Failed to process {path}: {e}")
            continue

    return len(semantic_map_paths)


def run_custom_inference(cfg: DictConfig,
                        model: FurnitureAttributesModel,
                        pix_ratio_threshold: Dict,
                        new_label_to_generic_label: Dict,
                        objects_dataset: ThreedFutureDataset) -> int:
    """Run inference on custom floor plan samples with scaling and processing."""
    # Process custom samples
    semantic_map_paths, floor_plan_info, scaled_semantic_dir = process_custom_samples(
        cfg.custom_samples_dir, cfg.custom_output_dir, cfg.image_width, cfg.image_height
    )

    # Create output directories
    anno_file_name = f"generate_anno_custom_{cfg.room_type}_{cfg.version}"
    annotations_output_dir = os.path.join(cfg.custom_output_dir, anno_file_name)
    floor_plan_export_dir = os.path.join(cfg.custom_output_dir, "floor_plans_for_export")
    os.makedirs(annotations_output_dir, exist_ok=True)
    os.makedirs(floor_plan_export_dir, exist_ok=True)

    print(f"Processing {len(semantic_map_paths)} custom samples...")

    processed_count = 0
    for i, semantic_map_path in enumerate(tqdm(semantic_map_paths, desc="Custom Inference")):
        try:
            floor_info = floor_plan_info[i]
            image_id = os.path.basename(semantic_map_path).replace('_label.png', '')
            floor_plan_dir = os.path.dirname(semantic_map_path).split('/')[-1]
            full_image_id = f"{floor_plan_dir}_{image_id}"

            # Process semantic map with scaling and coloring
            scaled_map, scaled_path, colored_path = process_semantic_map(
                semantic_map_path,
                cfg.image_width,
                cfg.image_height,
                color_palette_path=getattr(cfg, 'color_palette_path', None),
                idx_to_label_path=getattr(cfg, 'idx_to_label_path', None),
                output_dir=scaled_semantic_dir,
                sample_id=full_image_id
            )

            if scaled_map is None:
                continue

            # Create tensor from processed map
            if scaled_path and os.path.exists(scaled_path):
                semantic_map = Image.open(scaled_path).convert("L")
            else:
                semantic_map = Image.fromarray(scaled_map, mode='L')

            transform = transforms.ToTensor()
            semantic_map_tensor = transform(semantic_map) * 255

            # Get instance data and run predictions
            instances_data = get_instance_masks(semantic_map_tensor, pix_ratio_threshold, new_label_to_generic_label)
            if not instances_data:
                continue

            predictions, scene_state_pred = predict_furniture_attributes(
                model, semantic_map_tensor, instances_data, cfg, new_label_to_generic_label
            )

            # Save annotations
            annotation_path = os.path.join(annotations_output_dir, f"{full_image_id}_for_scenestate.json")
            with open(annotation_path, 'w') as f:
                json.dump(scene_state_pred, f, indent=4)

            # Export scene state with floor plan if enabled
            if cfg.for_scenestate and scene_state_pred:
                # Copy floor plan for export
                numeric_floor_path = os.path.join(floor_plan_export_dir, f"{i}.png")
                shutil.copy2(floor_info['converted_path'], numeric_floor_path)

                # Configure scene state for custom floor plans
                scene_state_cfg = cfg.scenestate
                scene_state_cfg.floor_plan_dir = floor_plan_export_dir
                scene_state_cfg.w_arch = True
                scene_state_cfg.w_floor = True

                # Export scene state
                semantic_map_cv = scaled_map if scaled_path is None else cv2.imread(scaled_path, cv2.IMREAD_GRAYSCALE)
                numeric_room_id = f"custom-{i:06d}"

                export_scenestate(
                    scene_state_cfg, new_label_to_generic_label, pix_ratio_threshold,
                    objects_dataset, scene_state_pred, numeric_room_id, 
                    semantic_map_cv, [f"{i}.png"]
                )

            processed_count += 1

        except Exception as e:
            print(f"Warning: Failed to process {semantic_map_path}: {e}")
            continue

    # Save summary
    output_paths = {
        "annotations": annotations_output_dir,
        "floor_plans": floor_plan_export_dir,
        "scaled_maps": scaled_semantic_dir
    }

    config_info = {
        "checkpoint_path": cfg.checkpoint_path,
        "room_type": cfg.room_type,
        "version": cfg.version,
        "resolution": f"{cfg.image_width}x{cfg.image_height}",
        "generate_colored": bool(getattr(cfg, 'color_palette_path', None))
    }

    summary = create_inference_summary(processed_count, output_paths, config_info, is_custom=True)
    summary_path = os.path.join(cfg.custom_output_dir, "inference_summary.json")
    save_inference_summary(summary, summary_path)

    print(f"Custom inference completed! Summary saved to: {summary_path}")
    return processed_count


@hydra.main(config_path="../configs/apm", config_name="unified_config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main inference function supporting both standard and custom inference modes.

    Automatically detects inference mode based on configuration parameters:
    - Standard mode: Uses existing semantic maps from configured directory
    - Custom mode: Processes custom floor plan samples with scaling and conversion

    Args:
        cfg (DictConfig): Hydra configuration object
    """
    try:
        print("=== Furniture Attributes Prediction Inference ===")

        # Load model and datasets
        model, pix_ratio_threshold, new_label_to_generic_label, objects_dataset = load_model_and_datasets(cfg)
        print(f"Model loaded successfully from: {cfg.checkpoint_path}")

        # Determine inference mode
        is_custom_inference = hasattr(cfg, 'custom_samples_dir') and cfg.custom_samples_dir
        mode = "custom" if is_custom_inference else "standard"
        print(f"Running {mode} inference mode...")

        # Run appropriate inference
        if is_custom_inference:
            processed_count = run_custom_inference(
                cfg, model, pix_ratio_threshold, new_label_to_generic_label, objects_dataset
            )
        else:
            processed_count = run_standard_inference(
                cfg, model, pix_ratio_threshold, new_label_to_generic_label, objects_dataset
            )

        print(f"✓ Inference completed successfully! Processed {processed_count} samples.")

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
