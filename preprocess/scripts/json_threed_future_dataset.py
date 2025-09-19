#
# Modified from:
#   https://github.com/nv-tlabs/ATISS.
#   https://github.com/MIT-SPARK/ThreedFront
#
"""Script used for creating the 3D Future dataset in JSON format to be subsequently
used by our scripts.
"""
import argparse
import json
import os
import sys
import numpy as np

from threed_front.datasets import filter_function
from utils import PATH_TO_DATASET_FILES, PATH_TO_PICKLED_3D_FRONT_DATASET, \
    load_pickled_threed_front, PROJ_DIR


def numpy_to_python(obj):
    """Convert numpy arrays and types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting their attributes
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = numpy_to_python(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    else:
        return obj


def serialize_object(obj):
    """Serialize a 3D Future object to a dictionary."""
    obj_dict = {}
    
    # Basic attributes that should be present
    essential_attrs = [
        'model_jid', 'model_uid', 'size', 'label', 'scale', 
        'position', 'rotation', 'model_info'
    ]
    
    # Optional attributes that might be present
    optional_attrs = [
        'up', 'front', 'path_to_models', 'texture_image_path'
    ]
    
    # Serialize essential attributes
    for attr in essential_attrs:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            obj_dict[attr] = numpy_to_python(value)
    
    # Serialize optional attributes
    for attr in optional_attrs:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            obj_dict[attr] = numpy_to_python(value)
    
    # Handle computed properties that might be needed
    if hasattr(obj, 'z_angle'):
        try:
            obj_dict['z_angle'] = numpy_to_python(obj.z_angle)
        except:
            pass
    
    # Add any other important attributes
    for attr_name in dir(obj):
        if (not attr_name.startswith('_') and 
            not callable(getattr(obj, attr_name)) and
            attr_name not in essential_attrs and 
            attr_name not in optional_attrs and
            attr_name not in ['z_angle']):
            try:
                value = getattr(obj, attr_name)
                # Only add if it's a simple type that can be JSON serialized
                if isinstance(value, (str, int, float, bool, list, dict)) or isinstance(value, np.ndarray):
                    obj_dict[attr_name] = numpy_to_python(value)
            except:
                # Skip attributes that can't be serialized
                pass
    
    return obj_dict


def main(argv):
    parser = argparse.ArgumentParser(
        description="Create the 3D Future dataset in JSON format"
    )
    parser.add_argument(
        "dataset_filtering",
        choices=[
            "threed_front_bedroom",
            "threed_front_livingroom",
            "threed_front_diningroom",
            "threed_front_library",
            "threed_front_unified"
        ],
        help="The type of dataset filtering to be used"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to output JSON file (default: datasets/output/threed_future_model_{room_type}.json)"
    )
    parser.add_argument(
        "--path_to_pickled_3d_front_dataset",
        default=PATH_TO_PICKLED_3D_FRONT_DATASET,
        help="Path to pickled 3D-FRONT dataset (default: output/threed_front.pkl)"
    )
    parser.add_argument(
        "--path_to_dataset_files_directory",
        default=PATH_TO_DATASET_FILES,
        help="Path to directory storing black_list.txt, invalid_threed_front_rooms.txt, "
        "and <room_type>_threed_front_splits.csv",
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="Filter out lamps when extracting objects in the scene"
    )

    args = parser.parse_args(argv)
    
    room_type = args.dataset_filtering.split('_')[-1]
    
    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = os.path.join(PROJ_DIR, f"datasets/output/threed_future_model_{room_type}.json")
    
    if os.path.exists(args.output_path):
        input(f"Warning: {args.output_path} exists. Press any key to overwrite...")
    
    # Set up config for filtering
    config = {
        "filter_fn":                 args.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids":
            os.path.join(args.path_to_dataset_files_directory, "invalid_threed_front_rooms.txt"),
        "path_to_invalid_bbox_jids": 
            os.path.join(args.path_to_dataset_files_directory, "black_list.txt"),
        "annotation_file": 
            os.path.join(args.path_to_dataset_files_directory, f"{room_type}_threed_front_splits.csv")
    }

    # Extract scenes from train split
    filter_fn = filter_function(config, ["train", "val"], args.without_lamps)
    scenes_dataset = load_pickled_threed_front(
        args.path_to_pickled_3d_front_dataset, filter_fn
    )
    print("Loading dataset with {} rooms".format(len(scenes_dataset)))

    # Collect the set of objects in the scenes
    objects = {}
    for scene in scenes_dataset:
        for obj in scene.bboxes:
            objects[obj.model_jid] = obj
    objects = [vi for vi in objects.values()]

    print(f"Found {len(objects)} unique objects")

    # Convert objects to JSON-serializable format
    json_objects = []
    
    for i, obj in enumerate(objects):
        try:
            obj_dict = serialize_object(obj)
            json_objects.append(obj_dict)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(objects)} objects")
                
        except Exception as e:
            print(f"Error serializing object {i}: {e}")
            continue

    print(f"Successfully serialized {len(json_objects)} objects")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save to JSON file
    with open(args.output_path, "w") as f:
        json.dump(json_objects, f, indent=2)
    print("Saved result to: {}".format(args.output_path))

    # Print some statistics
    if json_objects:
        sample_obj = json_objects[0]
        print(f"\nSample object keys: {list(sample_obj.keys())}")
        
        # Count objects by label
        labels = {}
        for obj in json_objects:
            label = obj.get('label', 'unknown')
            labels[label] = labels.get(label, 0) + 1
        
        print(f"\nObject counts by label:")
        for label, count in sorted(labels.items()):
            print(f"  {label}: {count}")


if __name__ == "__main__":
    main(sys.argv[1:])
