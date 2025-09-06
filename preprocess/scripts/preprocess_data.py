# 
# Modified from:
#   https://github.com/nv-tlabs/ATISS.
#   https://github.com/MIT-SPARK/ThreedFront
#
"""Script to parse the 3D-FRONT data scenes to numpy files in order to avoid 
I/O overhead in training.
"""
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm
import seaborn as sns

from threed_front.datasets import filter_function
from threed_front.datasets.threed_front_encoding_base import get_basic_encoding
from threed_front.rendering import scene_from_args, get_floor_plan, \
    get_textured_objects_in_scene, render_projection, get_arch_plan
from threed_front.simple_3dviz_setup import ORTHOGRAPHIC_PROJECTION_SCENE
from utils import PATH_TO_PROCESSED_DATA, PATH_TO_PICKLED_3D_FRONT_DATASET, PATH_TO_PICKLED_3D_FRONT_W_ARCH_DATASET, \
    PATH_TO_DATASET_FILES, PATH_TO_FLOOR_PLAN_TEXTURES, load_pickled_threed_front
from threed_front.utils import * 

# os.environ['MODERNGL_BACKEND'] = 'egl'

def class_to_new_id(class_names, room_type):
    config_dir = PATH_TO_DATASET_FILES + f"/{room_type}_idx_to_generic_label.json"
    with open(config_dir, "r") as f:
        config = json.load(f)
    
    name_to_id = {v: k for k, v in config.items()}
    # map class names to object ids
    object_ids = [int(name_to_id[name]) for name in class_names]
    
    return object_ids

def main(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
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
        "--output_directory",
        default=PATH_TO_PROCESSED_DATA,
        help="Path to output directory (default: output/3d_front_processed/<room_type>)"
    )
    parser.add_argument(
        "--path_to_pickled_3d_front_dataset",
        default=PATH_TO_PICKLED_3D_FRONT_DATASET,
        help="Path to pickled 3D-FRONT dataset (default: output/threed_front.pkl)"
    )
    parser.add_argument(
        "--path_to_pickled_3d_front_w_arch_dataset",
        default=PATH_TO_PICKLED_3D_FRONT_W_ARCH_DATASET,
        help="Path to pickled 3D-FRONT dataset (default: output/threed_front_w_arch.pkl)"
    )
    parser.add_argument(
        "--path_to_dataset_files_directory",
        default=PATH_TO_DATASET_FILES,
        help="Path to directory storing black_list.txt, invalid_threed_front_rooms.txt, "
        "and <room_type>_threed_front_splits.csv",
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default=PATH_TO_FLOOR_PLAN_TEXTURES,
        help="Path to floor texture image directory"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=None,
        help="The size of the room along a side (default:3.1 or 6.1)"
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="Filter out lamps when extracting objects in the scene"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="Color objects by semantic label, and set floor plan to white"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        help="Remove the floor plane"
    )
    parser.add_argument(
        "--with_arch",
        action="store_true",
        help="Remove the floor plane"
    )
    # add objfeat
    parser.add_argument(
        "--add_objfeats",
        action="store_true",
        help="Add object point cloud features (make sure raw_model_norm_pc_lat.npz "
        "and raw_model_norm_pc_lat32.npz exist in the raw dataset directory)"
    )

    args = parser.parse_args(argv)
    if args.with_arch:
        args.path_to_pickled_3d_front_dataset = args.path_to_pickled_3d_front_w_arch_dataset

    room_type = args.dataset_filtering.split("_")[-1]
    print(f"Room type: {room_type}")

    # Check if output directory exists and if it doesn't create it
    args.output_directory = args.output_directory.format(room_type)
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    elif len(os.listdir(args.output_directory)) > 0:
        print(f"Warning: a non-empty output directory {args.output_directory} exists.")
        overwrite_subdirectory = None
        while overwrite_subdirectory not in {"y", "n"}:
            overwrite_subdirectory = \
                input("Do you want to overwrite existing subdirectories? [y/n] ")
                
    # set door and window color
    door_color = (0.8, 0, 0)
    window_color = (1.0, 0.8, 0.8)

    # Set floor texture/color (color has higher priority if args.no_texture)
    if args.without_floor:
        floor_color = None
        floor_textures = [None]
    elif args.no_texture:
        floor_color = (1, 1, 1, 1)  # white floor
        floor_textures = [None]
    else:
        floor_color = None
        floor_textures = \
            [os.path.join(args.path_to_floor_plan_textures, fi)
                for fi in os.listdir(args.path_to_floor_plan_textures)]

    # Create the scene
    scene_params = ORTHOGRAPHIC_PROJECTION_SCENE
    if args.room_side is None:
        scene_params["room_side"] = 3.1 if room_type in ["bedroom", "library"] \
            else 6.1
    else:
        scene_params["room_side"] = args.room_side
    if args.without_floor:
        scene_params["background"] = (1, 1, 1, 1)
    scene = scene_from_args(scene_params)
    print("Room side:", scene_params["room_side"])

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

    # Initially, we only consider the train split to compute the dataset
    # statistics, e.g the translations, sizes and angles bounds
    filter_fn = filter_function(config, ["train", "val"], args.without_lamps)
    train_dataset = load_pickled_threed_front(
        args.path_to_pickled_3d_front_dataset, filter_fn
    )
    print("Loaded dataset with {} training rooms".format(len(train_dataset)))

    # Compute the bounds for the translations, sizes and angles in the dataset.
    # This will then be used to properly align rooms.
    tr_bounds = train_dataset.centroids
    si_bounds = train_dataset.sizes
    train_dataset_stats = {
        "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
        "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
        "bounds_angles": [float(b) for b in train_dataset.angles],
        "class_labels": train_dataset.class_labels,
        "object_types": train_dataset.object_types,
        "class_frequencies": train_dataset.class_frequencies,
        "class_order": train_dataset.class_order,
        "count_furniture": train_dataset.count_furniture
    }
    if args.add_objfeats:
        if train_dataset.objfeats is not None:
            train_dataset_stats.update({
                "bounds_objfeats": [float(b) for b in train_dataset.objfeats],
            })
        if train_dataset.objfeats_32 is not None:
            train_dataset_stats.update({
                "bounds_objfeats_32": [float(b) for b in train_dataset.objfeats_32]
            })

    path_to_json = os.path.join(args.output_directory, "dataset_stats.txt")
    with open(path_to_json, "w") as f:
        json.dump(train_dataset_stats, f)
    print("Saving training statistics for dataset with bounds: {} to {}".format(
            train_dataset.bounds, path_to_json))

    # Load full dataset
    filter_fn = filter_function(config, ["train", "val", "test"], args.without_lamps)
    dataset = load_pickled_threed_front(
        args.path_to_pickled_3d_front_dataset, filter_fn
    )
    print("Loaded full dataset with {} rooms".format(len(dataset)))

    encoded_dataset = get_basic_encoding(
        dataset, box_ordering=None, add_objfeats=args.add_objfeats
    )

    if args.no_texture:
        color_palette = sns.color_palette('hls', dataset.n_object_types)

    layout_image = "rendered_scene{}_{}{}.png".format(
        "_notexture" if args.no_texture else "", 
        scene_params["window_size"][0],
        "_nofloor" if args.without_floor else ""
    )
    print("Saving layout images {} to each scene directory."\
          .format(layout_image))
    
     # Add a counter and limit
    processed_count = 0
    sample_limit = 100
    for es, ss in tqdm(zip(encoded_dataset, dataset)):
        # Create a separate folder for each room
        room_directory = os.path.join(args.output_directory, ss.uid)
        # processed_count += 1
        # if processed_count > sample_limit:
        #     break
        # if ss.uid != "cce1df99-43e2-4870-9e42-6c051513e84d_LivingDiningRoom-25117":
        #     continue

        # Skip existing room directory if the user does not want to overwrite
        if os.path.exists(room_directory) and overwrite_subdirectory == "n":
            continue
        else:
            os.makedirs(room_directory, exist_ok=True)

        # 3D-FUTURE model ids
        uids = [bi.model_uid for bi in ss.bboxes]
        jids = [bi.model_jid for bi in ss.bboxes]

        floor_plan_vertices, floor_plan_faces = ss.floor_plan

        # Render and save the room mask as an image
        if (not args.without_floor):
            room_mask = render_projection(
                scene,
                [get_floor_plan(ss)[0]],
                (1.0, 1.0, 1.0),
                "flat",
                os.path.join(room_directory, "room_mask.png")
            )[:, :, 0:1]

        # render and save the arch mask as an image, void is 0, floor is 1, doors are 2, windows are 3
        if args.with_arch:
            archs, _, _, model_type_list = get_arch_plan(ss)
            # breakpoint()
            arch_mask = render_projection(
                scene,
                archs,
                [(1.0, 1.0, 1.0), (0.8, 0.8, 0.8), (0.6, 0.6, 0.6)],
                "flat",
                os.path.join(room_directory, "arch_mask.png"),
                model_type_list
            )[:, :, 0:1]
        
        # get window and door vertices and faces
        window_meshes = ss.window  # List of (vertices, faces) tuples
        door_meshes = ss.door     # List of (vertices, faces) tuples
        # Convert to object arrays to store meshes
        window_meshes = np.array(window_meshes, dtype=object) if window_meshes else np.array([], dtype=object)
        door_meshes = np.array(door_meshes, dtype=object) if door_meshes else np.array([], dtype=object)
        
        # get window and door centroids
        window_centroids = np.array(ss.window_centroids) if ss.window_centroids else np.array([])
        door_centroids = np.array(ss.door_centroids) if ss.door_centroids else np.array([])

        # get object type
        class_labels = es["class_labels"]
        class_names = [train_dataset_stats["class_labels"][i] for i in class_labels.argmax(axis=1).tolist()]
        object_ids = class_to_new_id(class_names, room_type)

        # breakpoint()
        # if args.add_objfeats:
        #     if es["objfeats"] is not None:
        #         objfeats = es["objfeats"]
        #     else:
        #         es["objfeats"] = np.zeros(es["objfeats_32"].shape, dtype=np.float32)
        #         print(f"Warning: objfeats is None for {ss.uid}, set to zeros")
        scene_type = ss.uid.lower()
        if "bed" in scene_type:
            scene_label = np.array([1.0, 0.0, 0.0, 0.0])
        elif "dining" in scene_type:
            scene_label = np.array([0.0, 1.0, 0.0, 0.0])
        elif "living" in scene_type:
            scene_label = np.array([0.0, 0.0, 1.0, 0.0])
        elif "library" in scene_type:
            scene_label = np.array([0.0, 0.0, 0.0, 1.0])



        # Save layout to boxes.npz
        if not args.add_objfeats:
            np.savez_compressed(
                os.path.join(room_directory, "boxes"),
                uids=uids,
                jids=jids,
                scene_id=ss.scene_id,
                scene_uid=ss.uid,
                scene_type=ss.scene_type,
                json_path=ss.json_path,
                room_layout=room_mask,
                floor_plan_vertices=floor_plan_vertices,
                floor_plan_faces=floor_plan_faces,
                floor_plan_centroid=ss.floor_plan_centroid,
                class_labels=es["class_labels"],
                object_ids=object_ids,
                translations=es["translations"],
                sizes=es["sizes"],
                angles=es["angles"],
                window_meshes=window_meshes,
                window_centroids=window_centroids,
                door_meshes=door_meshes,
                door_centroids=door_centroids,
                arch_mask=arch_mask,
                scene_label=scene_label
            )
        else:
            np.savez_compressed(
                os.path.join(room_directory, "boxes"),
                uids=uids,
                jids=jids,
                scene_id=ss.scene_id,
                scene_uid=ss.uid,
                scene_type=ss.scene_type,
                json_path=ss.json_path,
                room_layout=room_mask,
                floor_plan_vertices=floor_plan_vertices,
                floor_plan_faces=floor_plan_faces,
                floor_plan_centroid=ss.floor_plan_centroid,
                class_labels=es["class_labels"],
                object_ids=object_ids,
                translations=es["translations"],
                sizes=es["sizes"],
                angles=es["angles"],
                objfeats=es["objfeats"],
                objfeats_32=es["objfeats_32"],
                window_meshes=window_meshes,
                window_centroids=window_centroids,
                door_meshes=door_meshes,
                door_centroids=door_centroids,
                arch_mask=arch_mask
            )

        # Render a top-down orthographic projection of the room at a
        # specific pixel resoluti
        path_to_image = os.path.join(room_directory, layout_image)
        
        # object renderables
        if args.no_texture:
            # read class labels and get the color of each object
            class_labels = es["class_labels"]
            class_index = class_labels.argmax(axis=1).tolist()
            cc = [color_palette[ind] for ind in class_index]
            renderables = get_textured_objects_in_scene(ss, colors=cc)
        else:
            # use default texture files                
            renderables = get_textured_objects_in_scene(ss)
        
        # floor plan renderable
        texture = np.random.choice(floor_textures)
        if not args.with_arch:
            floor_plan, _, _ = get_floor_plan(
                ss, texture=texture, color=floor_color, 
                with_trimesh=False, with_room_mask=False
            )
            
            render_projection(
            scene, renderables + [floor_plan], color=None, mode="shading",
            frame_path=path_to_image
            )
        else:
            floor_plan, _, _, _ = get_arch_plan(
                ss, floor_texture=texture, floor_color=floor_color, door_color=door_color, window_color=window_color,
                with_trimesh=False, with_room_mask=False
            )

            render_projection(
                scene, renderables + floor_plan, color=None, mode="shading",
                frame_path=path_to_image
            )


if __name__ == "__main__":
    main(sys.argv[1:])
