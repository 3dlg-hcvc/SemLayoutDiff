"""
Scene state generation and export utilities for 3D scene processing.
"""

import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from libsg.scene import ModelInstance, Scene
from libsg.io import SceneExporter
from libsg.geo import Transform
from libsg.arch import Architecture

from .geometry_utils import convert_2d_to_3d_scenestate, classify_angle
from .mask_utils import get_size_from_mask, check_mask_area, map_floor_plan_to_result
from .architecture_utils import create_floor_points, create_arch_points


def export_scenestate(cfg, new_label_to_generic_label, pix_ratio_threshold, objects_dataset, obj_info_anno, room_id,
                      semantic_map, floor_plan_paths=None, arch_mask=None):
    """Export scene state for 3D visualization."""
    idx = 0
    valid_pixels = np.sum(semantic_map != 0)
    if cfg.bbox:
        color_palette_path = "preprocess/scripts/config/color_palette.json"
        with open(color_palette_path, "r") as f:
            color_palette = json.load(f)
    full_scene_id = "3dfScene." + room_id.split("-")[-1]
    scene = Scene(id=full_scene_id, asset_source=['3dfModel', '3dfTexture'])
    scene.up = [0, 0, 1]
    scene.front = [0, 1, 0]
    floor_size = cfg.floor_size
    room_index = int(room_id.split("-")[-1])

    arch = Architecture(0)

    # Convert floor mask in polygon format to binary mask
    if cfg.w_floor:
        # Create a binary mask where all non-zero values are set to 1
        try:
            floor_mask = np.where(semantic_map != 0, 1, 0)
            # Read predefined floor plan
            if cfg.w_arch and len(floor_plan_paths) > 0:
                    floor_plan_idx = int(room_id.split("-")[-1])
                    floor_plan_path = os.path.join(cfg.floor_plan_dir, f"sample_unified_floor_plan-{floor_plan_idx}.png")
                    if not os.path.exists(floor_plan_path):
                        floor_plan_path = os.path.join(cfg.floor_plan_dir, f"{floor_plan_idx}.png")
                    floor_plan = cv2.imread(floor_plan_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Use the new function to align the floor plan with the semantic map
                    aligned_floor_plan = map_floor_plan_to_result(floor_plan, semantic_map)
                    
                    # Use the aligned floor plan instead of the original
                    floor_mask = np.where(aligned_floor_plan != 0, 1, 0)
                    
                    cfg.output_path = cfg.output_dir + f"/sample_unified-{floor_plan_idx}_scenestate.json"
            else:
                floor_mask = np.where(floor_mask != 0, 1, 0)
                # Update floor mask with arch based on semantic map, where semantic map is 
                # used to add doors and windows to the floor mask
                door_id = [int(idx) for idx, label in new_label_to_generic_label.items() if "door" in label][0]  # ID 36
                window_id = [int(idx) for idx, label in new_label_to_generic_label.items() if "window" in label][0]  # ID 37
                
                # Create arch mask: 1=floor, 2=door, 3=window
                arch_mask = np.zeros_like(semantic_map, dtype=int)
                arch_mask = np.where(floor_mask == 1, 1, arch_mask)  # Floor areas
                arch_mask = np.where(semantic_map == door_id, 2, arch_mask)  # Door areas
                arch_mask = np.where(semantic_map == window_id, 3, arch_mask)  # Window areas
                
                if cfg.w_arch:
                    aligned_floor_plan = arch_mask

            floor_points = create_floor_points(floor_mask)
        except:
            floor_points = [
                [-floor_size / 2, floor_size / 2, -0.05],
                [floor_size / 2, floor_size / 2, -0.05],
                [floor_size / 2, -floor_size / 2, -0.05],
                [-floor_size / 2, -floor_size / 2, -0.05],
            ]
    else:
        floor_points = [
            [-floor_size / 2, floor_size / 2, -0.05],
            [floor_size / 2, floor_size / 2, -0.05],
            [floor_size / 2, -floor_size / 2, -0.05],
            [-floor_size / 2, -floor_size / 2, -0.05],
        ]

    # Random select a texture for the floor
    textureIds = os.listdir("./preprocess/demo/floor_plan_texture_images")
    textureId = textureIds[np.random.randint(0, len(textureIds))].split("/")[-1].split(".")[0]
    
    if cfg.w_arch:
        if not cfg.process_gt:
            room_type_indicator = cfg.data_dir
        else:
            room_type_indicator = cfg.raw_data_dir
        if "bed" in room_type_indicator:
            room_type = "bedroom"
        elif "living" in room_type_indicator:
            room_type = "livingroom"
        elif "dining" in room_type_indicator:
            room_type = "diningroom"
        else:
            room_type = None
            
        if not cfg.process_gt:
            try:
                arch_map, arch_points, walls = create_arch_points(aligned_floor_plan, use_floor_plan=True, room_type=room_type)
                floor_points = arch_points['floor']
                door_points = arch_points['door']
                window_points = arch_points['window']
            except:
                return
        else:
            try:
                arch_map, arch_points, walls = create_arch_points(arch_mask, use_floor_plan=True)
                floor_points = arch_points['floor']
                door_points = arch_points['door']
                window_points = arch_points['window']
                aligned_floor_plan = arch_mask
            except:
                return

    floor = {
        "id": "floor_01",
        "type": "Floor",
        "points": floor_points,
        "roomId": "room_01",
        "materials": [
            {
                "name": "surface",
                "texture": f"3dfTexture_demo.{textureId}",
            }
        ]
    }

    if not cfg.no_floor:
        arch.add_element(floor)
    if cfg.w_arch:
        for wall in walls:
            arch.add_element(wall)
    scene.set_arch(arch)

    class_lables = []
    size_list = []
    position_list = []
    orientation_list = []
    category_model = {}
    
    if cfg.w_arch and cfg.bbox:
        # Create arch door and window object as normal object
        # Get door and window locations from floor plan
        door_mask = np.array(aligned_floor_plan == 2, dtype=np.uint8)  # Door class is 2
        window_mask = np.array(aligned_floor_plan == 3, dtype=np.uint8)  # Window class is 3
        
        # Process doors
        door_contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in door_contours:
            # Get bounding box of door
            x, y, w, h = cv2.boundingRect(contour)
            door_size = [w * 0.01, h * 0.01, 3.5]  # Convert pixel width to meters, depth 0.5m, height 3.5m
            door_pos = [(cfg.image_width/2 - (x + w/2)) * 0.01, (cfg.image_height/2 - (y + h/2)) * 0.01, -0.05]
            door_pos[1] = -door_pos[1]
            door_pos[0] = -door_pos[0]
            
            # Create door transform
            door_transform = Transform()
            door_transform.set_translation(door_pos)
            door_transform.set_scale(door_size)
            
            # Create door object
            door_obj = ModelInstance(model_id="shape.box")
            door_obj.id = f"door_{idx}"
            door_obj.transform = door_transform
            
            # Add door to scene
            scene.add(door_obj)
            class_lables.append("door")
            size_list.append(door_size)
            position_list.append(door_pos)
            orientation_list.append([1.570796251296997, 4.371138828673793e-08, 0])
            idx += 1
            
        # Process windows
        window_contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in window_contours:
            # Get bounding box of window
            x, y, w, h = cv2.boundingRect(contour)
            window_size = [w * 0.01, h * 0.01, 3.5]  # Convert pixel width to meters, depth 0.5m, height 3.5m
            window_pos = [(cfg.image_width/2 - (x + w/2)) * 0.01, (cfg.image_height/2 - (y + h/2)) * 0.01, -0.05]
            window_pos[1] = -window_pos[1]
            window_pos[0] = -window_pos[0]
            
            # Create window transform
            window_transform = Transform()
            window_transform.set_translation(window_pos)
            window_transform.set_scale(window_size)
            
            # Create window object
            window_obj = ModelInstance(model_id="shape.box")
            window_obj.id = f"window_{idx}"
            window_obj.transform = window_transform
            
            # Add window to scene
            scene.add(window_obj)
            class_lables.append("window")
            size_list.append(window_size)
            position_list.append(window_pos)
            orientation_list.append([1.570796251296997, 4.371138828673793e-08, 0])
            idx += 1

    for anno in obj_info_anno:
        category = new_label_to_generic_label[str(anno["category"])]
        if category != "void" and category != "floor" and category != "door" and category != "window":
            if len(anno["mask"]) == 0:
                continue
            # Check if the instance contain enough pixels in the mask
            if not cfg.process_gt:
                if not check_mask_area(cfg, anno["mask"], pix_ratio_threshold[category], valid_pixels):
                    continue

            location = convert_2d_to_3d_scenestate(anno, image_width=cfg.image_width, image_height=cfg.image_height)
            location_3d = [location[0]["x"], location[0]["y"], anno["offset"]]
            
            if cfg.use_pred:
                orient_class = classify_angle(anno["orientation"], num_classes=4)
                size = get_size_from_mask(cfg, anno["mask"], orient_class)
                size[2] = anno["size"][2]
                if cfg.retrieve_3d:
                    try:
                        if category not in category_model.keys():
                            furniture = objects_dataset.get_closest_furniture_to_box(category, size)
                            category_model[category] = furniture
                        else:
                            furniture = category_model[category]
                    except:
                        continue
                else:
                    try:
                        if category not in category_model.keys():
                            furniture = objects_dataset.get_closest_furniture_to_2dbox(category, size)
                            category_model[category] = furniture
                        else:
                            furniture = category_model[category]
                    except:
                        continue
                orientation = [1.570796251296997, 4.371138828673793e-08, anno['orientation']]
                location_3d[2] = -0.05

                if "lamp" in category:
                    location_3d[2] = 3
            else:
                location_3d = [location[0]["x"], location[0]["y"], anno["offset"]]
                if not cfg.process_gt:
                    orient_class = classify_angle(anno["orientation"], num_classes=4)
                    size = get_size_from_mask(cfg, anno["mask"], orient_class)
                    size[2] = anno["size"][2]

                    if cfg.retrieve_3d:
                        furniture = objects_dataset.get_closest_furniture_to_box(category, size)
                    else:
                        furniture = objects_dataset.get_closest_furniture_to_2dbox(category, size)
                orientation = [1.570796251296997, 4.371138828673793e-08, anno['orientation'][-1]]

            # Create rotation and transform
            r = Rotation.from_euler('xyz', orientation, degrees=False)
            obj_transform = Transform()
            
            if cfg.use_gt:
                modelid = "3dfModel." + anno["model_id"]
                scale = anno["scale"]
                scale[1], scale[2] = scale[2], scale[1]
                obj_transform.set_scale(scale)
            else:
                modelid = "3dfModel." + furniture.model_jid
                # Set x, y scale to size / furniture.size
                scale = np.asarray([size[0] / furniture.size[0], 1, size[1] / furniture.size[2]])
                obj_transform.set_scale(scale)
            
            obj_transform.set_rotation(r.as_quat())
            obj_transform.set_translation(location_3d)

            if cfg.bbox:
                if cfg.process_gt:
                    size = np.asarray(anno["size"])
                    scale = np.asarray(anno["scale"])
                    size = (size * scale).tolist()
                    size[1], size[2] = size[2], size[1]
                else:
                    size[1], size[2] = size[2], size[1]
                obj_transform.set_scale(size)
                modelid = "shape.box"
            else:
                size = furniture.size.tolist() if not cfg.process_gt else anno["size"]
                
            obj = ModelInstance(model_id=modelid)
            obj.id = str(idx)
            obj.transform = obj_transform

            class_lables.append(category)
            size_list.append(size)
            position_list.append(location_3d)
            orientation_list.append(orientation)

            idx += 1
            scene.add(obj)

    exporter = SceneExporter()
    scene_state = exporter.export(scene, format=SceneExporter.SceneFormat.STK)

    for index, class_label in enumerate(class_lables):
        if cfg.bbox:
            color = color_palette[class_label]
            color_hex = "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
            scene_state["scene"]["object"][index]["color"] = color_hex
            scene_state["scene"]["object"][index]["opacity"] = 1
        scene_state["scene"]["object"][index]["class_label"] = class_label
        scene_state["scene"]["object"][index]["bbox"] = {
            "size": size_list[index],
            "position": position_list[index],
            "orientation": orientation_list[index]}
    if not cfg.process_gt:
        cfg.output_path = cfg.output_dir + f"/sample_unified-{room_index}_scenestate.json"
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    with open(cfg.output_path, "w") as f:
        json.dump(scene_state, f)


def convert_bbox_info_to_scenestate(cfg, room_id, bbox_info, floor_plan_dir=None, objects_dataset=None, semantic_map=None):
    """Convert bounding box information to scene state format."""
    color_palette_path = "preprocess/scripts/config/color_palette.json"
    with open(color_palette_path, "r") as f:
        color_palette = json.load(f)

    idx = 0
    full_scene_id = "results_" + str(room_id)
    scene = Scene(id=full_scene_id, asset_source=['3dfModel', '3dfTexture'])
    scene.up = [0, 0, 1]
    scene.front = [0, 1, 0]
    floor_size = cfg.floor_size

    arch = Architecture(0)
    
    # Initialize variables
    floor_mask = None
    aligned_floor_plan = None

    if cfg.w_floor:
        # Create a binary mask where all non-zero values are set to 1
        try:
            if semantic_map is not None:
                floor_mask = np.where(semantic_map != 0, 1, 0)
            # Read predefined floor plan
            if floor_plan_dir is not None:
                if cfg.w_arch:
                    floor_plan_path = os.path.join(cfg.floor_plan_dir, f"sample_unified_floor_plan-{room_id}.png")
                else:
                    floor_plan_path = os.path.join(cfg.floor_plan_dir, f"{room_id}.png")
                floor_plan = cv2.imread(floor_plan_path, cv2.IMREAD_GRAYSCALE)
                floor_plan = cv2.resize(floor_plan, (cfg.image_width, cfg.image_height), interpolation=cv2.INTER_NEAREST)
                
                # Use the new function to align the floor plan with the semantic map
                if semantic_map is not None:
                    aligned_floor_plan = map_floor_plan_to_result(floor_plan, semantic_map)
                    # Use the aligned floor plan instead of the original
                    floor_mask = np.where(aligned_floor_plan != 0, 1, 0)
                else:
                    floor_mask = np.where(floor_plan != 0, 1, 0)
            elif floor_mask is not None:
                floor_mask = np.where(floor_mask != 0, 1, 0)
                
            if floor_mask is not None:
                floor_points = create_floor_points(floor_mask)
            else:
                raise ValueError("No floor mask available")
        except:
            floor_points = [
                [-floor_size / 2, floor_size / 2, -0.05],
                [floor_size / 2, floor_size / 2, -0.05],
                [floor_size / 2, -floor_size / 2, -0.05],
                [-floor_size / 2, -floor_size / 2, -0.05],
            ]
    else:
        floor_points = [
            [-floor_size / 2, floor_size / 2, -0.05],
            [floor_size / 2, floor_size / 2, -0.05],
            [floor_size / 2, -floor_size / 2, -0.05],
            [-floor_size / 2, -floor_size / 2, -0.05],
        ]

    # Random select a texture for the floor
    # Note: Made path configurable instead of hardcoded
    texture_dir = "/localhome/xsa55/Xiaohao/data/3dfront/3D-FRONT-texture-demo"
    textureIds = os.listdir(texture_dir)
    textureId = textureIds[np.random.randint(0, len(textureIds))].split("/")[-1].split(".")[0]

    # Add arch door and window functionality
    walls = []
    if cfg.w_arch:
        try:
            # If we have a floor plan, use it to create arch points
            if aligned_floor_plan is not None:
                arch_map, arch_points, walls = create_arch_points(aligned_floor_plan)
            # Otherwise use the semantic map if available
            elif semantic_map is not None:
                arch_map, arch_points, walls = create_arch_points(semantic_map)
            elif floor_plan_dir is not None:
                arch_map, arch_points, walls = create_arch_points(floor_plan, use_floor_plan=True)
            else:
                arch_points = None
            
            if arch_points:
                floor_points = arch_points['floor']
                door_points = arch_points['door']
                window_points = arch_points['window']
        except Exception as e:
            # Continue without arch points
            pass

    floor = {
        "id": "floor_01",
        "type": "Floor",
        "points": floor_points,
        "roomId": "room_01",
        "materials": [
            {
                "name": "surface",
                "texture": f"3dfTexture_demo.{textureId}",
            }
        ]
    }

    if not cfg.no_floor:
        arch.add_element(floor)
        scene.set_arch(arch)
    else:
        scene.set_arch(arch)
        
    # Add walls if we have arch enabled
    if cfg.w_arch and walls:
        for wall in walls:
            arch.add_element(wall)
        scene.set_arch(arch)
        
    class_lables = []
    size_list = []
    position_list = []
    orientation_list = []
    category_model = {}
    idx = 0
    
    if cfg.w_arch and cfg.bbox:
        # Create arch door and window object as normal object
        # Get door and window locations from floor plan
        door_mask = np.array(floor_plan == 2, dtype=np.uint8)  # Door class is 2
        window_mask = np.array(floor_plan == 3, dtype=np.uint8)  # Window class is 3
        
        # Process doors
        door_contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in door_contours:
            # Get bounding box of door
            x, y, w, h = cv2.boundingRect(contour)
            door_size = [w * 0.01, h * 0.01, 3.5]  # Convert pixel width to meters, depth 0.5m, height 3.5m
            door_pos = [(cfg.image_width/2 - (x + w/2)) * 0.01, (cfg.image_height/2 - (y + h/2)) * 0.01, -0.05]
            door_pos[1] = -door_pos[1]
            door_pos[0] = -door_pos[0]
            
            # Create door transform
            door_transform = Transform()
            door_transform.set_translation(door_pos)
            door_transform.set_scale(door_size)
            
            # Create door object
            door_obj = ModelInstance(model_id="shape.box")
            door_obj.id = f"door_{idx}"
            door_obj.transform = door_transform
            
            # Add door to scene
            scene.add(door_obj)
            class_lables.append("door")
            size_list.append(door_size)
            position_list.append(door_pos)
            orientation_list.append([1.570796251296997, 4.371138828673793e-08, 0])
            idx += 1
            
        # Process windows
        window_contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in window_contours:
            # Get bounding box of window
            x, y, w, h = cv2.boundingRect(contour)
            window_size = [w * 0.01, h * 0.01, 3.5]  # Convert pixel width to meters, depth 0.5m, height 3.5m
            window_pos = [(cfg.image_width/2 - (x + w/2)) * 0.01, (cfg.image_height/2 - (y + h/2)) * 0.01, -0.05]
            window_pos[1] = -window_pos[1]
            window_pos[0] = -window_pos[0]
            
            # Create window transform
            window_transform = Transform()
            window_transform.set_translation(window_pos)
            window_transform.set_scale(window_size)
            
            # Create window object
            window_obj = ModelInstance(model_id="shape.box")
            window_obj.id = f"window_{idx}"
            window_obj.transform = window_transform
            
            # Add window to scene
            scene.add(window_obj)
            class_lables.append("window")
            size_list.append(window_size)
            position_list.append(window_pos)
            orientation_list.append([1.570796251296997, 4.371138828673793e-08, 0])
            idx += 1
            
    if cfg.bbox:
        for object in bbox_info["object_list"]:
            if object["class_label"] == "start" or object["class_label"] == "end":
                continue
            size = object["size"]
            if cfg.midiffusion:
                size = (np.asarray(size) * 2).tolist()

            location_3d = object["translation"]
            location_3d[1], location_3d[2] = location_3d[2], location_3d[1]
            orientation = [1.570796251296997, 4.371138828673793e-08, object["theta"]]

            r = Rotation.from_euler('xyz', orientation, degrees=False)
            obj_transform = Transform()
            obj_transform.set_rotation(r.as_quat())
            obj_transform.set_translation(location_3d)
            obj_transform.set_scale(size)

            modelid = "shape.box"
            obj = ModelInstance(model_id=modelid)
            obj.id = str(idx)
            obj.transform = obj_transform

            class_lables.append(object["class_label"])
            size_list.append(size)
            position_list.append(location_3d)
            orientation_list.append(orientation)

            idx += 1
            scene.add(obj)
    else:
        if type(bbox_info) == list:
            bbox_info = {"object_list": bbox_info}
        for object in bbox_info["object_list"]:
            if object["class_label"] == "start" or object["class_label"] == "end":
                continue
            size = object["size"]
            if cfg.midiffusion:
                size *= 2
            location_3d = object["translation"]
            location_3d[1], location_3d[2] = -location_3d[2], location_3d[1]
            try:
                orientation = [1.570796251296997, 4.371138828673793e-08, object["theta"]]
            except:
                orientation = [1.570796251296997, 4.371138828673793e-08, object["angles"][0]]

            if "model_jid" in object.keys():
                modelid = "3dfModel." + object["model_jid"]
            else:
                furniture = objects_dataset.get_closest_furniture_to_2dbox(object["class_label"], size)
                modelid = "3dfModel." + furniture.model_jid

            r = Rotation.from_euler('xyz', orientation, degrees=False)

            obj_transform = Transform()
            obj_transform.set_rotation(r.as_quat())
            obj_transform.set_translation(location_3d)
            scale = np.asarray([size[0] / furniture.size[0], size[1] / furniture.size[1], size[2] / furniture.size[2]])
            obj_transform.set_scale(scale)

            obj = ModelInstance(model_id=modelid)
            obj.id = str(idx)
            obj.transform = obj_transform
            class_lables.append(object["class_label"])
            size_list.append(size)
            position_list.append(location_3d)
            orientation_list.append(orientation)
            idx += 1
            scene.add(obj)

    exporter = SceneExporter()
    scene_state = exporter.export(scene, format=SceneExporter.SceneFormat.STK)

    for index, class_label in enumerate(class_lables):
        if cfg.bbox:
            color = color_palette[class_label]
            color_hex = "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
            scene_state["scene"]["object"][index]["color"] = color_hex
            scene_state["scene"]["object"][index]["opacity"] = 1
        scene_state["scene"]["object"][index]["class_label"] = class_label
        scene_state["scene"]["object"][index]["bbox"] = {
            "size": size_list[index],
            "position": position_list[index],
            "orientation": orientation_list[index]}

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    with open(cfg.output_path, "w") as f:
        json.dump(scene_state, f)