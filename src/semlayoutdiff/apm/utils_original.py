import os
import torch
from scipy.ndimage import label
import numpy as np
import cv2
import json
import math
from scipy.spatial.transform import Rotation

from libsg.scene import ModelInstance, Scene
from libsg.io import SceneExporter
from libsg.geo import Transform
from libsg.arch import Architecture


def class_to_angle(class_label, num_classes):
    if num_classes == 4:
        # Each class represents a 90-degree segment
        angle = class_label * 90
    else:
        # Each class represents a 45-degree segment
        angle = class_label * 45

    return angle


def face_inward_orientation(x, y):
    # Calculate the angle in degrees from the positive x-axis
    angle = math.degrees(math.atan2(y, x)) % 360

    # Classify the angle into one of the four categories
    if 315 <= angle or angle < 45:
        orientation = 3
    elif 45 <= angle < 135:
        orientation = 0
    elif 135 <= angle < 225:
        orientation = 1
    else:
        orientation = 2

    return orientation

def get_size_from_mask(cfg, mask, orientation):
    # object coordinate: z front, y up. use the orientation to decide where is the z axis and use the 2D mask get x and z direction size
    # Convert the flat list to a list of (x, y) tuples
    canvas = np.zeros((cfg.image_height, cfg.image_width), dtype=np.uint8)
    mask_arrays = [np.array(polygon).reshape(-1, 2) for polygon in mask]
    cv2.fillPoly(canvas, mask_arrays, 1)

    # Concatenate all polygons into a single array
    all_points = np.concatenate(mask_arrays)

    # get the bounding box
    x, y, w, h = cv2.boundingRect(all_points)
    # align the bounding box to the orientation
    w = w * 0.01
    h = h * 0.01
    if orientation == 0:
        size = [w, h, 0.01]
    elif orientation == 1:
        size = [h, w, 0.01]
    elif orientation == 2:
        size = [w, h, 0.01]
    elif orientation == 3:
        size = [h, w, 0.01]
    else:
        raise ValueError("Orientation should be in [0, 1, 2, 3]")
    return size

def classify_angle(angle, num_classes):
    # Calculate the class by dividing the angle by 45
    # import pdb;pdb.set_trace()
    if isinstance(angle, list):
        angle = angle[2]
    angle = angle * 180 / math.pi
    if num_classes == 4:
        class_label = int((angle + 45) // 90) % 4
    else:
        class_label = int((angle + 22.5) // 45) % 8
    return class_label

def create_floor_points(floor_mask):
    # Find the contours of the binary mask
    contours, _ = cv2.findContours(floor_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour which should be the floor
    largest_contour = max(contours, key=cv2.contourArea)


    # Convert the contour points to the required format
    # Assuming image center is (0,0), convert pixel coordinates to meters
    points = [[(point[0][0] - floor_mask.shape[1]/2) * 0.01, 
               (point[0][1] - floor_mask.shape[0]/2) * 0.01, 
               -0.05] for point in largest_contour]

    return points

def get_arch_ids(room_type):
    """Get class IDs for architectural elements for a specific room"""
    config_path = os.path.join("preprocess/metadata", f"{room_type}_idx_to_generic_label.json")
    
    with open(config_path, 'r') as f:
        cls_label_map = json.load(f)
        floor_id = int(next(key for key, value in cls_label_map.items() if value == "floor"))
        door_id = int(next(key for key, value in cls_label_map.items() if value == "door"))
        window_id = int(next(key for key, value in cls_label_map.items() if value == "window"))
    
    return floor_id, door_id, window_id

def create_arch_points(semantic_map, use_floor_plan=False, room_type=None):
    """Creates architecture map for bedroom semantic segmentation"""
    # Get IDs for architectural elements

    
    if room_type is None:
        room_type = "bedroom"
    if use_floor_plan:
        floor_id, door_id, window_id = 1, 2, 3
    else:
        floor_id, door_id, window_id = get_arch_ids(room_type)
    

    
    # Create initial arch map
    arch_map = np.zeros_like(semantic_map)
    arch_map[semantic_map != 0] = 1
    arch_map[semantic_map == door_id] = 2
    arch_map[semantic_map == window_id] = 3
    



    def get_contour_points(mask, scale=0.01, offset=1200 / 2):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [[[ (point[0][0] - offset) * scale, (point[0][1] - offset) * scale, -0.05] for point in contour] for contour in contours]

    # Get door, window, and floor points
    floor_mask = (arch_map == 1).astype(np.uint8)
    # breakpoint()
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    door_points = get_contour_points(arch_map == 2)
    window_points = get_contour_points(arch_map == 3)
    floor_points = get_contour_points(arch_map == 1)
    

    
    floor_points = max(floor_points, key=len) if floor_points else []
    


    def create_holes(points, is_vertical, wall_coord, p1, p2, i, hole_id, hole_type):
        holes = []
        for point_set in points:
            point_array = np.array(point_set)[:, 1 if is_vertical else 0]
            min_index, max_index = np.argmin(np.abs(point_array)), np.argmax(np.abs(point_array))
            point_min, point_max = point_array[min_index], point_array[max_index]
            ref_min = np.array(point_set)[:, 0 if is_vertical else 1][np.argmin(np.abs(np.array(point_set)[:, 0 if is_vertical else 1]))]
            
            # Calculate wall length for proper boundary checking
            wall_start = p1[1 if is_vertical else 0]
            wall_end = p2[1 if is_vertical else 0]
            wall_length = abs(wall_end - wall_start)
            
            if abs(ref_min - wall_coord) < 0.1 and min(wall_start, wall_end) <= point_min <= max(wall_start, wall_end):
                # Calculate distances from wall start point
                diff_min = abs(point_min - wall_start)
                diff_max = abs(point_max - wall_start)
                
                # Ensure proper ordering (min < max)
                if diff_min > diff_max:
                    diff_min, diff_max = diff_max, diff_min
                
                # Clamp the hole boundaries to wall length
                diff_min = max(0.0, min(diff_min, wall_length))
                diff_max = max(diff_min, min(diff_max, wall_length))
                
                # Only create hole if it has valid dimensions within the wall
                if diff_max > diff_min and diff_max <= wall_length:
                    min_point = [diff_min, -0.05 if hole_type == "Door" else 0.5]
                    max_point = [diff_max, 2]
                    holes.append({
                        "id": f"{hole_type.lower()}_{i}_{hole_id}",
                        "type": hole_type,
                        "box": {
                            "min": min_point,
                            "max": max_point
                        }
                    })
                    hole_id += 1
        return holes

    # Get wall points and match doors/windows to walls
    walls = []
    for i, (p1, p2) in enumerate(zip(floor_points, floor_points[1:] + [floor_points[0]])):
        is_vertical = abs(p2[0] - p1[0]) < 0.1
        wall_coord = p1[0] if is_vertical else p1[1]
        holes = create_holes(door_points, is_vertical, wall_coord, p1, p2, i, 0, "Door")
        holes += create_holes(window_points, is_vertical, wall_coord, p1, p2, i, len(holes), "Window")
        wall = {
            "id": f"wall_{i+1:02d}",
            "type": "Wall",
            "points": [p1, p2],
            "height": 3,
            "roomId": "room_01",
            "materials": [{"name": "surface", "diffuse": "#888899"}]
        }
        if holes:
            wall["holes"] = holes
        walls.append(wall)
        
    arch_points = {'door': door_points, 'window': window_points, 'floor': floor_points}
    

    
    return arch_map, arch_points, walls


def check_mask_area(cfg, mask, thresh, valid_pixels):
    valid_instance = False

    canvas_height, canvas_width = cfg.image_height, cfg.image_width
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    for poly in mask:
        # Convert the flat list to a list of (x, y) tuples for each polygon
        try:
            mask_polygon = np.array(poly).reshape(-1, 2)
        except:
            import pdb
            pdb.set_trace()

        # Draw the polygon on the canvas
        cv2.fillPoly(canvas, [mask_polygon], 1)

    # Count the pixels
    num_pixels = np.sum(canvas)

    pixel_ratio = num_pixels / valid_pixels

    if pixel_ratio > thresh:
        valid_instance = True

    return valid_instance

def mask_to_coco_polygon(mask_tensor):
    # Convert mask tensor to numpy array
    mask_np = mask_tensor.numpy().astype('uint8')

    # Find contours of the objects
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # COCO polygon format requires the points to be in a flat list
    polygons = []
    for contour in contours:
        # Simplify the contour to reduce the number of points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Flatten the contour points and add to polygons list
        polygon = approx.ravel().tolist()
        polygons.append(polygon)

    return polygons


def get_instance_masks(semantic_map, pix_ratio_threshold, new_label_to_generic_label, num_categories=36):
    # Convert the semantic map to a numpy array for processing with scipy
    semantic_map_np = semantic_map.squeeze().numpy()
    valid_pixels = np.sum(semantic_map_np != 0)

    # Find unique category IDs in the semantic map
    unique_categories = np.unique(semantic_map_np)

    # Prepare a dictionary to store instance masks and category one-hot encodings
    instances_data = {}
    instance_id = 0  # We start instance IDs at 1

    # Prepare a dictionary to store lamp masks
    lamp_masks = {}

    # Perform connected component analysis for each category
    for category_id in unique_categories:
        if category_id == 0:  # Assuming 0 is the background or not an instance category
            continue
        category = new_label_to_generic_label[str(int(category_id))]

        # Ensure that category_id is an integer and subtract 1 for zero-based indexing
        category_id_int = int(category_id)

        # Create a binary mask for the current category
        category_mask = (semantic_map_np == category_id)

        # Perform connected component analysis to separate instances of the same category
        labeled_instances, num_features = label(category_mask)

        # Iterate over each detected instance for the current category
        for i in range(1, num_features + 1):
            instance_id += 1  # Increment the instance ID

            # Extract the individual instance mask
            instance_mask = (labeled_instances == i)

            # Count the pixels
            num_pixels = np.sum(instance_mask)

            # get categroy id for floor and lamps from new_label_to_generic_label
            floor_id = [int(idx) for idx, label in new_label_to_generic_label.items() if label == "floor"][0]
            lamp_ids = [int(idx) for idx, label in new_label_to_generic_label.items() if "lamp" in label]
            door_id = [int(idx) for idx, label in new_label_to_generic_label.items() if "door" in label][0]
            window_id = [int(idx) for idx, label in new_label_to_generic_label.items() if "window" in label][0]
            # check if the instance is lamp (5 or 16) or not, if not check the pixel ratio
            # If the instance is a lamp, add it to the lamp_masks dictionary
            if category_id_int in lamp_ids:
                if category_id_int not in lamp_masks:
                    lamp_masks[category_id_int] = instance_mask
                else:
                    lamp_masks[category_id_int] = np.logical_or(lamp_masks[category_id_int], instance_mask)
                continue
            elif category_id_int != floor_id and category_id_int != door_id and category_id_int != window_id:
                # image_size = semantic_map_np.shape[0] * semantic_map_np.shape[1]
                pixel_ratio = num_pixels / valid_pixels
                # if pixel_ratio < pix_ratio_threshold[category]:
                #     continue
                if pixel_ratio < 0.0001:
                    continue
            else:
                continue

            # Convert instance mask to tensor and add batch and channel dimensions
            instance_mask_tensor = torch.from_numpy(instance_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

            # One-hot encode the category ID
            # Make sure the category ID is within the valid range for one-hot encoding
            try:
                category_tensor = torch.nn.functional.one_hot(torch.tensor(category_id_int),
                                                              num_classes=num_categories).float()
            except:
                breakpoint()

            # Store the instance mask and category ID using the instance ID as the key
            instances_data[instance_id] = {
                'mask': instance_mask_tensor,
                'category': category_tensor
            }
    # Add the combined lamp masks to the instances_data dictionary
    for lamp_id, lamp_mask in lamp_masks.items():
        instance_id += 1
        lamp_mask_tensor = torch.from_numpy(lamp_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        lamp_category_tensor = torch.nn.functional.one_hot(torch.tensor(lamp_id),
                                                           num_classes=num_categories).float()
        instances_data[instance_id] = {
            'mask': lamp_mask_tensor,
            'category': lamp_category_tensor
        }

    return instances_data


def convert_2d_to_3d(mask, image_width, image_height):
    # Initialize 2D points list (in the XZ-plane)
    points_3d = []

    x_coords, y_coords = np.nonzero(mask)

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    centroid_x = (max_x + min_x) / 2
    centroid_y = (max_y + min_y) / 2

    scale_x = 0.01
    scale_y = 0.01

    # Convert to 2D coordinates in the XZ-plane
    x_3d = -(image_width / 2 - centroid_x) * scale_x
    y_3d = (centroid_y - image_height / 2) * scale_y

    # Add to points list
    points_3d.append({'x': x_3d, 'y': y_3d})

    return points_3d

def convert_2d_to_3d_scenestate(annotation, image_width, image_height):
    centroids = []
    bounding_boxes = []
    polygon = annotation['mask']
    size = annotation["size"]
    inst_id = {annotation["inst_id"]}

    # debug the mask
    mask = draw_binary_mask(annotation['mask'], image_width, image_height)
    cv2.imwrite(f"../../tmp_log/{inst_id}.png", mask * 255)

    # Initialize 2D points list (in the XZ-plane)
    points_2d = []
    for polygon in annotation['mask']:
        # Calculate 2D centroids and convert to 2D coordinates in the XZ-plane
        centroids.append(calculate_centroid(polygon))

        # Calculate bounding box
        bounding_boxes.append(calculate_bounding_box(polygon))

    # Combine centroids and bounding boxes
    try:
        min_x, max_x, min_y, max_y = combine_bounding_boxes(bounding_boxes)
    except:
        import pdb
        pdb.set_trace()
    centroid_x = (max_x + min_x) / 2
    centroid_y = (max_y + min_y) / 2

    scale = 0.01

    # Convert to 2D coordinates in the XZ-plane
    x_2d = -(image_width / 2 - centroid_x) * scale
    y_2d = (centroid_y - image_height / 2) * scale

    # Add to points list
    points_2d.append({'x': x_2d, 'y': y_2d})

    return points_2d

def draw_binary_mask(polygons, image_width, image_height):
    # Create an empty mask with the same dimensions as the image
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Loop over all polygons (which are masks for the object)
    for polygon in polygons:
        # Convert the flat list of coordinates into a list of (x, y) tuples
        points = np.array([[polygon[i], polygon[i + 1]] for i in range(0, len(polygon), 2)], dtype=np.int32)

        # Draw the polygon on the mask
        cv2.fillPoly(mask, [points], color=1)

    return mask

def calculate_centroid(polygon):
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return centroid_x, centroid_y

def calculate_bounding_box(polygon):
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return min_x, max_x, min_y, max_y

def combine_bounding_boxes(bounding_boxes):
    # Combine bounding boxes to get a box that includes all masks
    min_x = min([bbox[0] for bbox in bounding_boxes])
    max_x = max([bbox[1] for bbox in bounding_boxes])
    min_y = min([bbox[2] for bbox in bounding_boxes])
    max_y = max([bbox[3] for bbox in bounding_boxes])
    return min_x, max_x, min_y, max_y

def map_floor_plan_to_result(floor_plan, result_semantic_map):
    """
    Maps a floor plan to match the position and scale of a result semantic map.
    First resizes the floor plan, then translates and scales to align content.
    
    Args:
        floor_plan: The source floor plan image with values for floor, door, window
        result_semantic_map: The target semantic map to align with
        
    Returns:
        Aligned floor plan matching the result semantic map
    """
    # First resize the floor plan to match the dimensions of the result

    resized_floor_plan = cv2.resize(
        floor_plan, 
        (result_semantic_map.shape[1], result_semantic_map.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create binary masks for finding content regions
    resized_binary = np.where(resized_floor_plan > 0, 1, 0).astype(np.uint8)
    result_binary = np.where(result_semantic_map > 0, 1, 0).astype(np.uint8)
    
    # Find content regions
    floor_y, floor_x = np.where(resized_binary > 0)
    result_y, result_x = np.where(result_binary > 0)
    
    if len(floor_y) == 0 or len(result_y) == 0:
        # Handle empty images
        return np.zeros_like(result_semantic_map)
    
    # Get content bounds
    floor_y_min, floor_y_max = np.min(floor_y), np.max(floor_y)
    floor_x_min, floor_x_max = np.min(floor_x), np.max(floor_x)
    
    result_y_min, result_y_max = np.min(result_y), np.max(result_y)
    result_x_min, result_x_max = np.min(result_x), np.max(result_x)
    
    # Calculate content centers
    floor_center_y = (floor_y_min + floor_y_max) // 2
    floor_center_x = (floor_x_min + floor_x_max) // 2
    
    result_center_y = (result_y_min + result_y_max) // 2
    result_center_x = (result_x_min + result_x_max) // 2
    
    # Calculate translation to align centers
    tx = result_center_x - floor_center_x
    ty = result_center_y - floor_center_y
    
    # Calculate content dimensions
    floor_height = floor_y_max - floor_y_min + 1
    floor_width = floor_x_max - floor_x_min + 1
    
    result_height = result_y_max - result_y_min + 1
    result_width = result_x_max - result_x_min + 1
    
    # Calculate scale factors
    scale_x = result_width / floor_width if floor_width > 0 else 1
    scale_y = result_height / floor_height if floor_height > 0 else 1
    
    # Use a uniform scale to maintain aspect ratio
    scale = max(scale_x, scale_y)
    
    # Create transformation matrix for translation and scaling
    # Scale around the center of the floor content
    M = np.float32([
        [scale, 0, tx + floor_center_x * (1 - scale)],
        [0, scale, ty + floor_center_y * (1 - scale)]
    ])
    
    # Apply the transformation
    aligned_floor_plan = cv2.warpAffine(
        resized_floor_plan,
        M,
        (result_semantic_map.shape[1], result_semantic_map.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Try different rotations to find the best match
    best_iou = np.sum(np.where(aligned_floor_plan > 0, 1, 0) & result_binary) / \
               max(np.sum(np.where(aligned_floor_plan > 0, 1, 0) | result_binary), 1)
    best_plan = aligned_floor_plan
    for angle in [90, 180, 270]:
        # Create a rotation matrix around the result center
        # Convert center coordinates to Python integers to avoid type errors
        center_x = int(result_center_x)
        center_y = int(result_center_y)
        rot_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(
            aligned_floor_plan,
            rot_matrix,
            (result_semantic_map.shape[1], result_semantic_map.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Calculate IoU
        rot_iou = np.sum(np.where(rotated > 0, 1, 0) & result_binary) / \
                  max(np.sum(np.where(rotated > 0, 1, 0) | result_binary), 1)
        
        if rot_iou > best_iou:
            best_iou = rot_iou
            best_plan = rotated
    # breakpoint()
    return best_plan



def export_scenestate(cfg, new_label_to_generic_label, pix_ratio_threshold, objects_dataset, obj_info_anno, room_id,
                      semantic_map, floor_plan_paths=None, arch_mask=None):
    # obj_info_anno_path = "/localhome/xsa55/Xiaohao/SemDiffLayout/tmp_log/temp_generated_scene.json"
    
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

    # convert floor mask in polygon format to binary mask
    if cfg.w_floor:
        # Create a binary mask where all non-zero values are set to 1
        try:
            floor_mask = np.where(semantic_map != 0, 1, 0)
            # read predefined floor plan
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
                # update floor mask with arch based on semantic map, where semantic map is 
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
    textureIds = os.listdir(
        "./preprocess/demo/floor_plan_texture_images")
    # textureId = textureIds[np.random.randint(0, len(textureIds))].split("/")[-1]
    textureId = textureIds[np.random.randint(0, len(textureIds))].split("/")[-1].split(".")[0]
    # breakpoint()
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
                # breakpoint()
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
        # create arch door and window object as normal object
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
        # breakpoint()
        # if "door" in anno["basename"].lower() or "window" in anno["basename"].lower():
        #     continue
        category = new_label_to_generic_label[str(anno["category"])]
        if category != "void" and category != "floor" and category != "door" and category != "window":
            if len(anno["mask"]) == 0:
                continue
            # check if the instance contain enough pixels in the mask
            if not cfg.process_gt:
                if not check_mask_area(cfg, anno["mask"], pix_ratio_threshold[category], valid_pixels):
                    continue
            # if not check_mask_area(cfg, anno["mask"], 1e-6, valid_pixels):
            #     continue

            location = convert_2d_to_3d_scenestate(anno, image_width=cfg.image_width, image_height=cfg.image_height)
            location_3d = [location[0]["x"], location[0]["y"], anno["offset"]]
            # breakpoint()
            if cfg.use_pred:
                orient_class = classify_angle(anno["orientation"], num_classes=4)
                size = get_size_from_mask(cfg, anno["mask"], orient_class)
                size[2] = anno["size"][2]
                if cfg.retrieve_3d:
                    try:
                        if category not in category_model.keys():
                            furniture = objects_dataset.get_closest_furniture_to_box(
                                category, size
                            )
                            category_model[category] = furniture
                        else:
                            furniture = category_model[category]
                    except:
                        continue
                else:

                    try:
                        if category not in category_model.keys():
                            furniture = objects_dataset.get_closest_furniture_to_2dbox(
                                category, size
                            )
                            category_model[category] = furniture
                        else:
                            furniture = category_model[category]
                    except:
                        continue
                    # furniture = objects_dataset.get_closest_furniture_to_2dbox(
                    #     category, size
                    # )
                orientation = [
                    1.570796251296997,
                    4.371138828673793e-08,
                    anno['orientation'],
                ]
                location_3d[2] = -0.05

                if "lamp" in category:
                    location_3d[2] = 3
            else:
                location_3d = [location[0]["x"], location[0]["y"], anno["offset"]]
                # location_3d[2] = 0
                # size = np.asarray(anno["size"])
                if not cfg.process_gt:
                    orient_class = classify_angle(anno["orientation"], num_classes=4)
                    size = get_size_from_mask(cfg, anno["mask"], orient_class)
                    size[2] = anno["size"][2]

                    if cfg.retrieve_3d:
                        furniture = objects_dataset.get_closest_furniture_to_box(
                            category, size
                        )
                        # furniture = furniture[0]
                    else:
                        furniture = objects_dataset.get_closest_furniture_to_2dbox(
                            category, size
                        )
                # orientation = anno['orientation']
                orientation = [
                    1.570796251296997,
                    4.371138828673793e-08,
                    anno['orientation'][-1],
                ]

            # import pdb; pdb.set_trace()
            r = Rotation.from_euler('xyz', orientation, degrees=False)
            obj_transform = Transform()
            # obj_transform.set_scale(scale)
            # modelid = "3dfModel." + furniture.model_jid
            if cfg.use_gt:
                modelid = "3dfModel." + anno["model_id"]
                scale = anno["scale"]
                scale[1], scale[2] = scale[2], scale[1]
                obj_transform.set_scale(scale)
            else:
                modelid = "3dfModel." + furniture.model_jid
                # set x, y scale to size / furniture.size
                scale = np.asarray([size[0] / furniture.size[0], 1, size[1] / furniture.size[2]])
                
                # scale = np.asarray([1, 1, 1])
                # breakpoint()
                obj_transform.set_scale(scale)
            obj_transform.set_rotation(r.as_quat())
            # obj_transform.set_rotation(orientation)
            obj_transform.set_translation(location_3d)

            # import pdb; pdb.set_trace()
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
                size =  furniture.size.tolist() if not cfg.process_gt else anno["size"]
                
            obj = ModelInstance(model_id=modelid)
            obj.id = str(idx)
            obj.transform = obj_transform
            # obj.color = [255,0,0]
            # obj.metadata = {"class_label": category}


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
            # import pdb; pdb.set_trace()
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
    # breakpoint()
    color_palette_path = "preprocess/scripts/config/color_palette.json"
    with open(color_palette_path, "r") as f:
        color_palette = json.load(f)

    idx = 0

    # import pdb;pdb.set_trace()
    full_scene_id = "results_" + str(room_id)
    scene = Scene(id=full_scene_id, asset_source=['3dfModel', '3dfTexture'])
    scene.up = [0, 0, 1]
    scene.front = [0, 1, 0]
    floor_size = cfg.floor_size
    # breakpoint()

    arch = Architecture(0)
    
    # Initialize variables
    floor_mask = None
    aligned_floor_plan = None

    if cfg.w_floor:
        # breakpoint()
        # Create a binary mask where all non-zero values are set to 1
        try:
            if semantic_map is not None:
                floor_mask = np.where(semantic_map != 0, 1, 0)
            # read predefined floor plan
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
    textureIds = os.listdir(
        "/localhome/xsa55/Xiaohao/data/3dfront/3D-FRONT-texture-demo")
    # textureId = textureIds[np.random.randint(0, len(textureIds))].split("/")[-1]
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
                # "diffuse": "#757678",
                # "texture": f"3dfTexture.{textureId}",
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
        # create arch door and window object as normal object
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
            # size[1], size[2] = size[2], size[1]

            location_3d = object["translation"]
            location_3d[1], location_3d[2] = location_3d[2], location_3d[1]
            orientation = [
                1.570796251296997,
                4.371138828673793e-08,
                object["theta"],
            ]

            r = Rotation.from_euler('xyz', orientation, degrees=False)
            obj_transform = Transform()
            obj_transform.set_rotation(r.as_quat())
            # obj_transform.set_rotation(orientation)
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
            # breakpoint()
            if object["class_label"] == "start" or object["class_label"] == "end":
                continue
            size = object["size"]
            if cfg.midiffusion:
                size *= 2
            # scale = object["scale"]
            # scale[1], scale[2] = scale[2], scale[1]
            location_3d = object["translation"]
            location_3d[1], location_3d[2] = -location_3d[2], location_3d[1]
            try:
                orientation = [
                    1.570796251296997,
                    4.371138828673793e-08,
                    object["theta"],
                ]
            except:
                orientation = [
                    1.570796251296997,
                    4.371138828673793e-08,
                    object["angles"][0],
                ]

            if "model_jid" in object.keys():
                modelid = "3dfModel." + object["model_jid"]
            else:
                # if "cabinet" in object["class_label"]:
                #     object["class_label"] = "cabinet"
                furniture = objects_dataset.get_closest_furniture_to_2dbox(
                                    object["class_label"], size
                                )
                modelid = "3dfModel." + furniture.model_jid

            r = Rotation.from_euler('xyz', orientation, degrees=False)

            obj_transform = Transform()
            obj_transform.set_rotation(r.as_quat())
            obj_transform.set_translation(location_3d)
            scale = np.asarray([size[0] / furniture.size[0], size[1] / furniture.size[1], size[2] / furniture.size[2]])
            # scale = object["scale"]
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
            # import pdb; pdb.set_trace()
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