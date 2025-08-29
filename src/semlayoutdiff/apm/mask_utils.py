"""
Mask processing and instance detection utilities for semantic segmentation.
"""

import torch
import numpy as np
import cv2
from scipy.ndimage import label


def get_size_from_mask(cfg, mask, orientation):
    """Get size from mask based on orientation.
    
    Object coordinate: z front, y up. Use the orientation to decide where is the z axis 
    and use the 2D mask to get x and z direction size.
    """
    # Convert the flat list to a list of (x, y) tuples
    canvas = np.zeros((cfg.image_height, cfg.image_width), dtype=np.uint8)
    mask_arrays = [np.array(polygon).reshape(-1, 2) for polygon in mask]
    cv2.fillPoly(canvas, mask_arrays, 1)

    # Concatenate all polygons into a single array
    all_points = np.concatenate(mask_arrays)

    # Get the bounding box
    x, y, w, h = cv2.boundingRect(all_points)
    # Align the bounding box to the orientation
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


def check_mask_area(cfg, mask, thresh, valid_pixels):
    """Check if mask area meets threshold requirements."""
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
    """Convert mask tensor to COCO polygon format."""
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
    """Extract instance masks from semantic segmentation map.
    
    Performs connected component analysis to separate instances of the same category.
    """
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

            # Get category id for floor and lamps from new_label_to_generic_label
            floor_id = [int(idx) for idx, label in new_label_to_generic_label.items() if label == "floor"][0]
            lamp_ids = [int(idx) for idx, label in new_label_to_generic_label.items() if "lamp" in label]
            door_id = [int(idx) for idx, label in new_label_to_generic_label.items() if "door" in label][0]
            window_id = [int(idx) for idx, label in new_label_to_generic_label.items() if "window" in label][0]
            
            # Check if the instance is lamp (5 or 16) or not, if not check the pixel ratio
            # If the instance is a lamp, add it to the lamp_masks dictionary
            if category_id_int in lamp_ids:
                if category_id_int not in lamp_masks:
                    lamp_masks[category_id_int] = instance_mask
                else:
                    lamp_masks[category_id_int] = np.logical_or(lamp_masks[category_id_int], instance_mask)
                continue
            elif category_id_int != floor_id and category_id_int != door_id and category_id_int != window_id:
                pixel_ratio = num_pixels / valid_pixels
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
            
    return best_plan