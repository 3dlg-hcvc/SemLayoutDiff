"""
Architecture and floor plan utilities for 3D scene generation.
"""

import os
import json
import cv2
import numpy as np


def get_arch_ids(room_type):
    """Get class IDs for architectural elements for a specific room."""
    config_path = os.path.join("preprocess/metadata", f"{room_type}_idx_to_generic_label.json")
    
    with open(config_path, 'r') as f:
        cls_label_map = json.load(f)
        floor_id = int(next(key for key, value in cls_label_map.items() if value == "floor"))
        door_id = int(next(key for key, value in cls_label_map.items() if value == "door"))
        window_id = int(next(key for key, value in cls_label_map.items() if value == "window"))
    
    return floor_id, door_id, window_id


def create_floor_points(floor_mask):
    """Create floor points from binary mask."""
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


def create_arch_points(semantic_map, use_floor_plan=False, room_type=None):
    """Creates architecture map for bedroom semantic segmentation."""
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
        """Get contour points from mask."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [[[ (point[0][0] - offset) * scale, (point[0][1] - offset) * scale, -0.05] for point in contour] for contour in contours]

    # Get door, window, and floor points
    floor_mask = (arch_map == 1).astype(np.uint8)
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    door_points = get_contour_points(arch_map == 2)
    window_points = get_contour_points(arch_map == 3)
    floor_points = get_contour_points(arch_map == 1)
    
    floor_points = max(floor_points, key=len) if floor_points else []

    def create_holes(points, is_vertical, wall_coord, p1, p2, i, hole_id, hole_type):
        """Create holes (doors/windows) in walls."""
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