"""
Geometry and coordinate transformation utilities for 3D scene processing.
"""

import math
import numpy as np
import cv2


def class_to_angle(class_label, num_classes):
    """Convert class label to angle in degrees."""
    if num_classes == 4:
        # Each class represents a 90-degree segment
        angle = class_label * 90
    else:
        # Each class represents a 45-degree segment
        angle = class_label * 45
    return angle


def classify_angle(angle, num_classes):
    """Classify angle into discrete classes."""
    if isinstance(angle, list):
        angle = angle[2]
    angle = angle * 180 / math.pi
    if num_classes == 4:
        class_label = int((angle + 45) // 90) % 4
    else:
        class_label = int((angle + 22.5) // 45) % 8
    return class_label


def face_inward_orientation(x, y):
    """Calculate face inward orientation based on x, y coordinates."""
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


def convert_2d_to_3d(mask, image_width, image_height):
    """Convert 2D mask to 3D coordinates in XZ-plane."""
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
    """Convert 2D annotation to 3D coordinates for scene state."""
    centroids = []
    bounding_boxes = []
    polygon = annotation['mask']
    size = annotation["size"]
    inst_id = annotation["inst_id"]

    # Debug the mask
    mask = draw_binary_mask(annotation['mask'], image_width, image_height)
    # Note: Removed hardcoded debug path - should be configurable
    # cv2.imwrite(f"../../tmp_log/{inst_id}.png", mask * 255)

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


def calculate_centroid(polygon):
    """Calculate centroid of a polygon."""
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return centroid_x, centroid_y


def calculate_bounding_box(polygon):
    """Calculate bounding box of a polygon."""
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return min_x, max_x, min_y, max_y


def combine_bounding_boxes(bounding_boxes):
    """Combine multiple bounding boxes to get a box that includes all masks."""
    min_x = min([bbox[0] for bbox in bounding_boxes])
    max_x = max([bbox[1] for bbox in bounding_boxes])
    min_y = min([bbox[2] for bbox in bounding_boxes])
    max_y = max([bbox[3] for bbox in bounding_boxes])
    return min_x, max_x, min_y, max_y


def draw_binary_mask(polygons, image_width, image_height):
    """Create a binary mask from polygons."""
    # Create an empty mask with the same dimensions as the image
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Loop over all polygons (which are masks for the object)
    for polygon in polygons:
        # Convert the flat list of coordinates into a list of (x, y) tuples
        points = np.array([[polygon[i], polygon[i + 1]] for i in range(0, len(polygon), 2)], dtype=np.int32)

        # Draw the polygon on the mask
        cv2.fillPoly(mask, [points], color=1)

    return mask