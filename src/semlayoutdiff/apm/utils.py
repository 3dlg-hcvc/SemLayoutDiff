"""
Backward compatibility module for the refactored utils.

This module maintains backward compatibility by importing all functions
from the new modular structure. The original large utils.py file has been
split into focused modules:

- geometry_utils.py: Geometric transformations and coordinate conversions
- mask_utils.py: Mask processing and instance detection
- architecture_utils.py: Floor plans and architectural elements
- scene_export.py: Scene state generation and export

All original function names and signatures remain unchanged.
"""

# Import all functions from the modular structure
from .geometry_utils import (
    class_to_angle,
    classify_angle,
    face_inward_orientation,
    convert_2d_to_3d,
    convert_2d_to_3d_scenestate,
    calculate_centroid,
    calculate_bounding_box,
    combine_bounding_boxes,
    draw_binary_mask
)

from .mask_utils import (
    get_size_from_mask,
    check_mask_area,
    mask_to_coco_polygon,
    get_instance_masks,
    map_floor_plan_to_result
)

from .architecture_utils import (
    get_arch_ids,
    create_floor_points,
    create_arch_points
)

from .scene_export import (
    export_scenestate,
    convert_bbox_info_to_scenestate
)

# Ensure all functions are available when importing this module
__all__ = [
    'class_to_angle',
    'classify_angle', 
    'face_inward_orientation',
    'convert_2d_to_3d',
    'convert_2d_to_3d_scenestate',
    'calculate_centroid',
    'calculate_bounding_box',
    'combine_bounding_boxes',
    'draw_binary_mask',
    'get_size_from_mask',
    'check_mask_area',
    'mask_to_coco_polygon',
    'get_instance_masks',
    'map_floor_plan_to_result',
    'get_arch_ids',
    'create_floor_points',
    'create_arch_points',
    'export_scenestate',
    'convert_bbox_info_to_scenestate'
]