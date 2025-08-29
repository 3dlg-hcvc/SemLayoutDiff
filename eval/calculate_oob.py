import os
import json
import glob
import datetime
from shapely.geometry import Point, Polygon


class OutOfBoundaryEvaluator:
    """Evaluator for calculating out-of-boundary object ratios in scene files."""
    
    def __init__(self):
        """Initialize the OOB evaluator."""
        pass
    
    @staticmethod
    def is_point_in_polygon(point, polygon):
        """Check if a point is inside a polygon."""
        return Point(point).within(Polygon(polygon))
    
    def analyze_scene_file(self, file_path):
        """Analyze a scene file to check for out of boundary objects."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract floor points to define the room boundary
        floor_elements = [elem for elem in data['scene']['arch']['elements'] if elem['type'] == 'Floor']
        
        if not floor_elements:
            print(f"Warning: No floor elements found in {file_path}")
            return None
        
        # Get the floor polygon (using the first floor if multiple exist)
        floor = floor_elements[0]
        floor_points = [(p[0], p[1]) for p in floor['points']]
        
        # Get objects
        objects = data['scene']['object']
        
        out_of_boundary_objects = []
        
        for obj in objects:
            # Get object position (x, y)
            position = obj['bbox']['position']
            obj_point = (position[0], position[1])
            
            # Check if object is inside the floor boundary
            if not self.is_point_in_polygon(obj_point, floor_points):
                out_of_boundary_objects.append(obj)
        
        return {
            'total_objects': len(objects),
            'oob_objects': len(out_of_boundary_objects),
            'has_oob': len(out_of_boundary_objects) > 0
        }

    def calculate_oob_ratios(self, directory):
        """Calculate OOB ratios for all scene files in a directory."""
        scene_files = glob.glob(os.path.join(directory, '*.json'))
        
        total_scenes = len(scene_files)
        scenes_with_oob = 0
        total_objects = 0
        total_oob_objects = 0
        
        for file_path in scene_files:
            result = self.analyze_scene_file(file_path)
            if result:
                if result['has_oob']:
                    scenes_with_oob += 1
                total_objects += result['total_objects']
                total_oob_objects += result['oob_objects']
        
        # Calculate ratios
        scene_oob_ratio = scenes_with_oob / total_scenes if total_scenes > 0 else 0
        object_oob_ratio = total_oob_objects / total_objects if total_objects > 0 else 0
        
        return {
            'total_scenes': total_scenes,
            'scenes_with_oob': scenes_with_oob,
            'scene_oob_ratio': scene_oob_ratio,
            'total_objects': total_objects,
            'oob_objects': total_oob_objects,
            'object_oob_ratio': object_oob_ratio
        }

    def evaluate(self, input_folder, output_file=None, verbose=True):
        """
        Evaluate out-of-boundary metrics for scene files in the given folder.
        
        Args:
            input_folder (str): Path to folder containing scene JSON files
            output_file (str, optional): Path to save results
            verbose (bool): Whether to print progress and results
            
        Returns:
            dict: Dictionary containing evaluation results
        """
        if not os.path.exists(input_folder):
            raise ValueError(f"Folder {input_folder} does not exist!")
        
        if verbose:
            print("Starting Out-of-Boundary Analysis...")
            print(f"Input folder: {input_folder}")
        
        # Analyze the directory directly
        results = self.calculate_oob_ratios(input_folder)
        
        if results['total_scenes'] == 0:
            if verbose:
                print(f"No scene files found in {input_folder}")
            return {'error': 'No scenes found'}
        
        if verbose:
            print(f"Total scenes: {results['total_scenes']}")
            print(f"Scenes with OOB: {results['scenes_with_oob']} ({results['scene_oob_ratio']:.2%})")
            print(f"Total objects: {results['total_objects']}")
            print(f"OOB objects: {results['oob_objects']} ({results['object_oob_ratio']:.2%})")
            print("Analysis complete!")
        
        return results 