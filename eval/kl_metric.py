"""KL divergence evaluation for object category distributions."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

CLASS_LABELS = [
    'armchair', 'bookshelf', 'cabinet', 'ceiling_lamp', 'chair',
    'chaise_longue_sofa', 'children_cabinet', 'chinese_chair',
    'coffee_table', 'console_table', 'corner_side_table', 'desk',
    'dining_chair', 'dining_table', 'double_bed', 'dressing_chair',
    'dressing_table', 'kids_bed', 'l_shaped_sofa', 'lazy_sofa',
    'lounge_chair', 'loveseat_sofa', 'multi_seat_sofa', 'nightstand',
    'pendant_lamp', 'round_end_table', 'shelf', 'single_bed', 'sofa',
    'stool', 'table', 'tv_stand', 'wardrobe', 'wine_cabinet'
]


def categorical_kl(p, q):
    """Compute KL divergence between two categorical distributions."""
    return (p * (np.log((p + 1e-6) / (q + 1e-6)))).sum()


class KLDivergenceEvaluator:
    """Evaluator for computing KL divergence between object categories of real and generated scenes."""
    
    def __init__(self, class_labels=None):
        """Initialize the KL divergence evaluator."""
        self.class_labels = class_labels if class_labels is not None else CLASS_LABELS
    
    def read_scene_state(self, path_to_scene):
        """Read and process scene state from JSON file."""
        with open(path_to_scene, "r") as f:
            data = json.load(f)

        scene_objects = []
        for obj in data["scene"]["object"]:
            label = obj["class_label"]
            
            if label in ["window", "door"]:
                continue
                
            if label in self.class_labels:
                one_hot_vector = [0] * len(self.class_labels)
                index = self.class_labels.index(label)
                one_hot_vector[index] = 1
                scene_objects.append(np.array(one_hot_vector))

        return {"class_labels": np.array(scene_objects)}
    
    def load_scenes_from_directory(self, scene_dir):
        """Load all scenes from a directory."""
        scenes = []
        for f in os.listdir(scene_dir):
            if f.endswith(".json"):
                file_path = os.path.join(scene_dir, f)
                scene_data = self.read_scene_state(file_path)
                if scene_data["class_labels"].size > 0:
                    scenes.append(scene_data)
        return scenes
    
    def compute_class_frequencies(self, scenes):
        """Compute frequency distribution of class labels across scenes."""
        if not scenes:
            return np.zeros(len(self.class_labels))
        
        total_objects = sum([d["class_labels"].shape[0] for d in scenes])
        if total_objects == 0:
            return np.zeros(len(self.class_labels))
        
        return sum([d["class_labels"].sum(0) for d in scenes]) / total_objects
    
    def create_frequency_plot(self, gt_frequencies, syn_frequencies, output_path=None):
        """Create a bar plot comparing frequencies of different categories."""
        fig, ax = plt.subplots(figsize=(10, 8))

        bar_width = 0.35
        r1 = np.arange(len(self.class_labels))
        r2 = [x + bar_width for x in r1]
        
        ax.barh(r1, syn_frequencies, height=bar_width, 
                label='Synthetic', color='lightblue')
        ax.barh(r2, gt_frequencies, height=bar_width, 
                label='Ground Truth', color='orange')

        ax.set_title('Object Category Frequency Comparison')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Class Labels')
        ax.set_yticks([r + bar_width/2 for r in range(len(self.class_labels))])
        ax.set_yticklabels(self.class_labels)
        ax.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def evaluate(self, gt_scenestate_dir, synthesized_scenestate_dir, 
                 output_directory=None, create_plot=False, verbose=True):
        """Evaluate KL divergence between ground truth and synthesized scenes."""
        if verbose:
            print("Loading scenes...")
        
        gt_scenes = self.load_scenes_from_directory(gt_scenestate_dir)
        synthesized_scenes = self.load_scenes_from_directory(synthesized_scenestate_dir)
        
        if not gt_scenes or not synthesized_scenes:
            raise ValueError("No valid scenes found")
        
        # Compute frequency distributions
        gt_frequencies = self.compute_class_frequencies(gt_scenes)
        syn_frequencies = self.compute_class_frequencies(synthesized_scenes)
        
        # Compute KL divergence
        kl_divergence = categorical_kl(gt_frequencies, syn_frequencies)
        
        if verbose:
            print(f"KL Divergence: {kl_divergence:.6f}")
        
        results = {
            'kl_divergence': kl_divergence,
            'gt_frequencies': gt_frequencies.tolist(),
            'syn_frequencies': syn_frequencies.tolist(),
            'num_gt_scenes': len(gt_scenes),
            'num_syn_scenes': len(synthesized_scenes)
        }
        
        # Create frequency comparison plot
        if create_plot:
            plot_path = os.path.join(output_directory, 'frequency_comparison.png') if output_directory else None
            self.create_frequency_plot(gt_frequencies, syn_frequencies, plot_path)
        
        return results



