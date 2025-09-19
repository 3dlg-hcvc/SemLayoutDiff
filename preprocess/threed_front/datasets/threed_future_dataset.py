# 
# modified from: 
#   https://github.com/nv-tlabs/ATISS.
#   https://github.com/tangjiapeng/DiffuScene
#

import pickle
import json
import numpy as np
import torch
from torch.utils.data import dataloader

# Add this import to help with pickle compatibility
import sys
# Create aliases for modules to handle pickle compatibility
sys.modules['threed_future_dataset'] = sys.modules[__name__]
sys.modules['threed_front'] = sys.modules.get('preprocess.threed_front', sys.modules.get('threed_front', sys.modules[__name__.split('.')[0]]))

# If the module structure has changed significantly, we might need to create more specific aliases
if 'preprocess.threed_front.datasets' in sys.modules:
    sys.modules['threed_front.datasets'] = sys.modules['preprocess.threed_front.datasets']


class ThreedFutureObject(object):
    """Simple object wrapper for 3D Future model data loaded from JSON."""
    
    def __init__(self, data_dict):
        # Convert lists back to numpy arrays for size, position, rotation, scale
        for key, value in data_dict.items():
            if key in ['size', 'position', 'rotation', 'scale'] and isinstance(value, list):
                setattr(self, key, np.array(value))
            elif key == 'z_angle':
                # Handle z_angle specially since it has a property
                self._z_angle = value
            elif key == 'raw_model_path':
                # Handle raw_model_path specially since it has a property
                self._raw_model_path = value
            else:
                setattr(self, key, value)
    
    def raw_model_norm_pc(self):
        """Load normalized point cloud data."""
        if hasattr(self, 'raw_model_norm_pc_path'):
            points = np.load(self.raw_model_norm_pc_path)["points"].astype(np.float32)
            return points
        else:
            raise AttributeError("raw_model_norm_pc_path not found")
    
    def raw_model_norm_pc_lat(self):
        """Load normalized point cloud latent data."""
        if hasattr(self, 'raw_model_norm_pc_lat_path'):
            latent = np.load(self.raw_model_norm_pc_lat_path)["latent"].astype(np.float32)
            return latent
        else:
            raise AttributeError("raw_model_norm_pc_lat_path not found")
    
    def raw_model_norm_pc_lat32(self):
        """Load 32-dimensional normalized point cloud latent data."""
        if hasattr(self, 'raw_model_norm_pc_lat32_path'):
            latent = np.load(self.raw_model_norm_pc_lat32_path)["latent"].astype(np.float32)
            return latent
        else:
            raise AttributeError("raw_model_norm_pc_lat32_path not found")
    
    @property
    def raw_model_path(self):
        """Get the path to the raw model file."""
        if hasattr(self, 'path_to_models') and hasattr(self, 'model_jid'):
            return f"{self.path_to_models}/{self.model_jid}/raw_model.obj"
        else:
            # Try to get it from the stored attribute
            return getattr(self, '_raw_model_path', None)
    
    def centroid(self, offset=None):
        """Compute centroid with optional offset."""
        if offset is None:
            offset = [0, 0, 0]
        # Simple implementation - in practice this would use bbox vertices
        return np.array(self.position) + np.array(offset)
    
    @property
    def z_angle(self):
        """Get the z-axis rotation angle."""
        if hasattr(self, '_z_angle'):
            return self._z_angle
        # Simple implementation - in practice this would compute from rotation quaternion
        return 0.0


class ThreedFutureDataset(object):
    def __init__(self, objects):
        assert len(objects) > 0
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        return self.objects[idx]

    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]

    def get_closest_furniture_to_box(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)
        if len(objects) == 0:
            raise RuntimeError("No {} in this dataset.".format(query_label))

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = np.sum((oi.size - query_size)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = \
                (oi.size[0] - query_size[0])**2 + \
                (oi.size[2] - query_size[1])**2
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_objfeats(self, query_label, query_objfeat):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            if query_objfeat.shape[0] == 32:
                mses[oi] = np.sum(
                    (oi.raw_model_norm_pc_lat32() - query_objfeat)**2, axis=-1
                )
            else:
                mses[oi] = np.sum(
                    (oi.raw_model_norm_pc_lat() - query_objfeat)**2, axis=-1
                )
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_objfeats_and_size(self, query_label, query_objfeat, query_size):
        objects = self._filter_objects_by_label(query_label)

        objs = []
        mses_feat = []
        mses_size = []
        for i, oi in enumerate(objects):
            if query_objfeat.shape[0] == 32:
                mses_feat.append(
                    np.sum((oi.raw_model_norm_pc_lat32() - query_objfeat)**2, axis=-1)
                )
            else:
                mses_feat.append(
                    np.sum((oi.raw_model_norm_pc_lat() - query_objfeat)**2, axis=-1)
                )
            mses_size.append(
                (oi.size[0] - query_size[0])**2 + \
                (oi.size[2] - query_size[1])**2
            )
            objs.append(oi)

        ind = np.lexsort( (mses_feat, mses_size) )
        return objs[ind[0]]
    
    @classmethod
    def from_pickled_dataset(cls, path_to_pickled_dataset):
        with open(path_to_pickled_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset
    
    @classmethod
    def from_json_dataset(cls, path_to_json_dataset):
        """Load dataset from JSON file."""
        with open(path_to_json_dataset, "r") as f:
            json_objects = json.load(f)
        
        # Convert JSON objects to ThreedFutureObject instances
        objects = [ThreedFutureObject(obj_dict) for obj_dict in json_objects]
        return cls(objects)
    
    @classmethod
    def from_dataset_file(cls, path_to_dataset):
        """Load dataset from either pickle or JSON file based on extension."""
        if path_to_dataset.endswith('.json'):
            return cls.from_json_dataset(path_to_dataset)
        elif path_to_dataset.endswith('.pkl'):
            return cls.from_pickled_dataset(path_to_dataset)
        else:
            raise ValueError(f"Unsupported file format: {path_to_dataset}. Use .pkl or .json")


class ThreedFutureNormPCDataset(ThreedFutureDataset):
    def __init__(self, objects, num_samples=2048):
        super().__init__(objects)

        self.num_samples = num_samples

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        obj = self.objects[idx]
        points = obj.raw_model_norm_pc()

        points_subsample = points[np.random.choice(points.shape[0], self.num_samples), :]

        points_torch = torch.from_numpy(points_subsample).float()
        data_dict =  {"points": points_torch, "idx": idx} 
        return data_dict

    def get_model_jid(self, idx):
        obj = self.objects[idx]
        model_jid = obj.model_jid
        data_dict =  {"model_jid": model_jid} 
        return data_dict

    def collate_fn(self, samples):
        ''' Collater that puts each data field into a tensor with outer dimension
            batch size.
        Args:
            samples: samples
        '''
    
        samples = list(filter(lambda x: x is not None, samples))
        return dataloader.default_collate(samples)
