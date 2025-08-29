"""
Unified Layout Sampling Script

This script generates furniture layouts using a pretrained diffusion model.
Supports both Front3D dataset floor plans and custom floor plan images.

Usage:
    python sample_layout.py [hydra options]

Modes:
    - Original mode (use_custom_data=False): Uses Front3D dataset floor plans
    - Custom mode (use_custom_data=True): Uses custom floor plan images
"""

import hydra
from omegaconf import DictConfig

from semlayoutdiff.sldn.sampling_utils import LayoutSampler


@hydra.main(config_path="../configs/sldn", config_name="sample_layout.yaml", version_base="1.2")
def main(cfg: DictConfig):
    """Main entry point for the layout sampling script."""
    sampler = LayoutSampler(cfg)
    sampler.run()


if __name__ == '__main__':
    main()