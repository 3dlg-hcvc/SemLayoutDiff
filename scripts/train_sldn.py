"""
Training script for Semantic Layout Diffusion (SLDN) models.

This script provides a complete training pipeline for diffusion-based layout generation models,
supporting various conditioning modes including floor plans, room types, text, and mixed conditions.
"""

import argparse
import os
import yaml

# Core utilities
from semlayoutdiff.sldn.diffusion_utils import add_parent_path, set_seeds

# Components
from semlayoutdiff.sldn.experiment import Experiment, add_exp_args
from semlayoutdiff.sldn.model import get_model, get_model_id, add_model_args
from semlayoutdiff.sldn.diffusion_utils.optim.multistep import get_optim, get_optim_id, add_optim_args

# Data handling
add_parent_path(level=1)
from semlayoutdiff.sldn.dataloader.dataset_front3d import get_data, get_data_id, add_data_args


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file {config_path}: {e}")


def setup_argument_parser():
    """
    Set up and configure the argument parser with all required arguments.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Train Semantic Layout Diffusion (SLDN) models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add component-specific arguments
    add_exp_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_optim_args(parser)
    
    # Add configuration file support
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to YAML configuration file (overrides command line arguments)'
    )
    
    return parser


def parse_args():
    """
    Parse command line arguments and merge with config file if provided.
    
    Returns:
        argparse.Namespace: Parsed and processed arguments
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Override with config file values if provided
    if args.config:
        config = load_config(args.config)
        arg_dict = vars(args)
        
        # Update only existing arguments to prevent conflicts
        for key, value in config.items():
            if key in arg_dict:
                arg_dict[key] = value
                print(f"Config override: {key} = {value}")
    
    return args


def initialize_data(args):
    """
    Initialize data loaders and extract data specifications.
    
    Args:
        args: Parsed arguments
        
    Returns:
        tuple: (train_loader, eval_loader, data_shape, num_classes, data_id)
    """
    train_loader, eval_loader, data_shape, num_classes = get_data(args)
    args.num_classes = num_classes  # Update args with actual number of classes
    data_id = get_data_id(args)
    
    print(f"Data initialized - Shape: {data_shape}, Classes: {num_classes}")
    return train_loader, eval_loader, data_shape, num_classes, data_id


def initialize_model(args, data_shape):
    """
    Initialize the diffusion model.
    
    Args:
        args: Parsed arguments
        data_shape: Shape of the input data
        
    Returns:
        tuple: (model, model_id)
    """
    model = get_model(args, data_shape=data_shape)
    model_id = get_model_id(args)
    
    print(f"Model initialized - ID: {model_id}")
    return model, model_id


def initialize_optimizer(args, model):
    """
    Initialize optimizer and learning rate schedulers.
    
    Args:
        args: Parsed arguments
        model: The model to optimize
        
    Returns:
        tuple: (optimizer, scheduler_iter, scheduler_epoch, optim_id)
    """
    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
    optim_id = get_optim_id(args)
    
    print(f"Optimizer initialized - ID: {optim_id}")
    return optimizer, scheduler_iter, scheduler_epoch, optim_id


def main():
    """Main training pipeline."""
    # Parse arguments and set up environment
    args = parse_args()
    set_seeds(args.seed)
    args.log_home = "tmp_log"
    
    print(f"Starting SLDN training with seed: {args.seed}")
    
    # Initialize components
    train_loader, eval_loader, data_shape, num_classes, data_id = initialize_data(args)
    model, model_id = initialize_model(args, data_shape)
    optimizer, scheduler_iter, scheduler_epoch, optim_id = initialize_optimizer(args, model)
    
    # Create and run experiment
    experiment = Experiment(
        args=args,
        data_id=data_id,
        model_id=model_id,
        optim_id=optim_id,
        train_loader=train_loader,
        eval_loader=eval_loader,
        model=model,
        optimizer=optimizer,
        scheduler_iter=scheduler_iter,
        scheduler_epoch=scheduler_epoch
    )
    
    print("Starting training...")
    experiment.run()
    print("Training completed!")


if __name__ == "__main__":
    main()
