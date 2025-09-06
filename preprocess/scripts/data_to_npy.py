"""
Data preprocessing script for converting Front3D dataset to numpy format.

This script processes Front3D room layout data and converts it to numpy arrays
for efficient training and inference. It supports both layout-only and 
layout+text embedding modes.
"""

import os
import argparse
import torch
import numpy as np
import torchvision
from torchvision import transforms

from front3d import FRONT3D
from front3d import ToTensorNoNorm


def main(H, W, train_data_dir, dataset_name, output_dir='./datasets'):
    """
    Convert Front3D dataset to numpy format.

    Args:
        H (int): Height of the output tensors
        W (int): Width of the output tensors  
        train_data_dir (str): Path to the training data directory
        dataset_name (str): Name identifier for the output files
        output_dir (str): Output directory for processed files
    """
    room_condition = True
    data_transforms = transforms.Compose([
        torchvision.transforms.Resize((H, W), interpolation=0),
        ToTensorNoNorm()])

    data_set = FRONT3D(
        root_dir=train_data_dir,
        transform=data_transforms,
        room_condition=room_condition)
    
    file_number_count = len(os.listdir(train_data_dir)) + data_set.duplicate

    train_set = torch.utils.data.Subset(data_set, torch.arange(0, file_number_count))

    n_train = len(train_set)

    # Calculate actual tensor size accounting for duplicates
    total_samples = 0
    for item in train_set:
        is_scalar = isinstance(item[0], int) or (torch.is_tensor(item[0]) and item[0].ndim == 0)
        if not is_scalar:
            total_samples += len(item)
        else:
            total_samples += 1

    print(f'Dataset contains {n_train} items, resulting in {total_samples} total samples')

    # Main tensor for room layout
    train_tensor = torch.zeros((total_samples, 2 if room_condition else 1, H, W)).long()

    print(f'Processing {dataset_name} dataset with dimensions {H}x{W}')

    # Initialize tracking variables
    current_idx = 0
    min_val, max_val = float('inf'), float('-inf')

    # Process each item in the dataset
    for i, item in enumerate(train_set):
        if i % 1000 == 0:
            print(f'Processing item {i}/{n_train}, current tensor index: {current_idx}')

        # Check if item contains multiple sub-items (unified room type)
        is_scalar = isinstance(item[0], int) or (torch.is_tensor(item[0]) and item[0].ndim == 0)

        if not is_scalar:
            # Process unified room type with multiple sub-items
            for j, sub_item in enumerate(item):
                if current_idx >= total_samples:
                    print(f"Warning: Index {current_idx} exceeds tensor size {total_samples}")
                    break

                room_type_id, layout_data = sub_item[0], sub_item[1]

                # Store layout data
                if room_condition:
                    train_tensor[current_idx][0] = room_type_id
                    train_tensor[current_idx][1] = layout_data
                else:
                    train_tensor[current_idx] = layout_data

                # Update min/max values
                min_val = min(min_val, layout_data.min().item())
                max_val = max(max_val, layout_data.max().item())

                current_idx += 1
        else:
            # Process single item
            if current_idx >= total_samples:
                print(f"Warning: Index {current_idx} exceeds tensor size {total_samples}")
                break

            room_type_id, layout_data = item[0], item[1]

            # Store layout data
            if room_condition:
                train_tensor[current_idx][0] = room_type_id
                train_tensor[current_idx][1] = layout_data
            else:
                train_tensor[current_idx] = layout_data

            # Update min/max values
            min_val = min(min_val, layout_data.min().item())
            max_val = max(max_val, layout_data.max().item())

            current_idx += 1

    print(f'Data range: min={min_val}, max={max_val}')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f'Used {current_idx} out of {total_samples} tensor slots')
    print('Converting tensors to numpy arrays...')

    # Trim tensor to actual used size if needed
    if current_idx < total_samples:
        train_tensor = train_tensor[:current_idx]

    layout_data = train_tensor.numpy().astype('uint8')

    # Generate filename based on configuration
    base_filename = f'{dataset_name}_{H}x{W}'
    filepath = os.path.join(output_dir, base_filename)

    # Save layout data
    print(f'Saving layout data to {filepath}.npy...')
    np.save(filepath, layout_data, allow_pickle=True, fix_imports=True)

    # Verify saved data
    try:
        loaded_data = np.load(f'{filepath}.npy')
        print(f"✓ Successfully saved layout data with shape: {loaded_data.shape}")
    except Exception as e:
        print(f"✗ Error verifying saved data: {e}")

    print(f'Processing completed! Output saved to: {filepath}.npy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Front3D dataset to numpy format for efficient training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--train_data_dir', type=str, 
                      default='./data/train',
                      help='Directory containing Front3D training data')
    parser.add_argument('--height', type=int, default=120,
                      help='Height of the output tensor resolution')
    parser.add_argument('--width', type=int, default=120,
                      help='Width of the output tensor resolution')
    parser.add_argument('--dataset_name', type=str, default='unified_w_arch',
                      help='Dataset name identifier for output filename')
    parser.add_argument('--output_dir', type=str, default='../../datasets',
                      help='Output directory for processed numpy files')

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.train_data_dir):
        print(f"Error: Training data directory '{args.train_data_dir}' does not exist.")
        exit(1)

    print("Starting Front3D data preprocessing...")
    main(args.height, args.width, args.train_data_dir, args.dataset_name, args.output_dir)
