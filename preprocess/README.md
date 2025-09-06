# Dataset Preprocessing

This directory contains preprocessing tools for the 3D-FRONT and 3D-FUTURE datasets. The preprocessing pipeline generates semantic layout maps, room architecture maps, and object instance information for training and evaluation.

## Overview

Our preprocessing pipeline creates two main types of data:
1. Semantic layout maps and room architecture maps (PNG format)
2. Object instance information (JSON format)

We leverage [BlenderProc-3DFront](https://github.com/yinyunie/BlenderProc-3DFront) to process 3D-FRONT scenes into top-down view renders.

## Prerequisites

### Required Datasets
- [3D-FRONT Dataset](https://tianchi.aliyun.com/dataset/65347)

### Software Requirements
- Python 3.7+
- Blender 3.0.0+ 
- Conda (recommended for environment management)

## Setup

### 1. Environment Setup
Create and activate the conda environment (if you did not already create it, follow the main README):
```bash
# From the project root directory
conda env create -f environment.yml
conda activate semlayoutdiff
```

### 2. Install BlenderProc-3DFront
```bash
pip install git+https://github.com/3dlg-hcvc/BlenderProc-3DFront.git@3dfront_2d_layout
```

### 3. Install Blender
Download and install [Blender 3.0.0](https://download.blender.org/release/Blender3.0/) or later version.

## Usage

### Step 1: Render 3D-FRONT Scenes

#### Download cctexture
```bash
blenderproc run preprocess/scripts/download_cc_textures.py datasets/cctextures
```

#### Single Scene Rendering
```bash
blenderproc run preprocess/semlayout/render_dataset_improved_mat.py \
    <3dfront_scene_path> \
    <3dfuture_model_path> \
    <3dfront_texture_path> \
    <3dfront_json_path> \
    <cctextures_path> \
    <output_directory> \
    --room_type <room_type> \
    --blender-install-path <blender_path>
```

#### Batch Rendering
For processing multiple scenes:
```bash
python preprocess/semlayout/multi_render.py \
    preprocess/semlayout/render_dataset_improved_mat.py \
    <3dfront_dataset_path> \
    <3dfuture_model_path> \
    <3dfront_texture_path> \
    <cctextures_path> \
    <output_directory> \
    --n_processes <num_processes> \
    --room_type <room_type> \
    --blender-install-path <blender_path>
```

#### Extract Maps
After rendering, extract semantic and architecture maps:
```bash
python preprocess/semlayout/visualization/front3d/data_process_front3d.py --room_type <room_type>
```
Make sure to process all three different room types (bedroom, livingroom, and diningroom)

### Step 2: Process Object Instance Information

#### Setup Python Path
Navigate to the preprocessing directory and set up the Python path:
```bash
cd preprocess
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### Unify Datasets
Process and unify the rendered data:
```bash
python scripts/data_processor.py \
    --input_dir <rendered_data_path> \
    --output_dir <unified_output_path> \
    --room_types bedroom livingroom \
    --include_arch
```

#### Convert to Training Format
Convert data to numpy format for model training:
```bash
python scripts/data_to_npy.py \
    --train_data_dir <unified_output_path>/train \
    --dataset_name unified_w_arch \
    --output_dir <final_datasets_path>
```

Create pickled datasets for inference:
```bash
# Process 3D-FRONT dataset
python scripts/pickle_threed_front_dataset.py \
    <3dfront_dataset_path> \
    <3dfuture_model_path> \
    <3dfuture_model_info_json> \
    --output_path <output_path>

# Process 3D-FUTURE dataset
python scripts/pickle_threed_future_dataset.py threed_front_<room_type>
```