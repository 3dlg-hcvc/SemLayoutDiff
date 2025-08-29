import torch
from diffusion_utils.loss import elbo_bpd, floor_loss
from diffusion_utils.utils import add_parent_path
import wandb
import numpy as np
import json
import os

add_parent_path(level=2)
from diffusion_utils.experiment import DiffusionExperiment
from diffusion_utils.experiment import add_exp_args as add_exp_args_parent


def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument("--clip_value", type=float, default=None)
    parser.add_argument("--clip_norm", type=float, default=None)
    parser.add_argument("--floor_loss", type=eval, default=False)


def load_json(json_path):
    """Load a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


class Experiment(DiffusionExperiment):
    """
    Experiment class for training and evaluating diffusion models on layout generation.
    
    This class extends DiffusionExperiment to provide specialized functionality for:
    - Training with multiple condition types (floor plans, room types, text, mixed)
    - Evaluation with comprehensive metrics
    - Sample generation and logging to wandb
    - Color mapping and visualization
    """

    def _process_data_batch(self, data):
        """Process a data batch and prepare conditions for model input."""
        x, floor_plan, room_type, text_condition, mixed_condition_id = data
        
        # Process floor plan condition
        floor_plan = (
            floor_plan.to(self.args.device) if torch.sum(floor_plan) != 0 else None
        )
        
        # Process room type condition
        room_type = (
            room_type.to(self.args.device) if torch.sum(room_type) >= 0 else None
        )
        
        # Process mixed condition
        if not self.args.mix_condition:
            mixed_condition_id = None
        else:
            mixed_condition_id = (
                mixed_condition_id.to(self.args.device) if torch.sum(mixed_condition_id) >= 0 else None
            )
        
        # Process text condition
        if not self.args.text_condition:
            text_condition = None
        else:
            text_condition = text_condition.to(self.args.device) if text_condition is not None else None
        
        return x.to(self.args.device), floor_plan, room_type, text_condition, mixed_condition_id

    def _compute_losses(self, x, floor_plan, room_type, text_condition, mixed_condition_id):
        """Compute ELBO and floor losses."""
        loss_elbo = elbo_bpd(self.model, x, floor_plan, room_type, text_condition, mixed_condition_id)
        
        if self.args.floor_loss:
            loss_floor = floor_loss(self.model, x, floor_plan, room_type)
            total_loss = loss_elbo + loss_floor
            return total_loss, loss_elbo, loss_floor
        else:
            return loss_elbo, loss_elbo, None

    def _print_progress(self, phase, epoch, loss_count, dataset_size, loss_sum, loss_floor_sum=None):
        """Print training/evaluation progress."""
        if self.args.floor_loss and loss_floor_sum is not None:
            print(
                f"{phase}. Epoch: {epoch + 1}/{self.args.epochs}, "
                f"Datapoint: {loss_count}/{dataset_size}, "
                f"Bits/dim: {loss_sum / loss_count:.3f}, "
                f"Floor_loss: {loss_floor_sum / loss_count:.3f}",
                end="\r"
            )
        else:
            print(
                f"{phase}. Epoch: {epoch + 1}/{self.args.epochs}, "
                f"Datapoint: {loss_count}/{dataset_size}, "
                f"Bits/dim: {loss_sum / loss_count:.3f}",
                end="\r"
            )

    def train_fn(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        loss_floor_sum = 0.0
        
        for data in self.train_loader:
            # Process data batch
            x, floor_plan, room_type, text_condition, mixed_condition_id = self._process_data_batch(data)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute losses
            loss, loss_elbo, loss_floor = self._compute_losses(x, floor_plan, room_type, text_condition, mixed_condition_id)
            
            # Accumulate floor loss if enabled
            if self.args.floor_loss and loss_floor is not None:
                loss_floor_sum += loss_floor.detach().cpu().item() * len(x)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.clip_value:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            
            # Update parameters
            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()
            
            # Accumulate losses and print progress
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            self._print_progress("Training", epoch, loss_count, len(self.train_loader.dataset), 
                               loss_sum, loss_floor_sum if self.args.floor_loss else None)
        
        print("")
        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        
        # Return metrics
        return (
            {"bpd": loss_sum / loss_count, "floor_loss": loss_floor_sum / loss_count}
            if self.args.floor_loss
            else {"bpd": loss_sum / loss_count}
        )

    def _evaluate_on_loader(self, loader, phase_name, epoch):
        """Helper method to evaluate on a specific data loader."""
        loss_sum = 0.0
        loss_count = 0
        loss_floor_sum = 0.0
        
        for data in loader:
            # Process data batch
            x, floor_plan, room_type, text_condition, mixed_condition_id = self._process_data_batch(data)
            
            # Compute losses
            loss, loss_elbo, loss_floor = self._compute_losses(x, floor_plan, room_type, text_condition, mixed_condition_id)
            
            # Accumulate floor loss if enabled
            if self.args.floor_loss and loss_floor is not None:
                loss_floor_sum += loss_floor.detach().cpu().item() * len(x)
            
            # Accumulate losses and print progress
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            dataset_size = len(loader.dataset)
            self._print_progress(phase_name, epoch, loss_count, dataset_size, 
                               loss_sum, loss_floor_sum if self.args.floor_loss else None)
        
        print("")
        return loss_sum, loss_count, loss_floor_sum

    def eval_fn(self, epoch):
        """Evaluate the model on training and validation sets."""
        self.model.eval()
        
        with torch.no_grad():
            # Evaluate on training set
            train_loss_sum, train_loss_count, train_floor_sum = self._evaluate_on_loader(
                self.train_loader, "Train evaluating", epoch
            )
            
            # Evaluate on validation set
            eval_loss_sum, eval_loss_count, eval_floor_sum = self._evaluate_on_loader(
                self.eval_loader, "     Evaluating", epoch
            )
        
        # Return validation metrics
        return (
            {"bpd": eval_loss_sum / eval_loss_count, "floor_loss": eval_floor_sum / eval_loss_count}
            if self.args.floor_loss
            else {"bpd": eval_loss_sum / eval_loss_count}
        )

    def instance_map_to_color(self, batch_instance_maps, color_map):
        """
        Convert batch of instance ID maps to colored RGB images.
        
        Args:
            batch_instance_maps (np.ndarray): Batch of instance ID maps with shape (B, 1, H, W)
            color_map (dict): Mapping from instance IDs to RGB color tuples
            
        Returns:
            np.ndarray: Batch of colored RGB images with shape (B, H, W, 3)
        """
        batch_size, _, height, width = batch_instance_maps.shape
        assert batch_instance_maps.shape[1] == 1, "Expected a single channel for instance maps."

        # Remove the channels dimension
        batch_instance_maps = np.squeeze(batch_instance_maps, axis=1)

        # Create an empty RGB image batch
        batch_colored_maps = np.zeros((batch_size, height, width, 3), dtype=np.uint8)

        # Apply colors to each instance ID map in the batch
        for i in range(batch_size):
            instance_map = batch_instance_maps[i]
            for instance_id, color in color_map.items():
                mask = (instance_map == instance_id)
                batch_colored_maps[i, mask] = color

        return batch_colored_maps

    def samples_process(self, batch, colormap):
        """
        Process generated samples for visualization.
        
        Args:
            batch (torch.Tensor): Generated layout samples
            colormap (dict): Color mapping for visualization
            
        Returns:
            tuple: (colored_samples, raw_layout) - processed samples and raw layout
        """
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(1)

        raw_layout = batch[0]
        batch = self.instance_map_to_color(batch.cpu().numpy(), colormap)
        batch_transposed = batch.transpose(0, 3, 1, 2)
        batch_tensor = torch.tensor(batch_transposed).to(torch.uint8)
        batch_tensor = batch_tensor.permute(0, 2, 3, 1)
        
        return batch_tensor, raw_layout

    def assign_color_rgb(self, color_palette_path, idx_to_label_path):
        """
        Assign RGB colors to room types based on color palette and label mapping.
        
        This method handles different architectural configurations:
        - w_arch: Include architectural elements (doors, windows) with original IDs
        - wo_room: Exclude doors and windows, adjust remaining IDs accordingly
        - Default: Use original mapping without modifications
        
        Args:
            color_palette_path (str): Path to JSON file containing color palette
            idx_to_label_path (str): Path to JSON file containing ID to label mapping
            
        Returns:
            dict: Mapping from adjusted IDs to RGB color tuples
        """
        color_palette = load_json(color_palette_path)
        idx_to_label = load_json(idx_to_label_path)

        if hasattr(self.args, 'w_arch') and self.args.w_arch:
            # Use original mapping with architectural elements
            colors = {}
            for idx, label in idx_to_label.items():
                if label in color_palette:
                    colors[int(idx)] = tuple(color_palette[label])
            return colors
            
        elif hasattr(self.args, 'wo_room') and self.args.wo_room:
            # Exclude doors and windows, adjust remaining IDs
            door_id = window_id = None
            
            # Find door and window IDs
            for idx, label in idx_to_label.items():
                if label.lower() == 'door':
                    door_id = int(idx)
                elif label.lower() == 'window':
                    window_id = int(idx)

            colors = {}
            for idx, label in idx_to_label.items():
                idx = int(idx)
                
                # Skip architectural elements
                if label.lower() in ['door', 'window']:
                    continue
                
                if label in color_palette:
                    # Adjust ID based on removed elements
                    adjusted_idx = idx
                    if door_id is not None and idx > door_id:
                        adjusted_idx -= 1
                    if window_id is not None and idx > window_id:
                        adjusted_idx -= 1
                    colors[adjusted_idx] = tuple(color_palette[label])

            return colors
        else:
            # Default: use original mapping
            colors = {}
            for idx, label in idx_to_label.items():
                if label in color_palette:
                    colors[int(idx)] = tuple(color_palette[label])
            return colors

    def log_samples_fn(self, epoch):
        """
        Generate layout samples and log them to wandb for visualization.
        
        This method:
        1. Extracts conditions from validation data
        2. Generates samples using the trained diffusion model
        3. Processes samples for visualization (applies colors)
        4. Creates floor plan visualizations if available
        5. Logs everything to wandb in an organized format
        
        Args:
            epoch (int): Current training epoch number
        """
        self.model.eval()
        
        with torch.no_grad():
            # Extract validation data batch
            data = next(iter(self.eval_loader))
            x, floor_plan, room_type, text_condition, mixed_condition_id = data
            
            # Process conditions based on configuration
            if not self.args.mix_condition:
                mixed_condition_id = None
            if not self.args.text_condition:
                text_condition = None
            
            # Move conditions to device and handle empty tensors
            mixed_condition_id = (
                mixed_condition_id.to(self.args.device) 
                if mixed_condition_id is not None and torch.sum(mixed_condition_id) >= 0 
                else None
            )
            text_condition = (
                text_condition.to(self.args.device) 
                if text_condition is not None 
                else None
            )
            
            # Prepare conditions for specified number of samples
            if floor_plan is not None and torch.sum(floor_plan) != 0:
                floor_plan = floor_plan[:self.args.num_samples].to(self.args.device)
            else:
                floor_plan = None
                
            room_type = (
                room_type[:self.args.num_samples].to(self.args.device) 
                if room_type is not None and torch.sum(room_type) >= 0 
                else None
            )
            
            if text_condition is not None:
                text_condition = text_condition[:self.args.num_samples]
            if mixed_condition_id is not None:
                mixed_condition_id = mixed_condition_id[:self.args.num_samples]

            # Generate samples through diffusion sampling process
            samples_chain = self.model.sample_chain(
                self.args.num_samples,
                floor_plan=floor_plan,
                room_type=room_type,
                text_condition=text_condition,
                mixed_condition_id=mixed_condition_id
            )
            
            # Reshape samples: [num_samples, timesteps, channels, height, width]
            samples_chain = samples_chain.permute(1, 0, 2, 3, 4)
            
            # Load color mapping for visualization
            color_palette_path = '../preprocess/metadata/color_palette.json'
            idx_to_label_path = '../preprocess/metadata/unified_idx_to_generic_label.json'
            colormap = self.assign_color_rgb(color_palette_path, idx_to_label_path)
            
            # Process each sample for logging
            all_samples = {}
            for i in range(self.args.num_samples):
                # Extract final generated layout (last timestep)
                final_sample = samples_chain[i][0].unsqueeze(0)
                colored_samples, raw_layout = self.samples_process(final_sample, colormap)
                
                # Create floor plan visualization if available
                floor_plan_img = None
                if floor_plan is not None:
                    fp = floor_plan[i].cpu().squeeze().numpy()
                    
                    if self.args.w_arch:
                        # Create colored architectural floor plan
                        colored_floor_plan = np.zeros((fp.shape[0], fp.shape[1], 3), dtype=np.uint8)
                        colored_floor_plan[fp == 0] = [255, 255, 255]  # Background: White
                        colored_floor_plan[fp == 1] = [211, 211, 211]  # Floor: Gray
                        colored_floor_plan[fp == 2] = [153, 0, 0]      # Door: Dark Red
                        colored_floor_plan[fp == 3] = [255, 153, 153]  # Window: Light Red
                        floor_plan_img = colored_floor_plan
                    else:
                        # Simple binary floor plan (grayscale)
                        floor_plan_img = fp * 255
                
                # Extract room type value
                room_type_value = (
                    int(room_type[i].cpu().item()) 
                    if room_type is not None 
                    else None
                )
                
                # Organize sample data for wandb logging
                all_samples[f"sample_{i}/1_room_type"] = room_type_value
                all_samples[f"sample_{i}/2_floor_plan"] = (
                    wandb.Image(floor_plan_img) if floor_plan_img is not None else None
                )
                all_samples[f"sample_{i}/3_generated_layout"] = wandb.Image(colored_samples[0].cpu().numpy())
                
                if mixed_condition_id is not None:
                    all_samples[f"sample_{i}/4_mixed_condition_id"] = mixed_condition_id[i].cpu().item()
            
            # Log all samples with epoch information
            all_samples["epoch"] = epoch + 1
            wandb.log(all_samples)
