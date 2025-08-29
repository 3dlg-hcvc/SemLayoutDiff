"""FID and KID evaluation for scene renderings."""

import os
import shutil
import tempfile
import numpy as np
import torch
from cleanfid import fid
from PIL import Image


def resize_image_to_256x256(image_path, output_path):
    """
    Resize image to 256x256. If smaller, pad with white background.
    If larger, crop from center. Also converts black pixels to white.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save resized image
    """
    img = Image.open(image_path)
    
    # Convert to RGB if not already (handles RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert black pixels (0,0,0) to white pixels (255,255,255)
    img_array = np.array(img)
    black_mask = (img_array[:, :, 0] == 0) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)
    img_array[black_mask] = [255, 255, 255]
    img = Image.fromarray(img_array)
    
    width, height = img.size
    target_size = 256
    if width == target_size and height == target_size:
        # Already correct size
        img.save(output_path)
        return
    
    if width < target_size or height < target_size:
        # Need to pad with white background
        new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        
        # Calculate position to paste the image (center it)
        paste_x = (target_size - width) // 2
        paste_y = (target_size - height) // 2
        
        new_img.paste(img, (paste_x, paste_y))
        new_img.save(output_path)
    else:
        # Need to crop from center
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)





class FIDEvaluator:
    """Evaluator for computing FID and KID scores between real and synthesized scenes."""
    
    def __init__(self, device=None, num_iterations=10):
        """Initialize the FID evaluator."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.num_iterations = num_iterations
    
    def prepare_images(self, path_to_images, temp_dir, num_images=None):
        """Prepare images by resizing and saving to temporary directory."""
        image_files = [
            os.path.join(path_to_images, f)
            for f in os.listdir(path_to_images)
            if f.endswith(".png")
        ]
        
        if num_images:
            np.random.shuffle(image_files)
            image_files = np.random.choice(image_files, num_images)
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        for i, img_path in enumerate(image_files):
            output_path = os.path.join(temp_dir, f"{i:05d}.png")
            resize_image_to_256x256(img_path, output_path)
            
        return len(image_files)
    
    def evaluate(self, path_to_real_renderings, path_to_synthesized_renderings, 
                 output_directory=None, temp_dir_base=None, verbose=True):
        """Evaluate FID and KID scores."""
        if verbose:
            print(f"Running FID/KID evaluation on {self.device}")
        
        if temp_dir_base is None:
            temp_dir_base = tempfile.gettempdir()
        
        temp_real_dir = os.path.join(temp_dir_base, "fid_eval_real")
        temp_fake_dir = os.path.join(temp_dir_base, "fid_eval_fake")
        
        # Prepare real images once
        if verbose:
            print("Preparing real images...")
        num_real_images = self.prepare_images(path_to_real_renderings, temp_real_dir)
        
        fid_scores = []
        kid_scores = []
        
        for iteration in range(self.num_iterations):
            if verbose:
                print(f"Iteration {iteration + 1}/{self.num_iterations}")
            
            # Prepare synthetic images for this iteration
            self.prepare_images(path_to_synthesized_renderings, temp_fake_dir, num_real_images)

            # Compute scores
            fid_score = fid.compute_fid(temp_real_dir, temp_fake_dir, device=self.device)
            kid_score = fid.compute_kid(temp_real_dir, temp_fake_dir, device=self.device)
            
            fid_scores.append(fid_score)
            kid_scores.append(kid_score)
            
            if verbose:
                print(f"  FID: {fid_score:.4f}, KID: {kid_score:.4f}")
            
            # Clean up fake directory for next iteration
            if os.path.exists(temp_fake_dir):
                shutil.rmtree(temp_fake_dir)
        
        # Calculate statistics
        fid_mean = sum(fid_scores) / len(fid_scores)
        fid_std = np.std(fid_scores)
        kid_mean = sum(kid_scores) / len(kid_scores)
        kid_std = np.std(kid_scores)
        
        if verbose:
            print(f"Final FID Score: {fid_mean:.4f} ± {fid_std:.4f}")
            print(f"Final KID Score: {kid_mean:.4f} ± {kid_std:.4f}")
        
        # Clean up temporary directories
        for temp_dir in [temp_real_dir, temp_fake_dir]:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        return {
            'fid_mean': fid_mean,
            'fid_std': fid_std,
            'fid_scores': fid_scores,
            'kid_mean': kid_mean,
            'kid_std': kid_std,
            'kid_scores': kid_scores,
            'num_iterations': self.num_iterations
        }



