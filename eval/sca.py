"""Scene Classification Accuracy evaluator for distinguishing real and synthesized scenes."""

import os
import numpy as np
import torch
from PIL import Image
from torchvision import models


def resize_image_to_256x256(image_path):
    """
    Resize image to 256x256. If smaller, pad with white background.
    If larger, crop from center. Also converts black pixels to white.
    Returns the processed PIL Image.
    
    Args:
        image_path (str): Path to input image
    
    Returns:
        PIL.Image: Processed image at 256x256
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
        return img
    
    if width < target_size or height < target_size:
        # Need to pad with white background
        new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        
        # Calculate position to paste the image (center it)
        paste_x = (target_size - width) // 2
        paste_y = (target_size - height) // 2
        
        new_img.paste(img, (paste_x, paste_y))
        return new_img
    else:
        # Need to crop from center
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, directory, train=True, real=True):
        images = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith("png")
            ]
        )
        if real:
            N = len(images)
        else:
            N = len(images) // 2

        start = 0 if train else N
        self.images = images[start : start + N]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]





class SyntheticVRealDataset(torch.utils.data.Dataset):
    def __init__(self, real, synthetic):
        self.N = min(len(real), len(synthetic))
        self.real = real
        self.synthetic = synthetic

    def __len__(self):
        return 2 * self.N

    def __getitem__(self, idx):
        if idx < self.N:
            image_path = self.real[idx]
            label = 1
        else:
            image_path = self.synthetic[idx - self.N]
            label = 0

        img = resize_image_to_256x256(image_path)
        img = np.asarray(img).astype(np.float32) / np.float32(255)
        img = np.transpose(img[:, :, :3], (2, 0, 1))

        return torch.from_numpy(img), torch.tensor([label], dtype=torch.float)


class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.alexnet(pretrained=True)
        self.fc = torch.nn.Linear(9216, 1)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = self.fc(x.view(len(x), -1))
        x = torch.sigmoid(x)

        return x


class AverageMeter:
    def __init__(self):
        self._value = 0
        self._cnt = 0

    def __iadd__(self, x):
        if torch.is_tensor(x):
            self._value += x.sum().item()
            self._cnt += x.numel()
        else:
            self._value += x
            self._cnt += 1
        return self

    @property
    def value(self):
        return self._value / self._cnt


class SceneClassificationAccuracyEvaluator:
    """Evaluator for scene classification accuracy between real and synthesized scenes."""
    
    def __init__(self, batch_size=256, num_workers=0, epochs=10, device=None):
        """Initialize the SCA evaluator."""
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    def create_datasets(self, path_to_train_renderings, path_to_test_renderings, path_to_synthesized_renderings):
        """Create train and test datasets from rendering directories."""
        train_real = ImageFolderDataset(path_to_train_renderings, True)
        test_real = ImageFolderDataset(path_to_test_renderings, True)
        train_synthetic = ImageFolderDataset(path_to_synthesized_renderings, True, False)
        test_synthetic = ImageFolderDataset(path_to_synthesized_renderings, False, False)

        train_dataset = SyntheticVRealDataset(train_real, train_synthetic)
        test_dataset = SyntheticVRealDataset(test_real, test_synthetic)
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, test_dataset):
        """Create data loaders from datasets."""
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        return train_dataloader, test_dataloader
    
    def train_and_evaluate(self, train_dataloader, test_dataloader, verbose=True):
        """Train model and evaluate classification accuracy."""
        model = AlexNet().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        scores = []
        for run in range(10):
            if verbose:
                print(f"Run {run + 1}/10")
            
            for e in range(self.epochs):
                # Training
                model.train()
                for x, y in train_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
                    loss.backward()
                    optimizer.step()

                # Evaluation every 5 epochs
                if (e + 1) % 5 == 0:
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for x, y in test_dataloader:
                            x, y = x.to(self.device), y.to(self.device)
                            y_hat = model(x)
                            correct += ((y_hat > 0.5) == (y > 0.5)).float().sum()
                            total += y.size(0)
                    
                    accuracy = correct / total
                    if verbose:
                        print(f"  Epoch {e+1}: {accuracy:.4f}")
            
            scores.append(accuracy.item())
        
        return scores
    
    def evaluate(self, path_to_train_renderings, path_to_test_renderings, path_to_synthesized_renderings, 
                 output_directory=None, verbose=True):
        """Evaluate scene classification accuracy."""
        if verbose:
            print(f"Running Scene Classification Accuracy evaluation on {self.device}")
        
        # Create datasets and data loaders
        train_dataset, test_dataset = self.create_datasets(
            path_to_train_renderings, path_to_test_renderings, path_to_synthesized_renderings
        )
        train_dataloader, test_dataloader = self.create_dataloaders(train_dataset, test_dataset)
        
        # Train and evaluate
        scores = self.train_and_evaluate(train_dataloader, test_dataloader, verbose)
        
        # Calculate statistics
        mean_accuracy = sum(scores) / len(scores)
        std_accuracy = np.std(scores)
        
        if verbose:
            print(f"Mean Classification Accuracy: {mean_accuracy:.4f}")
            print(f"Standard Deviation: {std_accuracy:.4f}")
        
        return {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'individual_scores': scores,
            'num_runs': len(scores)
        }


