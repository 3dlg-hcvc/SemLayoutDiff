from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torchmetrics
import json
import math
import numpy as np


def classify_angle(angle, num_classes=8):
    if num_classes == 4:
        class_label = int((angle + 45) // 90) % 4
    else:
        class_label = int((angle + 22.5) // 45) % 8
    return class_label


def face_inward_orientation(x, y, num_classes=8):
    angle = math.degrees(math.atan2(y, x))
    # Adjust the angle so that it's measured from the line y = -x
    adjusted_angle = (angle + 45) % 360
    return classify_angle(adjusted_angle, num_classes)


# Semantic Encoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = F.relu(self.conv1(x))
        # out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SemanticEncoder(LightningModule):
    def __init__(self, config):
        super(SemanticEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, config.conv1, kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResidualBlock, config.conv1, config.conv2, stride=2)
        self.layer2 = self._make_layer(ResidualBlock, config.conv2, config.conv3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.dropout(x)
        # return x.view(x.size(0), -1)  # Flattening the output
        return x


class InstanceEncoder(LightningModule):
    def __init__(self, config):
        super(InstanceEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, config.conv1, kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResidualBlock, config.conv1, config.conv2, stride=2)
        self.layer2 = self._make_layer(ResidualBlock, config.conv2, config.conv3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.dropout(x)
        # return x.view(x.size(0), -1)  # Flattening the output
        return x


class CrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(y).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out


# Category ID Encoder
class CategoryEncoder(LightningModule):
    def __init__(self, config):
        super(CategoryEncoder, self).__init__()
        self.embedding = nn.Embedding(config.num_categories, config.embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# Prediction Network
class PredictionNetwork(LightningModule):
    def __init__(self, config):
        super(PredictionNetwork, self).__init__()

        # Shared layers
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_shared1 = nn.Linear(64 * 16 * 16,
                                    config.prediction_network.fc1)
        self.fc_shared2 = nn.Linear(config.prediction_network.fc1, config.prediction_network.fc2)

        # Separate final layers for size, offset, and orientation
        self.fc_size = nn.Linear(config.prediction_network.fc2, 3)
        self.fc_offset = nn.Linear(config.prediction_network.fc2, 1)
        self.fc_orient = nn.Linear(config.prediction_network.fc2, config.num_orientation_class)

    def forward(self, x):
        # x = x.view(1, -1)
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        x = F.relu(self.fc_shared1(x))
        x = F.relu(self.fc_shared2(x))

        size_pred = self.fc_size(x)
        offset_pred = self.fc_offset(x)
        orient_pred = self.fc_orient(x)

        return size_pred, offset_pred, orient_pred


# Combined Model
class FurnitureAttributesModel(LightningModule):
    def __init__(self, config):
        super(FurnitureAttributesModel, self).__init__()
        self.semantic_encoder = SemanticEncoder(config.model.semantic_encoder)
        self.instance_encoder = InstanceEncoder(
            config.model.semantic_encoder)  # Using same config for instance encoder for simplicity
        self.cross_attention = CrossAttention(config.model.semantic_encoder.conv3)
        self.category_fc = nn.Linear(config.model.num_categories, config.model.embedding_dim)
        self.prediction_network = PredictionNetwork(config.model)
        self.learning_rate = config.trainer.lr
        self.l1_lambda = config.trainer.l1_lambda
        self.l2_lambda = config.trainer.l2_lambda

        self.orientation_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.size_mae = torchmetrics.MeanAbsoluteError()
        self.offset_mae = torchmetrics.MeanAbsoluteError()

        self.baseline_method = config.get('baseline_method', None)
        if self.baseline_method == 'category_majority':
            with open(config.category_majority_json_path, 'r') as file:
                self.category_majority_mapping = json.load(file)

        # self.start = torch.cuda.Event(enable_timing=True)
        # self.end = torch.cuda.Event(enable_timing=True)

    def forward(self, semantic_map, instance_mask, category):
        # Process the first semantic map (since they are all the same)
        semantic_features = self.semantic_encoder(semantic_map)
        instance_features = self.instance_encoder(instance_mask)
        # category_features = self.category_fc(category)

        # combined_features = torch.cat([semantic_features, instance_features, category_features], dim=1)
        combined_features = self.cross_attention(semantic_features, instance_features)
        size_pred, offset_pred, orient_pred = self.prediction_network(combined_features)

        return size_pred, offset_pred, orient_pred

    @staticmethod
    def calculate_centroids(instance_mask):
        x_indices, y_indices = np.where(instance_mask == 1)
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)

        scale = 0.01
        centroid_x = -(instance_mask.shape[1] / 2 - centroid_x) * scale
        centroid_y = (centroid_y - instance_mask.shape[0] / 2) * scale

        return centroid_x, centroid_y

    def training_step(self, batch, batch_idx):
        # return None
        semantic_map, instance_mask, category, attribute = batch

        size_prediction, offset_prediction, orientation_prediction = self(semantic_map, instance_mask, category)

        gt_size, gt_offset, gt_orientation = attribute[0], attribute[1], attribute[2]

        size_loss, offset_loss = F.mse_loss(size_prediction, gt_size), F.mse_loss(offset_prediction, gt_offset)

        # L1 regularization
        l1_lambda = self.l1_lambda
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l1_loss = l1_lambda * l1_norm

        total_loss = size_loss + offset_loss + (
            0 if self.baseline_method else F.cross_entropy(orientation_prediction, gt_orientation)) + l1_loss

        loss_types = ['size', 'offset', 'orient', 'total']
        losses = [size_loss, offset_loss, F.cross_entropy(orientation_prediction, gt_orientation),
                  total_loss]

        for name, loss in zip(loss_types, losses):
            self.log(f'train_{name}_loss', loss, on_step=True, on_epoch=True)

        orientation_prediction = torch.argmax(orientation_prediction, dim=-1)
        # print("\n", orientation_prediction, gt_orientation)

        self.log('train_orient_acc', self.orientation_accuracy(orientation_prediction, gt_orientation), on_step=True,
                 on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        semantic_map, instance_mask, category, attribute = batch
        # semantic_map, instance_mask, category = semantic_map.unsqueeze(1).float(), instance_mask.unsqueeze(
        #     1).float(), category.float()

        size_prediction, offset_prediction, orientation_prediction = self(semantic_map, instance_mask, category)

        gt_size, gt_offset, gt_orientation = attribute[0], attribute[1], attribute[2]

        orientation_loss = F.cross_entropy(orientation_prediction, gt_orientation)

        self.log('val_orient_loss', orientation_loss, on_step=True, on_epoch=True)

        orientation_prediction = torch.argmax(orientation_prediction, dim=-1)

        self.log('val_orient_acc', self.orientation_accuracy(orientation_prediction, gt_orientation), on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)

        # Define the number of warm-up epochs
        warmup_epochs = 3

        # Define the lambda function for warm-up
        lambda1 = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1

        # Create the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.3)
        return [optimizer], [scheduler]
