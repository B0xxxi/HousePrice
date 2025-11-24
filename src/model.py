import torch
import torch.nn as nn
from torchvision import models

class HousePriceModel(nn.Module):
    def __init__(self, pretrained=True):
        super(HousePriceModel, self).__init__()
        # Use ResNet18 for a good balance of speed and performance
        # We can switch to ResNet50 if needed
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the last fully connected layer
        # ResNet18 has 'fc' as the last layer with 512 input features
        num_features = self.backbone.fc.in_features
        
        # We want to predict a single value (price)
        # We can add some extra layers if we want, but a single linear layer is a good start
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.backbone(x)
