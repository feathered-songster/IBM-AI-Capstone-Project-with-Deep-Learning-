from pathlib import Path

import torch
import torch.nn as nn

from src.CNN_model_definitions import create_tiny_model


model_path = Path("models") / "cnn_0_best.pth"

# Load the pretrained CNN
cnn = create_tiny_model()
cnn.load_state_dict(
    torch.load(model_path , map_location="cpu")
)

class CNNTransformer(nn.Module):
    def __init__(
        self,
        cnn,
        num_classes=2,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    ):
        super().__init__()

        # Take the pretrained CNN up to the second MaxPool2d
        self.cnn_feature_extractor = nn.Sequential(
            *list(cnn.children())[:8]
        )

        # Freeze CNN parameters
        for param in self.cnn_feature_extractor.parameters():
            param.requires_grad = False

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):

        # CNN
        x = self.cnn_feature_extractor(x)
        # [B, 32, 16, 16]

        # Spatial locations become tokens
        x = x.flatten(2)
        # [B, 32, 256]

        x = x.transpose(1, 2)
        # [B, 256, 32]

        # Transformer
        x = self.transformer(x)
        # [B, 256, 32]

        # Global average over spatial tokens
        x = x.mean(dim=1)
        # [B, 32]

        # Classification
        x = self.classifier(x)
        # [B, 2]

        return x

    def train(self, mode=True):
        super().train(mode)
        self.cnn_feature_extractor.eval()
        return self


def create_hybrid_model(num_classes=2):
    return CNNTransformer(cnn, num_classes)
