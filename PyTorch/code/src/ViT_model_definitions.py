import torch
import torch.nn as nn


class SmallViT(nn.Module):
    def __init__(
        self,
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        num_heads=8,
        mlp_dim=64,
        num_layers=2,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        # 1. Convert image patches into embeddings
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # 2. Learn one positional embedding for every patch
        num_patches = (image_size // patch_size) ** 2

        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        # 3. Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 4. Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        # x: (batch, 3, 64, 64)
        x = self.patch_embedding(x)

        # x: (batch, 64, 64)
        x = x + self.position_embedding

        for block in self.transformer_blocks:
            x = block(x)

        # Global average pooling over the 64 tokens
        x = x.mean(dim=1)

        # x: (batch, 64)
        x = self.dropout(x)

        # x: (batch, 2)
        x = self.classifier(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            image_size=64,
            patch_size=8,
            in_channels=3,
            embed_dim=64,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Linear(
            in_channels * patch_size * patch_size,
            embed_dim
        )

    def forward(self, x):
        B, C, H, W = x.shape

        P = self.patch_size

        # Divide image into patches
        x = x.unfold(2, P, P).unfold(3, P, P)

        # (B, C, 8, 8, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5)

        # (B, 8, 8, C, P, P)
        x = x.reshape(B, self.num_patches, C * P * P)

        # (B, 64, 192) → (B, 64, 64)
        x = self.projection(x)

        return x


##########################################################
# This is demo code, if writing code to solve a problem efficiently
# please use nn.TransformerEncoder, not manual code
# nn.TransformerEncoder is essentially a stack of the Transformer blocks we wrote manually.
# See the implemetation of the hybrid model for anexample.
##########################################################

class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim=64,
            num_heads=8,
            mlp_dim=64,
            dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # ---- Self-attention ----

        residual = x

        x_norm = self.norm1(x)

        attention_output, attention_weights = self.attention(
            x_norm,
            x_norm,
            x_norm,
            need_weights=True,
        )

        x = residual + self.dropout1(attention_output)

        # ---- Feed-forward network ----

        residual = x

        x_norm = self.norm2(x)

        x = residual + self.mlp(x_norm)

        return x