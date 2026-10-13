import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model


IMG_SIZE = 64
CHANNELS = 3

# ---------------------------------------------------------
# 1. Patch extraction + linear patch embedding
# ---------------------------------------------------------
class PatchEmbedding(layers.Layer):
    def __init__(self, image_size=64, patch_size=8, embed_dim=64, **kwargs):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches = (image_size // patch_size) ** 2

        # Extract non-overlapping patches
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid"
        )

    def call(self, images):
        x = self.projection(images)

        # (batch, 8, 8, embed_dim)
        # -> (batch, 64, embed_dim)
        x = tf.reshape(
            x,
            [tf.shape(x)[0], self.num_patches, self.embed_dim]
        )

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
        })
        return config


# ---------------------------------------------------------
# 2. Learnable positional embedding
# ---------------------------------------------------------
class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True
        )

    def call(self, tokens):
        return tokens + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "embed_dim": self.embed_dim,
        })
        return config


# ---------------------------------------------------------
# 3. Small Transformer encoder block
# ---------------------------------------------------------
class TransformerBlock(layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads=4,
        mlp_dim=128,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # IMPORTANT:
        # key_dim is the dimension PER attention head.
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        # Self-attention + residual connection
        attention_output = self.mha(x, x)
        x = self.norm1(x + attention_output)

        # MLP + residual connection
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        })
        return config


# ---------------------------------------------------------
# 4. Build the small pure ViT
# ---------------------------------------------------------
def build_small_vit(
    image_size=64,
    patch_size=8,
    embed_dim=64,
    num_heads=4,
    mlp_dim=128,
    num_transformer_layers=2,
    dropout=0.1,
    num_classes=2
):

    num_patches = (image_size // patch_size) ** 2

    inputs = layers.Input(
        shape=(image_size, image_size, CHANNELS)
    )

    # Image -> patch tokens
    x = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim
    )(inputs)

    # Add positional information
    x = AddPositionEmbedding(
        num_patches=num_patches,
        embed_dim=embed_dim
    )(x)

    # Transformer encoder
    for _ in range(num_transformer_layers):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )(x)

    # Pool the token sequence
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return Model(inputs, outputs, name="small_vit")


def build_hybrid_model(pretrained_cnn_location):
    # ---------------------------------------------------------
    # 1. Load pretrained CNN
    # ---------------------------------------------------------

    cnn = tf.keras.models.load_model(pretrained_cnn_location)

    # Freeze the pretrained CNN
    cnn.trainable = False

    # ---------------------------------------------------------
    # 2. Create a new explicit input
    # ---------------------------------------------------------

    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, CHANNELS)
    )

    # ---------------------------------------------------------
    # 3. Pass the input through the pretrained CNN
    # ---------------------------------------------------------

    x = cnn(inputs)

    # ---------------------------------------------------------
    # 4. We DON'T want the CNN's final classifier.
    #
    # So instead, create a feature-extractor model
    # from the CNN's input to max_pooling2d_7.
    # ---------------------------------------------------------

    feature_extractor = Model(
        inputs=cnn.layers[0].input,
        outputs=cnn.get_layer("max_pooling2d_7").output
    )

    feature_extractor.trainable = False

    x = feature_extractor(inputs)

    # ---------------------------------------------------------
    # 5. Convert 4x4 feature map into 16 tokens
    #
    # (batch, 4, 4, 128)
    #       ↓
    # (batch, 16, 128)
    # ---------------------------------------------------------

    x = layers.Reshape((16, 128))(x)

    # ---------------------------------------------------------
    # 6. Positional embedding
    # ---------------------------------------------------------

    x = AddPositionEmbedding(
        num_patches=16,
        embed_dim=128
    )(x)

    # ---------------------------------------------------------
    # 7. Transformer block
    # ---------------------------------------------------------

    x = TransformerBlock(
        embed_dim=128,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1
    )(x)

    # ---------------------------------------------------------
    # 8. Classification head
    # ---------------------------------------------------------

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(
        2,
        activation="softmax"
    )(x)

    # ---------------------------------------------------------
    # 9. Build hybrid model
    # ---------------------------------------------------------

    hybrid_model = Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_vit_hybrid"
    )

    return hybrid_model
