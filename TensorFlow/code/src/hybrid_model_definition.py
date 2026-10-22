import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

from .ViT_model_definitions import TransformerBlock, AddPositionEmbedding, PatchEmbedding, IMG_SIZE, CHANNELS

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
