from pathlib import Path
import random

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "production_models"
TEST_DIR = BASE_DIR / "data" / "test"


# -------------------------------------------------------------------
# Discover models and classes
# -------------------------------------------------------------------

MODEL_PATHS = sorted(MODEL_DIR.glob("*.onnx"))

if not MODEL_PATHS:
    raise RuntimeError(
        f"No ONNX models found in {MODEL_DIR}"
    )

MODEL_NAMES = [path.stem for path in MODEL_PATHS]

# Infer class names from the test directory.
# Expected structure:
#
# data/test/
#     class_a/
#     class_b/
#
CLASS_NAMES = sorted(
    path.name
    for path in TEST_DIR.iterdir()
    if path.is_dir()
)


# -------------------------------------------------------------------
# Discover test images
# -------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

TEST_IMAGES = []

for class_dir in TEST_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    for image_path in class_dir.rglob("*"):

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:

            TEST_IMAGES.append(
                {
                    "path": image_path,
                    "true_label": class_dir.name,
                }
            )


if len(TEST_IMAGES) < 5:
    raise RuntimeError(
        f"Need at least 5 test images in {TEST_DIR}"
    )


# -------------------------------------------------------------------
# ONNX Runtime sessions - one session per model
# -------------------------------------------------------------------

SESSIONS = {
    path.stem: ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    for path in MODEL_PATHS
}


# -------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------

def get_random_examples():

    selected = random.sample(TEST_IMAGES, 5)

    images = []

    for item in selected:

        image = Image.open(item["path"]).convert("RGB")

        images.append(image)

    return images


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------

def preprocess_image(image, session):

    image = image.convert("RGB")
    image = image.resize((64, 64))

    image = np.asarray(image).astype(np.float32)
    image = image / 255.0

    # Same normalization used during NN training:
    #
    # (x - 0.5) / 0.5
    #
    # [0, 1] -> [-1, 1]
    image = (image - 0.5) / 0.5

    input_shape = session.get_inputs()[0].shape

    # NCHW: [batch, channels, height, width]
    if input_shape[1] == 3:
        image = np.transpose(image, (2, 0, 1))

    # NHWC: [batch, height, width, channels]
    elif input_shape[-1] == 3:
        pass

    else:
        raise ValueError(
            f"Don't know how to handle input shape: {input_shape}"
        )
    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32)


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

def predict(image, model_name):

    if image is None:
        return "Please select or upload an image.", None

    session = SESSIONS[model_name]

    input_name = session.get_inputs()[0].name

    input_tensor = preprocess_image(image, session)

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        },
    )

    raw_output = np.asarray(outputs[0])

    # Expected shape is normally (1, number_of_classes)
    logits = raw_output[0]

    # Convert logits to probabilities.
    # This is the same softmax operation used during PyTorch
    # classification.
    logits = logits - np.max(logits)

    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum()

    predicted_index = int(np.argmax(probabilities))

    predicted_label = (
        CLASS_NAMES[predicted_index]
        if predicted_index < len(CLASS_NAMES)
        else str(predicted_index)
    )

    confidence = float(probabilities[predicted_index])

    result = (
        f"**Predicted label:** {predicted_label}\n\n"
        f"**Confidence:** {confidence:.3f} "
        f"({confidence:.1%})"
    )

    # Return probabilities as a small table
    probability_data = [
        [
            CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i),
            float(probabilities[i]),
        ]
        for i in range(len(probabilities))
    ]

    return result, probability_data


# -------------------------------------------------------------------
# Example selection
# -------------------------------------------------------------------

# Keep track of which test image corresponds to each displayed image.
example_selection = []


def load_examples():

    global example_selection

    example_selection = random.sample(TEST_IMAGES, 5)

    images = [
        Image.open(item["path"]).convert("RGB")
        for item in example_selection
    ]

    return images


def select_example(evt: gr.SelectData):

    index = evt.index

    if isinstance(index, tuple):
        index = index[0]

    if index >= len(example_selection):
        return None, "No example selected.", "Unknown"

    item = example_selection[index]

    image = Image.open(item["path"]).convert("RGB")

    return (
        image,
        f"Selected example — true label: **{item['true_label']}**",
    )



# -------------------------------------------------------------------
# Gradio interface
# -------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # Geospatial images classification

        Select one of the example satellite images below, or upload
        your own image.
        """
    )

    gr.Markdown("## Example images")

    example_gallery = gr.Gallery(
        value=load_examples,
        columns=5,
        rows=1,
        height="auto",
        label="Click an image to select it",
        show_label=True,
        object_fit="contain",
    )

    refresh_button = gr.Button("Load five new random examples")

    selected_image = gr.Image(
        type="pil",
        label="Selected or provided image",
        sources=["upload", "clipboard"],
    )

    selection_info = gr.Markdown()

    gr.Markdown("## Classification")

    model_dropdown = gr.Dropdown(
        choices=MODEL_NAMES,
        value=MODEL_NAMES[0],
        label="Model",
    )

    predict_button = gr.Button(
        "Classify image",
        variant="primary",
    )

    result = gr.Markdown()

    probability_table = gr.Dataframe(
        headers=["Class", "Probability"],
        datatype=["str", "number"],
        label="Class probabilities",
        interactive=False,
    )

    gr.Markdown("--- Capstone project for [IBM Deep Learning with PyTorch, Keras and Tensorflow Professional Certificate](https://www.coursera.org/specializations/ibm-deep-learning-with-pytorch-keras-tensorflow)")

    # ---------------------------------------------------------------
    # Events
    # ---------------------------------------------------------------

    example_gallery.select(
        select_example,
        outputs=[
            selected_image,
            selection_info,
        ],
    )

    refresh_button.click(
        load_examples,
        outputs=example_gallery,
    )

    predict_button.click(
        predict,
        inputs=[
            selected_image,
            model_dropdown,
        ],
        outputs=[
            result,
            probability_table,
        ],
    )


# -------------------------------------------------------------------
# Start Gradio
# -------------------------------------------------------------------

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
)


