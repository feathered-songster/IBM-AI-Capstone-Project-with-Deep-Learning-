# IBM AI Capstone Project with Deep Learning

This repository contains the capstone project for the [IBM Deep Learning with PyTorch, Keras and TensorFlow Professional Certificate](https://www.coursera.org/specializations/ibm-deep-learning-with-pytorch-keras-tensorflow) on Coursera.

The project demonstrates the development, evaluation, export, and serving of deep learning image-classification models using **PyTorch**, **TensorFlow/Keras**, **ONNX**, and **Gradio**.
It explores several model architectures: **CNNs, Vision Transformers, and a hybrid CNN-Transformer model** and also demonstrates **transfer learning**, using a pre-trained CNN as part of the hybrid architecture with its learned layers frozen.

The trained models are exported to **ONNX** and served through a separate **Gradio** application using ONNX Runtime.

## Project structure

The project is organized into three independent Docker environments:

```text
.
├── data/
├── PyTorch/
├── TensorFlow/
└── GradioApp/
```

* **PyTorch** — model development, training and evaluation using PyTorch and JupyterLab.
* **TensorFlow** — model development, training and evaluation using TensorFlow/Keras and JupyterLab.
* **GradioApp** — inference application using ONNX Runtime and Gradio.
* **data** — common dataset used by all three environments.

Each application has its own `Dockerfile` and Docker Compose configuration.

The containers are separate environments rather than a single automated pipeline.

## Running the project

The project is intended to run in Docker.

### PyTorch

```bash
cd PyTorch
docker compose up --build
```

Open the JupyterLab URL beginning with `http://127.0.0.1:8888/` printed in the terminal. The complete URL contains the access token required to log in.

### TensorFlow

```bash
cd TensorFlow
docker compose up --build
```

Again, open the JupyterLab URL beginning with `http://127.0.0.1:8888/` printed in the terminal.

### Gradio

```bash
cd GradioApp
docker compose up --build
```

Once the container is running, open:

```text
http://localhost:7860
```

The Gradio application allows users to select an available model, choose one of the example images, upload an image, or paste an image from the clipboard, and then run inference.

## ONNX models

The Gradio application loads its ONNX models from:

```text
GradioApp/
└── production_models/
```

For the current version of the project, the exported ONNX files are manually copied into this directory. This represents the model-artifact handoff between the training environments and the serving application.

In a production setup, these models would instead be retrieved from an **artifact/model repository** as part of the deployment process.

## Stopping a container

From the corresponding project directory:

```bash
docker compose down
```

The individual environments can be started and stopped independently.
