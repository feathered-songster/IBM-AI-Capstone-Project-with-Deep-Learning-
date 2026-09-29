## IBM AI Capstone Project with Deep Learning

This is the capstone project of the IBM Deep Learning with PyTorch, Keras and Tensorflow Professional Certificate on Coursera.
https://www.coursera.org/learn/ai-deep-learning-capstone?specialization=ibm-deep-learning-with-pytorch-keras-tensorflow 

The project contains two folders - on for PyTorch and one for TensorFlow. They each contain their own docker file and compose file. The data directory is common and mounted by both containers. 

## Running

The project is intended to run in Docker. Pick the version you would like to run and from the respective subfolder start the respective container with:

```bash
docker compose up --build
```

The local code and data folders will be bind mounted in the container, which runs the jupyter lab. 
Click the link beginning with http://127.0.0.1:8888/ from the terminal where you started the container so that the generated access token is used. This will open the jupyterlab environment in your browser.