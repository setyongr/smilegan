# Install tensorflow
FROM tensorflow/tensorflow:latest-gpu-py3
# OR
# FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /root

# Installs pandas, and google-cloud-storage.
RUN pip install google-cloud-storage

# Copies the trainer code to the docker image.
COPY trainer/input.py ./trainer/input.py
COPY trainer/model.py ./trainer/model.py
COPY trainer/network.py ./trainer/network.py
COPY trainer/task.py ./trainer/task.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "trainer/task.py"]