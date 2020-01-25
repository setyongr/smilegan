# Install tensorflow
FROM tensorflow/tensorflow:latest-gpu-py3
# OR
# FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /root

# Installs pandas, and google-cloud-storage.
RUN pip install google-cloud-storage

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "trainer/task.py"]