# Install tensorflow
FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /root

# Installs pandas, and google-cloud-storage.
RUN pip install google-cloud-storage
RUN pip install scikit-image

# Copies the trainer code to the docker image.
COPY trainer/evaluator.py ./trainer/evaluator.py
COPY trainer/fid_calculator.py ./trainer/fid_calculator.py
COPY trainer/input.py ./trainer/input.py
COPY trainer/model.py ./trainer/model.py
COPY trainer/network.py ./trainer/network.py
COPY trainer/task.py ./trainer/task.py
COPY main.py ./main.py

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "main.py"]