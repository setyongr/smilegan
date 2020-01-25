#!/bin/bash

echo "Training cloud ML model"

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=smilegan

# IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_TAG=latest

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=smilegan_job_$(date +%Y%m%d_%H%M%S)

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION=us-central1

# Build the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./

# Deploy the docker image to Cloud Container Registry
docker push ${IMAGE_URI}

# Submit your training job
echo "Submitting the training job"

# These variables are passed to the docker image
JOB_DIR=gs://${BUCKET_ID}/models

gcloud beta ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC_GPU \
    -- \
    --train-files ${TRAIN_FILES} \
    --eval-files ${EVAL_FILES} \
    --num-epochs=50 \
    --learning-rate=0.001 \
    --job-dir=${JOB_DIR}

echo "Job Sent"