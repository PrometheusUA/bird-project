#!/bin/bash
PROJECT_ID="bird-project-mlops-vertex"
REGION="us-central1"
REPOSITORY="bird-containers"
IMAGE="training"
IMAGE_TAG="training:latest"

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG