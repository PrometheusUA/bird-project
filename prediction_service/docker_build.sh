#!/bin/bash
PROJECT_ID="bird-project-mlops-vertex"
REGION="us-central1"
REPOSITORY="bird-containers"
IMAGE='serving'
IMAGE_TAG='serving:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG