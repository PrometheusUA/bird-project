#!/bin/bash
PROJECT_ID="bird-project-mlops-vertex"
REGION="us-central1"
REPOSITORY="bird-containers"
IMAGE_TAG="serving:latest"

# Create repository in the artifact registry
gcloud beta artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$REGION
 
# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
 
 # Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
